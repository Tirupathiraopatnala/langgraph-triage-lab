"""The chat model: Azure OpenAI in Foundry, or a scripted stand-in.

`get_model()` is the only place a model is created. MODEL_MODE=azure gives you a real
deployment; MODEL_MODE=fake gives you `FakeTriageModel`, which reads the same prompts
and answers with simple rules. The fake exists so every lesson runs offline, free and
deterministically (and so the tests can run in CI). It is not clever, and it only
really understands the fake workspace's three failures.

Azure notes worth knowing:
- The v1 endpoint is <resource>/openai/v1 and takes the DEPLOYMENT name as `model`.
- Auth is an API key, or an Entra token from DefaultAzureCredential. The token
  provider is a callable, so the SDK fetches a fresh token whenever it needs one.
- The OpenAI SDK retries 429/5xx itself (max_retries). Don't add a second retry layer
  around model calls.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterator, Sequence
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ValidationError

from triage_lab.settings import Settings, get_settings


def foundry_v1_url(endpoint: str) -> str:
    """Turn whatever the portal gave you into <resource>/openai/v1."""
    url = endpoint.strip().rstrip("/")
    for suffix in ("/openai/v1/responses", "/openai/v1/chat/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)] + "/openai/v1"
    return url if url.endswith("/openai/v1") else url + "/openai/v1"


def get_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.model_mode == "fake":
        return FakeTriageModel()
    if settings.model_mode != "azure":
        raise ValueError(f"MODEL_MODE must be 'fake' or 'azure', not {settings.model_mode!r}")
    if not (settings.azure_endpoint and settings.azure_deployment):
        raise ValueError("MODEL_MODE=azure needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT")

    from langchain_openai import ChatOpenAI

    if settings.azure_api_key:
        api_key: Any = settings.azure_api_key
    else:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        api_key = get_bearer_token_provider(DefaultAzureCredential(), settings.azure_scope)
    return ChatOpenAI(
        model=settings.azure_deployment,  # the deployment name, not the model name
        base_url=foundry_v1_url(settings.azure_endpoint),
        api_key=api_key,
        use_responses_api=settings.azure_use_responses,
        timeout=90,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# The scripted stand-in
# ---------------------------------------------------------------------------

_ITEM_LINE = re.compile(r"Item: (?P<name>.+?) \((?P<type>[^,]+), id (?P<id>[0-9a-fA-F-]{36})\)")
_DEFINITION = re.compile(r"^## (?P<name>.+?) \((?P<type>[^,]+), id (?P<id>[0-9a-fA-F-]{36})\)$", re.M)
_PART = re.compile(r"^=== (?P<path>.+?) ===\n(?P<body>.*?)(?=^=== |^## |\Z)", re.M | re.S)
_TABLES = re.compile(r"Tables in lakehouse (?P<id>[0-9a-fA-F-]{36}): (?P<names>.+)")
_READ_TABLE = re.compile(r"spark\.read\.table\(\s*[\"'](?P<table>[\w.]+)[\"']\s*\)")
_NOTEBOOK_REF = re.compile(r'"notebookId"\s*:\s*"(?P<id>[0-9a-fA-F-]{36})"')
_CASE_LINE = re.compile(r"^\d+\. (?P<who>\w+) answered (?P<outcome>\w+): (?P<cause>.*)$", re.M)
_HANDOFF = re.compile(r"handed to (?P<who>\w+)")

_OWNER_BY_TYPE = {
    "Notebook": "notebook",
    "DataPipeline": "pipeline",
    "Lakehouse": "lakehouse",
    "SemanticModel": "semantic_model",
}


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


class _Case:
    """What the fake can work out from the conversation so far."""

    def __init__(self, messages: Sequence[BaseMessage]):
        self.messages = list(messages)
        self.all = "\n".join(_text(m) for m in self.messages)
        prompts = "\n".join(_text(m) for m in self.messages if not isinstance(m, ToolMessage))
        item = _ITEM_LINE.search(prompts) or _ITEM_LINE.search(self.all)
        self.item = item.groupdict() if item else {"name": "?", "type": "?", "id": ""}
        error = re.search(r"^Error: (.+)$", prompts, re.M) or re.search(r"^Error: (.+)$", self.all, re.M)
        self.error = error.group(1) if error else ""
        system = next((_text(m) for m in self.messages if isinstance(m, SystemMessage)), "")
        me = re.search(r"You are the (\w+) specialist", system)
        self.me = me.group(1) if me else None
        self.called = [c["name"] for m in self.messages if isinstance(m, AIMessage) for c in m.tool_calls]

        # Definitions are parsed one message at a time, so a part never runs into the
        # next message's text.
        self.definitions: dict[str, dict[str, str]] = {}
        for text in (_text(m) for m in self.messages):
            for match in _DEFINITION.finditer(text):
                nxt = _DEFINITION.search(text, match.end())
                block = text[match.end() : nxt.start() if nxt else len(text)]
                self.definitions[match["id"]] = {
                    p["path"]: p["body"].rstrip("\n") + "\n"
                    for p in _PART.finditer(block)
                    if not p["path"].startswith("end of ")
                }
        self.tables: dict[str, list[str]] = {
            m["id"]: [n.strip() for n in m["names"].split(",") if n.strip() and n.strip() != "(none)"]
            for m in _TABLES.finditer(self.all)
        }
        self.lakehouses = re.findall(r"^([0-9a-fA-F-]{36}) \| Lakehouse \| (\S+)$", self.all, re.M)

    # -- facts --------------------------------------------------------------

    @property
    def category(self) -> str:
        error = self.error.lower()
        if "table_or_view_not_found" in error:
            return "missing_table"
        if "credential" in error:
            return "credentials"
        if "429" in error or "capacity" in error or "throttl" in error:
            return "throttling"
        return "other"

    @property
    def owner(self) -> str:
        return _OWNER_BY_TYPE.get(self.item["type"], "general")

    def missing_table(self) -> str | None:
        found = re.search(r"`([\w]+)`\.`([\w]+)` cannot be found", self.error)
        return f"{found.group(1)}.{found.group(2)}" if found else None

    def notebook_part_reading(self, table: str) -> tuple[str, str, str] | None:
        """(item_id, part_path, source) of a read definition that reads `table`."""
        for item_id, parts in self.definitions.items():
            for path, body in parts.items():
                if any(t == table for t in _READ_TABLE.findall(body)):
                    return item_id, path, body
        return None

    def closest_table(self, table: str) -> str | None:
        short = table.split(".")[-1]
        names = [n for names in self.tables.values() for n in names]
        match = difflib.get_close_matches(short, names, n=1, cutoff=0.4)
        return match[0] if match else None

    # -- tool policy --------------------------------------------------------

    def next_tool_call(self, tools: list[str]) -> dict[str, Any] | None:
        def want(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
            return {"name": name, "args": args, "id": f"call_{uuid4().hex[:10]}"} if name in tools else None

        if "list_workspace_items" not in self.called and (call := want("list_workspace_items", {})):
            return call
        if self.item["id"] and self.item["id"] not in self.definitions:
            if call := want("read_item_definition", {"item_id": self.item["id"]}):
                return call
        for parts in self.definitions.values():
            for body in parts.values():
                for ref in _NOTEBOOK_REF.findall(body):
                    if ref not in self.definitions and (call := want("read_item_definition", {"item_id": ref})):
                        return call
        table = self.missing_table()
        if table and not self.tables and self.lakehouses:
            prefix = table.split(".")[0]
            lakehouse = next((i for i, name in self.lakehouses if name == prefix), self.lakehouses[0][0])
            if call := want("list_lakehouse_tables", {"lakehouse_id": lakehouse}):
                return call
        return None

    # -- answers ------------------------------------------------------------

    def summary(self) -> str:
        table = self.missing_table()
        if table:
            fix = self.closest_table(table)
            known = ", ".join(n for names in self.tables.values() for n in names)
            subject = f"{self.item['name']} fails because " + (
                "the notebook it runs reads" if self.item["type"] == "DataPipeline" else "it reads"
            )
            text = f"{subject} `{table}`, which does not exist."
            if known:
                text += f" The lakehouse has: {known}."
            if fix:
                text += f" The closest match is `{fix}`, so the read should probably point there."
            return text
        if self.category == "credentials":
            return (
                f"{self.item['name']} cannot refresh because its data source credentials are invalid "
                "or expired. Someone with access must update them in the semantic model settings."
            )
        return f"The latest error on {self.item['name']} was: {self.error or 'not reported'}."

    def report(self) -> dict[str, Any]:
        me = self.me or self.owner
        table = self.missing_table()
        evidence = [self.error] if self.error else []
        if me == "pipeline" and table:
            return {
                "outcome": "not_mine",
                "root_cause": "The pipeline only runs nb_load_sales; the error was raised inside that notebook.",
                "evidence": evidence,
                "handoff_to": "notebook",
                "handoff_reason": f"The notebook activity failed on missing table `{table}`.",
            }
        if me == "notebook" and table:
            found = self.notebook_part_reading(table)
            fix = self.closest_table(table)
            if found and fix:
                item_id, path, source = found
                new_table = f"{table.split('.')[0]}.{fix}"
                return {
                    "outcome": "resolved",
                    "root_cause": f"The notebook reads `{table}`, which does not exist; the data is in `{new_table}`.",
                    "evidence": evidence + [f'spark.read.table("{table}")', f"Lakehouse tables include {fix}"],
                    "patches": [
                        {
                            "item_id": item_id,
                            "part_path": path,
                            "after": source.replace(f'"{table}"', f'"{new_table}"'),
                            "explanation": f"Read `{new_table}` instead of the missing `{table}`.",
                        }
                    ],
                }
            return {
                "outcome": "partial",
                "root_cause": f"The notebook reads `{table}`, which does not exist.",
                "evidence": evidence,
                "needs_from_other": "List the lakehouse tables to find where the data lives.",
            }
        if self.category == "credentials" and me in {"semantic_model", "general"}:
            return {
                "outcome": "resolved",
                "root_cause": "The semantic model's data source credentials are invalid or expired.",
                "evidence": evidence,
                "manual_steps": [
                    "Open the semantic model settings in the Fabric portal.",
                    "Under Data source credentials, edit the SqlEndpoint source and sign in again.",
                    "Run a manual refresh to confirm.",
                ],
            }
        if me != self.owner:
            return {
                "outcome": "not_mine",
                "root_cause": f"This failure belongs to the {self.owner} specialist.",
                "evidence": evidence,
                "handoff_to": self.owner,
                "handoff_reason": f"The failing item is a {self.item['type']}.",
            }
        return {"outcome": "partial", "root_cause": self.summary(), "evidence": evidence}

    def facts(self, schema_name: str) -> dict[str, Any]:
        facts: dict[str, Any] = {
            "category": self.category,
            "specialist": self.owner,
            "summary": self.summary(),
            "reason": f"The failing item is a {self.item['type']} and the error is {self.category}.",
            "briefing": f"Investigate {self.item['name']}. Error: {self.error}",
            "confidence": "high" if self.category != "other" else "low",
        }
        if schema_name == "Review":
            cases = list(_CASE_LINE.finditer(self.all))
            last = cases[-1] if cases else None
            handoff = _HANDOFF.findall(self.all)
            if last and last["outcome"] == "not_mine" and handoff:
                facts.update(decision="reroute", next_specialist=handoff[-1],
                             reason=f"{last['who']} says this is {handoff[-1]}'s case.",
                             briefing=f"{last['who']} found: {last['cause']}")
            else:
                facts.update(decision="accept", reason="The report is backed by quoted evidence.")
        elif schema_name == "Conclusion":
            cases = [c for c in _CASE_LINE.finditer(self.all) if c["outcome"] == "resolved"]
            steps = re.findall(r"manual steps: (.+)", self.all)
            facts.update(
                root_cause=cases[-1]["cause"] if cases else self.summary(),
                evidence=[self.error] if self.error else [],
                manual_steps=steps[-1].split(" | ") if steps else [],
                unresolved=None if cases else "No specialist could close the case.",
            )
        else:
            facts.update(self.report())
        return facts


class FakeTriageModel(BaseChatModel):
    """A rule-based stand-in for a real model. See the module docstring."""

    bound_tools: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-triage"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeTriageModel:  # type: ignore[override]
        names = [getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools]
        return self.model_copy(update={"bound_tools": names})

    def with_structured_output(self, schema: Any, **kwargs: Any):  # type: ignore[override]
        def answer(model_input: Any) -> BaseModel:
            messages = self._convert_input(model_input).to_messages()
            facts = _Case(messages).facts(schema.__name__)
            data = {k: v for k, v in facts.items() if k in schema.model_fields}
            try:
                return schema.model_validate(data)
            except ValidationError as exc:
                raise ValueError(
                    f"FakeTriageModel has no good answer for {schema.__name__}. "
                    "Set MODEL_MODE=azure to use a real model."
                ) from exc

        return RunnableLambda(answer)

    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        case = _Case(messages)
        if self.bound_tools and (call := case.next_tool_call(self.bound_tools)):
            return AIMessage(content="", tool_calls=[call])
        return AIMessage(content=case.summary())

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._decide(messages))])

    def _stream(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        decided = self._decide(messages)
        if decided.tool_calls:
            chunks = [{"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                      for i, c in enumerate(decided.tool_calls)]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for word in re.findall(r"\S+\s*", str(decided.content)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word))
            if run_manager:
                run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk
