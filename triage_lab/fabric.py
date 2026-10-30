"""Talking to Microsoft Fabric: one interface, a real client and a fake workspace.

The lessons only ever use the `FabricApi` interface, so the same graph runs against
your real workspace (FABRIC_MODE=real) or the built-in fake one (FABRIC_MODE=fake).

Real-client notes worth knowing for interviews:
- Auth is a token for scope https://api.fabric.microsoft.com/.default, from a service
  principal (ClientSecretCredential) or DefaultAzureCredential (az login / managed identity).
- getDefinition and updateDefinition are long-running: Fabric answers 202 with an
  operation id, and you poll /v1/operations/{id} until it succeeds, then read /result.
- Fabric throttles hard (429 with Retry-After). This client honours Retry-After itself.
  Other failures are raised as FabricError, and the graph decides whether to retry
  (lesson 5). One layer owns each kind of retry, never two.
"""

from __future__ import annotations

import base64
import copy
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

from triage_lab.contracts import Definition, FailedRun, Failure, Item
from triage_lab.settings import Settings, get_settings

FABRIC_API = "https://api.fabric.microsoft.com"
FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"
ONELAKE_DFS = "https://onelake.dfs.fabric.microsoft.com"
STORAGE_SCOPE = "https://storage.azure.com/.default"

# Item types that have job history worth scanning.
JOB_ITEM_TYPES = {"Notebook", "DataPipeline", "SparkJobDefinition", "SemanticModel", "Dataflow"}
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}
log = logging.getLogger("triage_lab.fabric")


class FabricError(RuntimeError):
    """A Fabric call failed. `transient` says whether trying again could help."""

    def __init__(self, message: str, *, status: int | None = None, transient: bool = False):
        super().__init__(message)
        self.status = status
        self.transient = transient


@runtime_checkable
class FabricApi(Protocol):
    """Everything the agents may ask of Fabric. Only `update_definition` writes."""

    def workspace_name(self, workspace_id: str) -> str: ...
    def list_items(self, workspace_id: str) -> list[Item]: ...
    def list_job_instances(self, workspace_id: str, item_id: str) -> list[dict[str, Any]]: ...
    def read_definition(self, workspace_id: str, item_id: str) -> Definition: ...
    def list_lakehouse_tables(self, workspace_id: str, lakehouse_id: str) -> list[str]: ...
    def update_definition(self, workspace_id: str, item_id: str, raw: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Helpers shared by both implementations
# ---------------------------------------------------------------------------


def parse_time(raw: str | None) -> datetime | None:
    """Fabric returns UTC timestamps like 2026-10-30T13:08:11.6247281 (7 fraction digits)."""
    if not raw:
        return None
    text = raw.strip().rstrip("Zz")
    text = re.sub(r"\.(\d{6})\d+", r".\1", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def decode_definition(item: Item, payload: dict[str, Any]) -> Definition:
    """Fabric returns each part base64-encoded. Decode the text parts."""
    raw = payload.get("definition") or payload
    parts: dict[str, str] = {}
    for part in raw.get("parts") or []:
        if part.get("payloadType") != "InlineBase64":
            continue
        try:
            parts[part["path"]] = base64.b64decode(part["payload"]).decode("utf-8")
        except (KeyError, ValueError, UnicodeDecodeError):
            continue  # binary parts (images, .platform blobs) are skipped
    return Definition(item=item, parts=parts, raw=raw)


def with_patched_part(raw: dict[str, Any], part_path: str, new_text: str) -> dict[str, Any]:
    """A copy of a raw definition with one part replaced, ready for updateDefinition."""
    patched = copy.deepcopy(raw)
    for part in patched.get("parts") or []:
        if part.get("path") == part_path:
            part["payload"] = base64.b64encode(new_text.encode("utf-8")).decode("ascii")
            part["payloadType"] = "InlineBase64"
            return patched
    raise FabricError(f"Definition has no part {part_path}")


def find_failures(fabric: FabricApi, workspace_ids: list[str], since: datetime) -> list[Failure]:
    """Every item that failed since `since`, one Failure per item, newest first.

    Fabric has no workspace-wide job listing, so this asks each job-capable item for
    its history. That is many calls, which is why the real client paces itself.
    """
    failures: list[Failure] = []
    for workspace_id in workspace_ids:
        workspace_name = fabric.workspace_name(workspace_id)
        for item in fabric.list_items(workspace_id):
            if item.type not in JOB_ITEM_TYPES:
                continue
            try:
                jobs = fabric.list_job_instances(workspace_id, item.id)
            except FabricError as exc:
                if exc.transient:
                    raise
                # Some item types don't expose job history to every caller. Skip, don't abort the scan.
                log.warning("no job history for %s (%s): %s", item.name, item.type, exc)
                continue
            runs = []
            for job in jobs:
                started = parse_time(job.get("startTimeUtc"))
                if str(job.get("status", "")).lower() != "failed" or not started or started < since:
                    continue
                reason = job.get("failureReason") or {}
                runs.append(
                    FailedRun(
                        job_instance_id=str(job.get("id")),
                        job_type=job.get("jobType"),
                        started_at=started,
                        ended_at=parse_time(job.get("endTimeUtc")),
                        error_code=reason.get("errorCode"),
                        error_message=reason.get("message"),
                    )
                )
            if runs:
                failures.append(
                    Failure(workspace_id=workspace_id, workspace_name=workspace_name, item=item, runs=runs)
                )
    failures.sort(key=lambda f: f.latest.started_at, reverse=True)
    return failures


# ---------------------------------------------------------------------------
# The real client
# ---------------------------------------------------------------------------


class RealFabric:
    """Fabric REST client for a real workspace."""

    def __init__(self, settings: Settings, *, http: httpx.Client | None = None, credential: Any = None):
        """`http` and `credential` can be injected, which is how the tests fake Fabric."""
        self._settings = settings
        if credential is not None:
            self._credential = credential
        elif settings.tenant_id and settings.client_id and settings.client_secret:
            from azure.identity import ClientSecretCredential

            self._credential = ClientSecretCredential(settings.tenant_id, settings.client_id, settings.client_secret)
        else:
            from azure.identity import DefaultAzureCredential

            self._credential = DefaultAzureCredential()
        self._http = http or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        self._items: dict[str, list[Item]] = {}
        self.sleep = time.sleep  # swapped out in tests so they don't wait

    def close(self) -> None:
        self._http.close()

    # -- plumbing -------------------------------------------------------------

    def _request(self, method: str, url: str, *, scope: str = FABRIC_SCOPE, **kwargs: Any) -> httpx.Response:
        for attempt in range(4):
            token = self._credential.get_token(scope).token  # azure-identity caches it
            try:
                response = self._http.request(
                    method, url, headers={"Authorization": f"Bearer {token}"}, **kwargs
                )
            except httpx.TransportError as exc:
                raise FabricError(f"{method} {url}: network error {exc}", transient=True) from exc
            if response.status_code == 429 and attempt < 3:
                # Fabric tells us how long to back off. Honour it, then try again.
                wait = float(response.headers.get("Retry-After", "5") or 5)
                self.sleep(min(wait, 60))
                continue
            break
        if response.status_code >= 400:
            raise FabricError(
                f"{method} {urlparse(url).path}: HTTP {response.status_code} {response.text[:300]}",
                status=response.status_code,
                transient=response.status_code in TRANSIENT_STATUSES,
            )
        return response

    def _get(self, path: str) -> dict[str, Any]:
        response = self._request("GET", FABRIC_API + path)
        return response.json() if response.content else {}

    def _post_long_running(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """POST, and if Fabric answers 202, poll the operation until it finishes."""
        response = self._request("POST", FABRIC_API + path, json=body)
        if response.status_code != 202:
            return response.json() if response.content else {}
        operation_id = response.headers.get("x-ms-operation-id") or urlparse(
            response.headers.get("Location", "")
        ).path.rstrip("/").rsplit("/", 1)[-1]
        delay, deadline = 1.5, time.monotonic() + 600
        while time.monotonic() < deadline:
            status = str(self._get(f"/v1/operations/{operation_id}").get("status", "")).lower()
            if status == "succeeded":
                try:
                    return self._get(f"/v1/operations/{operation_id}/result")
                except FabricError:
                    return {}  # updateDefinition succeeds with no result body
            if status in {"failed", "cancelled", "canceled"}:
                raise FabricError(f"Fabric operation {operation_id} {status}")
            self.sleep(delay)
            delay = min(delay * 1.5, 15)
        raise FabricError(f"Fabric operation {operation_id} timed out", transient=True)

    # -- the interface ------------------------------------------------------

    def workspace_name(self, workspace_id: str) -> str:
        return str(self._get(f"/v1/workspaces/{workspace_id}").get("displayName") or workspace_id)

    def list_items(self, workspace_id: str) -> list[Item]:
        if workspace_id not in self._items:
            items, path = [], f"/v1/workspaces/{workspace_id}/items"
            while path:
                page = self._get(path)
                items += [Item(id=i["id"], name=i["displayName"], type=i["type"]) for i in page.get("value", [])]
                token = page.get("continuationToken")
                path = f"/v1/workspaces/{workspace_id}/items?continuationToken={token}" if token else ""
            self._items[workspace_id] = items
        return self._items[workspace_id]

    def list_job_instances(self, workspace_id: str, item_id: str) -> list[dict[str, Any]]:
        jobs, path = [], f"/v1/workspaces/{workspace_id}/items/{item_id}/jobs/instances"
        while path:
            page = self._get(path)
            jobs += page.get("value", [])
            token = page.get("continuationToken")
            path = f"/v1/workspaces/{workspace_id}/items/{item_id}/jobs/instances?continuationToken={token}" if token else ""
        return jobs

    def read_definition(self, workspace_id: str, item_id: str) -> Definition:
        item = next((i for i in self.list_items(workspace_id) if i.id == item_id), None)
        if item is None:
            raise FabricError(f"No item {item_id} in workspace {workspace_id}", status=404)
        payload = self._post_long_running(f"/v1/workspaces/{workspace_id}/items/{item_id}/getDefinition")
        return decode_definition(item, payload)

    def list_lakehouse_tables(self, workspace_id: str, lakehouse_id: str) -> list[str]:
        """Folder names under Tables/ in OneLake.

        For a schema-enabled lakehouse these are schema names (dbo, ...), not tables;
        list one level deeper if you use schemas.
        """
        response = self._request(
            "GET",
            f"{ONELAKE_DFS}/{workspace_id}",
            scope=STORAGE_SCOPE,
            params={"resource": "filesystem", "recursive": "false", "directory": f"{lakehouse_id}/Tables"},
        )
        names = [p.get("name", "").rsplit("/", 1)[-1] for p in response.json().get("paths", [])]
        return sorted(n for n in names if n)

    def update_definition(self, workspace_id: str, item_id: str, raw: dict[str, Any]) -> None:
        if not self._settings.writes_allowed_for(workspace_id):
            raise FabricError("Writes are disabled for this workspace (FABRIC_WRITES_ENABLED / allowlist).")
        self._post_long_running(
            f"/v1/workspaces/{workspace_id}/items/{item_id}/updateDefinition", {"definition": raw}
        )


# ---------------------------------------------------------------------------
# The fake workspace
# ---------------------------------------------------------------------------

FAKE_WS = "11111111-1111-4111-8111-111111111111"
FAKE_LAKEHOUSE = "22222222-2222-4222-8222-222222222222"
FAKE_NOTEBOOK = "33333333-3333-4333-8333-333333333333"
FAKE_PIPELINE = "44444444-4444-4444-8444-444444444444"
FAKE_MODEL = "55555555-5555-4555-8555-555555555555"

_NOTEBOOK_SOURCE = '''# Fabric notebook source

# CELL ********************

from pyspark.sql import functions as F

raw = spark.read.table("lh_sales.sales_raw")
clean = raw.filter(F.col("amount") > 0).withColumn("loaded_at", F.current_timestamp())
clean.write.mode("overwrite").saveAsTable("lh_sales.sales_silver")
'''

_PIPELINE_SOURCE = f'''{{
  "properties": {{
    "activities": [
      {{
        "name": "Run nb_load_sales",
        "type": "TridentNotebook",
        "typeProperties": {{"notebookId": "{FAKE_NOTEBOOK}", "workspaceId": "{FAKE_WS}"}}
      }}
    ]
  }}
}}
'''

_MODEL_SOURCE = """model Model
    culture: en-US

expression SqlEndpoint = Sql.Database("lh-sales.datawarehouse.fabric.microsoft.com", "lh_sales")
"""

_TABLE_ERROR = (
    "[TABLE_OR_VIEW_NOT_FOUND] The table or view `lh_sales`.`sales_raw` cannot be found. "
    "Verify the spelling and correctness of the schema and catalog."
)


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


class FakeFabric:
    """A pretend workspace with three realistic failures. Same interface as RealFabric.

    `flaky_reads=N` makes the first N reads fail with a transient 503, so you can watch
    retry policies work (lesson 5).
    """

    def __init__(self, settings: Settings | None = None, *, flaky_reads: int = 0):
        self._settings = settings or get_settings()
        self.flaky_reads = flaky_reads
        self.calls: list[str] = []  # what the agents asked, for tests and curiosity
        now = datetime.now(UTC)
        self._items = [
            Item(id=FAKE_LAKEHOUSE, name="lh_sales", type="Lakehouse"),
            Item(id=FAKE_NOTEBOOK, name="nb_load_sales", type="Notebook"),
            Item(id=FAKE_PIPELINE, name="pl_daily_sales", type="DataPipeline"),
            Item(id=FAKE_MODEL, name="sm_sales", type="SemanticModel"),
        ]
        self._definitions = {
            FAKE_NOTEBOOK: {"notebook-content.py": _NOTEBOOK_SOURCE},
            FAKE_PIPELINE: {"pipeline-content.json": _PIPELINE_SOURCE},
            FAKE_MODEL: {"definition/model.tmdl": _MODEL_SOURCE},
            FAKE_LAKEHOUSE: {},
        }
        self._tables = {FAKE_LAKEHOUSE: ["customers", "dim_date", "sales_bronze"]}

        def run(item_id: str, hours_ago: float, code: str, message: str, job_type: str) -> dict:
            start = now - timedelta(hours=hours_ago)
            return {
                "id": f"run-{item_id[:4]}-{int(hours_ago * 10)}",
                "itemId": item_id,
                "jobType": job_type,
                "status": "Failed",
                "startTimeUtc": start.strftime("%Y-%m-%dT%H:%M:%S.%f0"),
                "endTimeUtc": (start + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%S.%f0"),
                "failureReason": {"errorCode": code, "message": message},
            }

        self._jobs: dict[str, list[dict]] = {
            FAKE_NOTEBOOK: [
                run(FAKE_NOTEBOOK, 5, "NotebookExecutionFailed", _TABLE_ERROR, "RunNotebook"),
                run(FAKE_NOTEBOOK, 2, "NotebookExecutionFailed", _TABLE_ERROR, "RunNotebook"),
            ],
            FAKE_PIPELINE: [
                run(
                    FAKE_PIPELINE, 1.5, "ActivityFailed",
                    "Operation on target Run nb_load_sales failed: Notebook execution failed. " + _TABLE_ERROR,
                    "Pipeline",
                )
            ],
            FAKE_MODEL: [
                run(
                    FAKE_MODEL, 1, "DM_GWPipeline_Gateway_DataSourceAccessError",
                    "Refresh failed: the credentials provided for the SqlEndpoint data source are "
                    "invalid or expired. Update the data source credentials in the semantic model settings.",
                    "Refresh",
                )
            ],
        }

    def _touch(self, what: str) -> None:
        self.calls.append(what)
        if self.flaky_reads > 0:
            self.flaky_reads -= 1
            raise FabricError(f"{what}: HTTP 503 Service Unavailable (fake)", status=503, transient=True)

    def workspace_name(self, workspace_id: str) -> str:
        return "Sales Dev (fake)" if workspace_id == FAKE_WS else workspace_id

    def list_items(self, workspace_id: str) -> list[Item]:
        self._touch("list_items")
        return list(self._items) if workspace_id == FAKE_WS else []

    def list_job_instances(self, workspace_id: str, item_id: str) -> list[dict[str, Any]]:
        self.calls.append(f"list_job_instances:{item_id[:8]}")
        return list(self._jobs.get(item_id, []))

    def read_definition(self, workspace_id: str, item_id: str) -> Definition:
        self._touch(f"read_definition:{item_id[:8]}")
        item = next((i for i in self._items if i.id == item_id), None)
        if item is None:
            raise FabricError(f"No item {item_id}", status=404)
        parts = self._definitions[item_id]
        raw = {"parts": [{"path": p, "payload": _b64(t), "payloadType": "InlineBase64"} for p, t in parts.items()]}
        return decode_definition(item, {"definition": raw})

    def list_lakehouse_tables(self, workspace_id: str, lakehouse_id: str) -> list[str]:
        self._touch(f"list_lakehouse_tables:{lakehouse_id[:8]}")
        return list(self._tables.get(lakehouse_id, []))

    def update_definition(self, workspace_id: str, item_id: str, raw: dict[str, Any]) -> None:
        if not self._settings.writes_allowed_for(workspace_id):
            raise FabricError("Writes are disabled for this workspace (FABRIC_WRITES_ENABLED / allowlist).")
        self.calls.append(f"update_definition:{item_id[:8]}")
        for part in raw.get("parts", []):
            self._definitions[item_id][part["path"]] = base64.b64decode(part["payload"]).decode("utf-8")
        # Pretend the next scheduled run succeeds: the item and anything that runs it recover.
        name = next(i.name for i in self._items if i.id == item_id)
        for key, jobs in self._jobs.items():
            if key == item_id or any(name in (j["failureReason"]["message"]) for j in jobs):
                self._jobs[key] = []


def get_fabric(settings: Settings | None = None) -> FabricApi:
    """The Fabric client chosen by FABRIC_MODE."""
    settings = settings or get_settings()
    if settings.fabric_mode == "real":
        if not settings.workspace_ids:
            raise ValueError("FABRIC_MODE=real needs FABRIC_WORKSPACE_IDS in .env")
        return RealFabric(settings)
    return FakeFabric(settings)


def workspace_ids(settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    return settings.workspace_ids if settings.fabric_mode == "real" else [FAKE_WS]


def recent_failures(fabric: FabricApi | None = None, settings: Settings | None = None) -> list[Failure]:
    """Convenience for the lessons: failures in the configured lookback window."""
    settings = settings or get_settings()
    fabric = fabric or get_fabric(settings)
    since = datetime.now(UTC) - timedelta(hours=settings.lookback_hours)
    return find_failures(fabric, workspace_ids(settings), since)


def pick_failure(item_type: str | None = None, fabric: FabricApi | None = None,
                 settings: Settings | None = None) -> Failure:
    """One failure for a lesson to work on: the first of `item_type` if there is one, else the newest."""
    failures = recent_failures(fabric, settings)
    if not failures:
        raise SystemExit(
            "No failed runs in the lookback window. Widen FABRIC_LOOKBACK_HOURS, "
            "or use FABRIC_MODE=fake to run against the built-in workspace."
        )
    match = next((f for f in failures if f.item.type == item_type), None)
    if match is None and item_type:
        print(f"(No failing {item_type} found; using {failures[0].item.name} ({failures[0].item.type}) instead.)")
    return match or failures[0]
