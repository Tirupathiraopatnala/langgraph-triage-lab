"""The real Fabric client, against a mock Fabric (httpx.MockTransport)."""

import base64
import json

import httpx

from triage_lab.contracts import Item
from triage_lab.fabric import FabricError, RealFabric, decode_definition, parse_time, with_patched_part
from triage_lab.settings import Settings

WS = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
NB = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


class FakeCredential:
    def get_token(self, scope):
        return type("Token", (), {"token": f"token-for-{scope}"})()


def b64(text):
    return base64.b64encode(text.encode()).decode()


def client(handler, **settings):
    real = RealFabric(Settings(fabric_mode="real", workspace_ids=[WS], **settings),
                      http=httpx.Client(transport=httpx.MockTransport(handler)), credential=FakeCredential())
    real.sleep = lambda seconds: None
    return real


def test_parse_time_handles_fabric_fractions():
    parsed = parse_time("2026-10-30T13:08:11.6247281")
    assert parsed.microsecond == 624728 and parsed.tzinfo is not None


def test_429_is_retried_after_waiting():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        assert request.headers["Authorization"] == "Bearer token-for-https://api.fabric.microsoft.com/.default"
        return httpx.Response(200, json={"value": [{"id": NB, "displayName": "nb", "type": "Notebook"}]})

    assert client(handler).list_items(WS)[0].name == "nb" and calls["n"] == 2


def test_get_definition_polls_the_long_running_operation():
    source = "print('hi')\n"

    def handler(request):
        path = request.url.path
        if path.endswith(f"/items/{NB}/getDefinition"):
            return httpx.Response(202, headers={"x-ms-operation-id": "op-1"})
        if path == "/v1/operations/op-1":
            return httpx.Response(200, json={"status": "Succeeded"})
        if path == "/v1/operations/op-1/result":
            parts = [{"path": "notebook-content.py", "payload": b64(source), "payloadType": "InlineBase64"}]
            return httpx.Response(200, json={"definition": {"parts": parts}})
        if path == f"/v1/workspaces/{WS}/items":
            return httpx.Response(200, json={"value": [{"id": NB, "displayName": "nb", "type": "Notebook"}]})
        return httpx.Response(404)

    definition = client(handler).read_definition(WS, NB)
    assert definition.parts == {"notebook-content.py": source}


def test_403_is_not_transient():
    real = client(lambda request: httpx.Response(403, text="no"))
    try:
        real.list_items(WS)
    except FabricError as exc:
        assert exc.status == 403 and not exc.transient
    else:
        raise AssertionError("expected FabricError")


def test_writes_fail_closed():
    real = client(lambda request: httpx.Response(200))
    try:
        real.update_definition(WS, NB, {"parts": []})
    except FabricError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("expected writes to be refused")


def test_patched_definition_round_trips():
    raw = {"parts": [{"path": "a.py", "payload": b64("old"), "payloadType": "InlineBase64"}]}
    patched = with_patched_part(raw, "a.py", "new")
    assert decode_definition(Item(id=NB, name="nb", type="Notebook"), {"definition": patched}).parts["a.py"] == "new"
    assert json.loads(json.dumps(raw))["parts"][0]["payload"] == b64("old")  # original untouched


def test_scan_skips_items_whose_job_history_is_refused():
    from datetime import UTC, datetime, timedelta

    from triage_lab.fabric import find_failures

    model = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    start = (datetime.now(UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%f0")

    def handler(request):
        path = request.url.path
        if path == f"/v1/workspaces/{WS}":
            return httpx.Response(200, json={"displayName": "Dev"})
        if path == f"/v1/workspaces/{WS}/items":
            return httpx.Response(200, json={"value": [
                {"id": NB, "displayName": "nb", "type": "Notebook"},
                {"id": model, "displayName": "sm", "type": "SemanticModel"},
            ]})
        if path.endswith(f"/items/{model}/jobs/instances"):
            return httpx.Response(404, text="not supported for this item")
        if path.endswith(f"/items/{NB}/jobs/instances"):
            return httpx.Response(200, json={"value": [{
                "id": "run-1", "status": "Failed", "startTimeUtc": start, "jobType": "RunNotebook",
                "failureReason": {"errorCode": "X", "message": "[TABLE_OR_VIEW_NOT_FOUND] boom"},
            }]})
        return httpx.Response(404)

    failures = find_failures(client(handler), [WS], datetime.now(UTC) - timedelta(days=1))
    assert [f.item.name for f in failures] == ["nb"]
    assert failures[0].signature == "Notebook:TABLE_OR_VIEW_NOT_FOUND"
