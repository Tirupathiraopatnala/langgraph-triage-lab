"""The full agent and its system of record."""

import pytest

from triage_lab.agent.service import TriageService
from triage_lab.agent.store import StoreError, TriageStore
from triage_lab.fabric import FAKE_NOTEBOOK, FakeFabric, recent_failures


@pytest.fixture
def service(tmp_path):
    return TriageService(data_dir=tmp_path)


def incident_of(service, item_type):
    return next(i for i in service.scan() if i["item_type"] == item_type)


# --- store ------------------------------------------------------------------------


def test_incident_id_survives_a_sliding_window(tmp_path):
    store = TriageStore(tmp_path / "t.sqlite")
    failure = next(f for f in recent_failures(FakeFabric()) if f.item.type == "Notebook")
    first = store.upsert_incident(failure)
    # The window moved: the oldest failed run dropped out. Same item, same incident.
    later = failure.model_copy(update={"runs": failure.runs[-1:]})
    assert store.upsert_incident(later)["id"] == first["id"]


def test_status_changes_are_compare_and_swap(tmp_path):
    store = TriageStore(tmp_path / "t.sqlite")
    failure = recent_failures(FakeFabric())[0]
    incident = store.upsert_incident(failure)
    plan = store.new_plan(incident["id"], "test")
    store.finish_analysis(plan["id"], "proposed", "x", {})
    assert store.transition(plan["id"], "proposed", "approved", "alice")
    assert not store.transition(plan["id"], "proposed", "approved", "bob")  # second click loses
    with pytest.raises(StoreError):
        store.finish_analysis(plan["id"], "proposed", "y", {})  # plans are written once


def test_reanalysis_supersedes_the_old_proposal(service):
    incident = incident_of(service, "Notebook")
    v1 = service.analyze(incident["id"])
    v2 = service.analyze(incident["id"], note="look again")
    assert service.records.plan(v1["id"])["status"] == "superseded"
    assert v2["status"] == "proposed"
    with pytest.raises(StoreError):
        service.approve(v1["id"], actor="alice")


# --- graph + service ------------------------------------------------------------------


def test_pipeline_failure_is_handed_to_the_notebook_specialist(service):
    plan = service.analyze(incident_of(service, "DataPipeline")["id"])
    assert plan["status"] == "proposed"
    assert [e["specialist"] for e in plan["body"]["case_file"]] == ["pipeline", "notebook"]
    assert plan["body"]["patches"][0]["part_path"] == "notebook-content.py"


def test_approve_applies_once(service):
    plan = service.analyze(incident_of(service, "Notebook")["id"])
    result = service.approve(plan["id"], actor="alice")
    assert result["status"] == "applied"
    assert service.fabric.list_job_instances("", FAKE_NOTEBOOK) == []  # the fake recovered
    with pytest.raises(StoreError):
        service.approve(plan["id"], actor="bob")


def test_reject_ends_the_thread_and_writes_nothing(service):
    plan = service.analyze(incident_of(service, "Notebook")["id"])
    assert service.reject(plan["id"], actor="alice", reason="not today")["status"] == "rejected"
    state = service.app.get_state({"configurable": {"thread_id": plan["thread_id"]}})
    assert state.next == ()  # the thread finished instead of waiting forever
    assert not any(c.startswith("update_definition") for c in service.fabric.calls)


def test_apply_refuses_if_the_code_changed_while_waiting(service):
    plan = service.analyze(incident_of(service, "Notebook")["id"])
    service.fabric._definitions[FAKE_NOTEBOOK]["notebook-content.py"] += "\n# edited by someone\n"
    result = service.approve(plan["id"], actor="alice")
    assert result["status"] == "apply_failed"
    assert "REFUSED" in result["outcome"]["lines"][0]


def test_edit_applies_the_operators_text(service):
    plan = service.analyze(incident_of(service, "Notebook")["id"])
    mine = plan["body"]["patches"][0]["after"].replace("sales_bronze", "customers")
    service.approve(plan["id"], actor="alice", after=mine)
    assert service.fabric._definitions[FAKE_NOTEBOOK]["notebook-content.py"] == mine


def test_writes_disabled_is_a_dry_run(tmp_path):
    import dataclasses

    from triage_lab.settings import get_settings

    service = TriageService(data_dir=tmp_path)
    service.context.settings = dataclasses.replace(service.settings, writes_enabled=False)
    plan = service.analyze(incident_of(service, "Notebook")["id"])
    result = service.approve(plan["id"], actor="alice")
    assert result["status"] == "approved" and result["outcome"]["dry_run"]
    assert get_settings().writes_enabled is False


def test_memory_turns_a_past_fix_into_a_lead(tmp_path):
    first = TriageService(data_dir=tmp_path)
    plan = first.analyze(incident_of(first, "Notebook")["id"])
    first.approve(plan["id"], actor="alice")

    second = TriageService(data_dir=tmp_path)  # new process, fake workspace fails again
    again = second.analyze(incident_of(second, "Notebook")["id"])
    assert again["body"]["memory_used"] and "sales_bronze" in again["body"]["memory_used"]


def test_credentials_failure_becomes_manual_steps(service):
    plan = service.analyze(incident_of(service, "SemanticModel")["id"])
    assert plan["status"] == "manual"
    assert plan["body"]["manual_steps"] and not plan["body"]["patches"]


def test_hop_budget_stops_the_case(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERVISOR_MAX_HOPS", "1")
    from triage_lab.settings import get_settings

    service = TriageService(get_settings(), data_dir=tmp_path)
    plan = service.analyze(incident_of(service, "DataPipeline")["id"])
    assert [e["specialist"] for e in plan["body"]["case_file"]] == ["pipeline"]
    assert plan["status"] == "manual"  # nothing to apply: the notebook specialist never ran
