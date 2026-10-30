"""Every lesson runs end to end and shows what it claims to show."""

import importlib

from triage_lab.fabric import FAKE_WS


def run(name, *args):
    return importlib.import_module(f"lessons.{name}").main(*args)


def test_l01_state_graph():
    assert "does not exist" in run("l01_state_graph")["summary"]


def test_l02_reducers_and_routing():
    actions = {r["failure"].item.type: r["action"] for r in run("l02_reducers_and_routing")}
    assert "credentials" in actions["SemanticModel"]
    assert "code fix" in actions["Notebook"]
    assert all(len(r["notes"]) == 2 for r in run("l02_reducers_and_routing"))  # both branches merged


def test_l02_missing_reducer_raises():
    from lessons.l02_reducers_and_routing import demo_missing_reducer

    assert "one value per step" in demo_missing_reducer()


def test_l03_structured_output():
    results = run("l03_structured_output")
    assert {r["triage"].category for r in results} == {"missing_table", "credentials"}


def test_l04_agent_loop_reads_then_answers():
    messages = run("l04_tools_and_agent_loop")["messages"]
    called = [c["name"] for m in messages for c in getattr(m, "tool_calls", [])]
    assert called == ["list_workspace_items", "read_item_definition", "list_lakehouse_tables"]
    assert "sales_bronze" in messages[-1].content


def test_l05_retries_and_error_handler():
    clean, retried, handled = run("l05_context_retries_errors")
    assert retried["notes"] == ["preload: definition read"]
    assert "gave up" in handled["notes"][0]
    assert "sales_bronze" in handled["messages"][-1].content  # still answered


def test_l06_time_travel_fork():
    assert "lakehouse specialist" in run("l06_checkpoints_time_travel")["plan"]


def test_l07_approve_writes_and_clears_failure():
    final = run("l07_human_approval", "approve")
    assert "written to Fabric" in final["outcome"]


def test_l07_reject_writes_nothing():
    final = run("l07_human_approval", "reject")
    assert "outcome" not in final


def test_l08_fan_out_investigates_every_failure():
    final = run("l08_send_and_subgraphs")
    assert sorted(e.specialist for e in final["entry"]) == ["notebook", "pipeline", "semantic_model"]


def test_l09_supervisor_reroutes_then_patches():
    final = run("l09_supervisor_command")
    assert [e.specialist for e in final["case_file"]] == ["pipeline", "notebook"]
    assert final["patches"] and final["hops"] == 2


def test_l10_memory_hit_on_second_run():
    first, second = run("l10_store_memory")
    assert first["recalled"] is None and second["recalled"] is not None


def test_l11_streaming_sees_every_mode():
    counts = run("l11_streaming")
    assert counts["updates"] and counts["custom"] and counts["messages"]


def test_l12_demo_runs(capsys):
    run("l12_triage_agent")
    out = capsys.readouterr().out
    assert "written. -> applied" in out and "second approval refused" in out and FAKE_WS not in out
