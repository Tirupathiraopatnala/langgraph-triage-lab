"""Command line for the triage agent.

    python -m triage_lab.agent scan                       find failures, open/refresh incidents
    python -m triage_lab.agent analyze INC-0001 [--note]  run the team; pauses for approval
    python -m triage_lab.agent approve INC-0001/v1 --by tiru [--after-file new.py]
    python -m triage_lab.agent reject  INC-0001/v1 --by tiru --reason "..."
    python -m triage_lab.agent show INC-0001              incident, plans, full history
    python -m triage_lab.agent demo                       the whole loop in one go

Note: with FABRIC_MODE=fake the fake workspace starts fresh in every process, so a fix
approved in one command is not remembered by the fake in the next. `demo` runs the whole
loop in one process so you can see the failure disappear after the fix.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from triage_lab.agent.service import TriageService
from triage_lab.agent.store import StoreError
from triage_lab.settings import REPO_ROOT


def _progress(who: str, what: str) -> None:
    print(f"   [{who}] {what}")


def print_incidents(incidents: list[dict]) -> None:
    if not incidents:
        print("No failures in the lookback window.")
    for i in incidents:
        print(f"{i['id']}  {i['status']:<8} {i['item_type']:<13} {i['item_name']:<18} "
              f"{i['failed_runs']} run(s)  {i['latest_error'][:70]}")


def print_plan(plan: dict) -> None:
    body = plan["body"] or {}
    print(f"\nPlan {plan['id']}  status={plan['status']}  hops={body.get('hops')}")
    if body.get("memory_used"):
        print("Memory lead:", body["memory_used"])
    print("Root cause:", body.get("root_cause"))
    for entry in body.get("case_file", []):
        print(f"  - {entry['specialist']}: {entry['outcome']}"
              + (f" -> handed to {entry['handoff_to']}" if entry.get("handoff_to") else ""))
    for patch in body.get("patches", []):
        print(f"\nPatch {patch['item_name']}/{patch['part_path']}: {patch['explanation']}\n{patch['diff']}")
    for step in body.get("manual_steps", []):
        print("Manual step:", step)
    if body.get("unresolved"):
        print("Unresolved:", body["unresolved"])


def print_history(service: TriageService, incident_id: str) -> None:
    shown = service.show(incident_id)
    print(f"\nHistory of {incident_id}:")
    for a in shown["history"]:
        print(f"  {a['at']}  {a['kind']:<18} {a['actor']:<8} {a['plan_id'] or '':<12} {a['detail'][:70]}")


def demo() -> None:
    data = REPO_ROOT / "data" / "demo"
    shutil.rmtree(data, ignore_errors=True)
    service = TriageService(data_dir=data)

    print("1. Scan")
    incidents = service.scan()
    print_incidents(incidents)

    target = next(i for i in incidents if i["item_type"] == "DataPipeline")
    print(f"\n2. Analyse {target['id']} ({target['item_name']})")
    plan = service.analyze(target["id"], actor="demo", on_event=_progress)
    print_plan(plan)

    print(f"\n3. Approve {plan['id']} (and try to approve it twice)")
    result = service.approve(plan["id"], actor="tiru")
    print("   ", " ".join(result["outcome"]["lines"]), "->", result["status"])
    try:
        service.approve(plan["id"], actor="someone-else")
    except StoreError as exc:
        print("    second approval refused:", exc)

    print("\n4. Scan again (the fake reruns the jobs after a fix)")
    print_incidents(service.scan())
    print_history(service, target["id"])

    other = next(i for i in incidents if i["item_type"] == "SemanticModel")
    print(f"\n5. Analyse {other['id']} ({other['item_name']}): no code fix, manual steps")
    print_plan(service.analyze(other["id"], actor="demo", on_event=_progress))
    service.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan")
    p = sub.add_parser("analyze")
    p.add_argument("incident_id")
    p.add_argument("--note", help="What the operator wants the team to reconsider")
    p = sub.add_parser("approve")
    p.add_argument("plan_id")
    p.add_argument("--by", required=True)
    p.add_argument("--after-file", type=Path, help="Use this file's text as the first patch instead")
    p = sub.add_parser("reject")
    p.add_argument("plan_id")
    p.add_argument("--by", required=True)
    p.add_argument("--reason", default="")
    p = sub.add_parser("show")
    p.add_argument("incident_id")
    sub.add_parser("demo")
    args = parser.parse_args(argv)

    if args.command == "demo":
        demo()
        return 0
    service = TriageService()
    try:
        if args.command == "scan":
            print_incidents(service.scan())
        elif args.command == "analyze":
            print_plan(service.analyze(args.incident_id, note=args.note, on_event=_progress))
        elif args.command == "approve":
            after = args.after_file.read_text(encoding="utf-8") if args.after_file else None
            result = service.approve(args.plan_id, actor=args.by, after=after)
            print(" ".join(result["outcome"]["lines"]), "->", result["status"])
        elif args.command == "reject":
            print(service.reject(args.plan_id, actor=args.by, reason=args.reason)["status"])
        elif args.command == "show":
            shown = service.show(args.incident_id)
            print_incidents([shown["incident"]])
            for plan in shown["plans"]:
                print_plan(plan)
            print_history(service, args.incident_id)
    except StoreError as exc:
        print("Refused:", exc, file=sys.stderr)
        return 2
    finally:
        service.close()
    return 0
