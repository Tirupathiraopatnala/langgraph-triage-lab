"""Tests always run offline: the fake workspace and the scripted model.

Set before anything imports triage_lab, so a filled-in .env can't make the tests call
real Fabric or Azure OpenAI (python-dotenv never overrides variables already set).
"""

import os

os.environ["FABRIC_MODE"] = "fake"
os.environ["MODEL_MODE"] = "fake"
os.environ["FABRIC_WRITES_ENABLED"] = "false"
os.environ["SUPERVISOR_MAX_HOPS"] = "4"
os.environ["SPECIALIST_MAX_TOOL_CALLS"] = "8"
