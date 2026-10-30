"""Settings, read once from the environment (and .env if present).

Every lesson and the agent read configuration through `get_settings()`, so you
switch between the fake and the real world in one place: FABRIC_MODE and MODEL_MODE.
Secrets are held here and never printed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


def _csv(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    return [part.strip() for part in raw.split(",") if part.strip()]


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # Fabric
    fabric_mode: str = "fake"  # "fake" or "real"
    workspace_ids: list[str] = field(default_factory=list)
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = field(default="", repr=False)
    lookback_hours: int = 24
    writes_enabled: bool = False
    write_workspace_ids: list[str] = field(default_factory=list)

    # Model
    model_mode: str = "fake"  # "fake" or "azure"
    azure_endpoint: str = ""
    azure_deployment: str = ""
    azure_api_key: str = field(default="", repr=False)
    azure_scope: str = "https://cognitiveservices.azure.com/.default"
    azure_use_responses: bool = True

    # Agent
    db_path: str = "data/triage.sqlite"
    max_hops: int = 4
    max_tool_calls: int = 8

    def writes_allowed_for(self, workspace_id: str) -> bool:
        """Fail closed: writes need the switch on AND the workspace on the allowlist."""
        return self.writes_enabled and workspace_id in self.write_workspace_ids


def get_settings() -> Settings:
    return Settings(
        fabric_mode=os.getenv("FABRIC_MODE", "fake").strip().lower() or "fake",
        workspace_ids=_csv("FABRIC_WORKSPACE_IDS"),
        tenant_id=os.getenv("FABRIC_TENANT_ID", "").strip(),
        client_id=os.getenv("FABRIC_CLIENT_ID", "").strip(),
        client_secret=os.getenv("FABRIC_CLIENT_SECRET", "").strip(),
        lookback_hours=_int("FABRIC_LOOKBACK_HOURS", 24),
        writes_enabled=_flag("FABRIC_WRITES_ENABLED"),
        write_workspace_ids=_csv("FABRIC_WRITE_WORKSPACE_IDS"),
        model_mode=os.getenv("MODEL_MODE", "fake").strip().lower() or "fake",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip(),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
        azure_scope=os.getenv("AZURE_OPENAI_SCOPE", "").strip()
        or "https://cognitiveservices.azure.com/.default",
        azure_use_responses=_flag("AZURE_OPENAI_USE_RESPONSES", default=True),
        db_path=os.getenv("TRIAGE_DB_PATH", "").strip() or "data/triage.sqlite",
        max_hops=_int("SUPERVISOR_MAX_HOPS", 4),
        max_tool_calls=_int("SPECIALIST_MAX_TOOL_CALLS", 8),
    )
