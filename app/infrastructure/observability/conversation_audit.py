"""
Optional scrubbed conversation audit (JSON Lines).

Enable with ENABLE_CONVERSATION_AUDIT_LOG=true. Intended for RAG debugging, latency tracking,
and wellbeing-related review — not a replacement for a production DB or SIEM. All text fields
should be passed through PIIScrubber.for_logs() before calling append_conversation_audit_line().
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def append_conversation_audit_line(record: Mapping[str, Any]) -> None:
    if not _env_flag("ENABLE_CONVERSATION_AUDIT_LOG", False):
        return
    path = Path(os.getenv("CONVERSATION_AUDIT_LOG_PATH", "logs/conversation_audit.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = dict(record)
    line.setdefault("ts_utc", datetime.now(timezone.utc).isoformat())
    line.setdefault("schema_version", 1)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
