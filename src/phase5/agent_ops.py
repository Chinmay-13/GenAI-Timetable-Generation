"""
Agent operations logger and backup manager.
Every autonomous action the agent takes is logged here.

All functions accept an optional `sem_id` parameter.  When provided, the
operations directory and backup paths resolve under the semester's output
tree (``outputs/<sem_id>/agent_ops/``).  When ``sem_id`` is None the legacy
flat ``outputs/agent_ops/`` is used.
"""
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import resolve_output_path, get_sem_paths


# ── Legacy fallback paths (only used when sem_id is None) ─────────────────────
_LEGACY_AGENT_OPS_DIR = PROJECT_ROOT / "outputs" / "agent_ops"
_LEGACY_BACKUPS_DIR   = _LEGACY_AGENT_OPS_DIR / "backups"


def _ops_dir(sem_id: str = None) -> Path:
    """Return the agent_ops directory for the given semester."""
    if sem_id is not None:
        return get_sem_paths(sem_id).agent_ops_dir
    return _LEGACY_AGENT_OPS_DIR


def _backups_dir(sem_id: str = None) -> Path:
    return _ops_dir(sem_id) / "backups"


def _output_dir(sem_id: str = None) -> Path:
    """Return the output directory for the given semester."""
    if sem_id is not None:
        return get_sem_paths(sem_id).output_dir
    from config import OUTPUT_DIR
    return OUTPUT_DIR


def _ensure_dirs(sem_id: str = None):
    _ops_dir(sem_id).mkdir(parents=True, exist_ok=True)
    _backups_dir(sem_id).mkdir(parents=True, exist_ok=True)


def backup_timetable(section_id: str, sem_id: str = None) -> Path:
    """Copy current timetable CSV to backups before editing."""
    _ensure_dirs(sem_id)
    src = _output_dir(sem_id) / f"section_{section_id}_timetable.csv"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = _backups_dir(sem_id) / f"{ts}-section_{section_id}_timetable.csv"
    shutil.copy2(src, dst)
    return dst


def log_operation(
    action: str,
    absent_faculty: str,
    section_id: str,
    day: str,
    period_range: tuple,
    substitute_faculty: str,
    reasoning_chain: list,
    pre_state: str,
    post_state: str,
    commit_result: str,
    backup_path: str,
    sem_id: str = None,
) -> Path:
    """Write a structured JSON log entry to the semester's agent_ops dir."""
    _ensure_dirs(sem_id)
    op_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y%m%dT%H%M%SZ")
    record = {
        "operation_id": op_id,
        "timestamp_utc": now.isoformat(),
        "timestamp_local": datetime.now().isoformat(),
        "action": action,
        "absent_faculty": absent_faculty,
        "section_id": section_id,
        "day": day,
        "period_range": list(period_range),
        "substitute_faculty": substitute_faculty,
        "reasoning_chain": reasoning_chain,
        "pre_state": pre_state,
        "post_state": post_state,
        "commit_result": commit_result,
        "backup_path": str(backup_path),
        "sem_id": sem_id or "legacy",
    }
    log_path = _ops_dir(sem_id) / f"{ts_str}-{op_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return log_path


def list_operations(limit: int = 20, sem_id: str = None) -> list:
    """Return the most recent N operation logs for the given semester."""
    _ensure_dirs(sem_id)
    files = sorted(_ops_dir(sem_id).glob("*.json"), reverse=True)[:limit]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def rollback_operation(operation_id: str, sem_id: str = None) -> str:
    """
    Restore ALL backed-up artifacts for a given operation_id.

    Handles two backup formats:
    - New format (sync_manager): backup_path is a directory containing
      section CSV, faculty CSVs, and summary_report.txt.
    - Legacy format (old agent.py): backup_path is a single section CSV file.

    After restoring files, triggers a best-effort RAG re-index.
    """
    _ensure_dirs(sem_id)
    files = list(_ops_dir(sem_id).glob(f"*-{operation_id}.json"))
    if not files:
        return f"Operation {operation_id} not found."

    record = json.loads(files[0].read_text(encoding="utf-8"))
    backup_path = Path(record["backup_path"])
    section = record["section_id"]

    out_dir = _output_dir(sem_id)

    restored: list[str] = []

    if backup_path.is_dir():
        # New format: restore every file in the backup directory
        for bk_file in backup_path.iterdir():
            dst = out_dir / bk_file.name
            shutil.copy2(bk_file, dst)
            restored.append(bk_file.name)
    elif backup_path.is_file():
        # Legacy format: a single section CSV backup
        target = out_dir / f"section_{section}_timetable.csv"
        shutil.copy2(backup_path, target)
        restored.append(target.name)
    else:
        return f"Backup not found at: {backup_path}"

    # Best-effort RAG re-index (semester-aware)
    rag_status = "not refreshed"
    try:
        from src.phase5.rag_indexer import build_index
        if build_index(sem_id=sem_id)[0] is not None:
            rag_status = "refreshed"
    except Exception:
        pass

    return (
        f"Rolled back operation '{operation_id}'. "
        f"Restored: {', '.join(restored)}. "
        f"RAG index {rag_status}."
    )
