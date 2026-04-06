"""
sync_manager.py — Centralized write-back for all schedule changes.

Every commit (substitute or swap) must flow through
``commit_schedule_change(change_dict)``.  The function atomically:

  1. Backs up the section CSV + both faculty CSVs + summary_report.txt
  2. Patches the section CSV
  3. Rebuilds faculty CSVs for affected faculty (from all section CSVs on disk)
  4. Rebuilds summary_report.txt (slot counts + faculty loads)
  5. Triggers a RAG re-index (best-effort, skipped if deps missing)
  6. Logs the operation via agent_ops.log_operation()

On ANY failure the backup is restored for ALL touched files and an exception
is re-raised with a clear message.

Usage:
    from src.phase5.sync_manager import commit_schedule_change, rollback_change
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import sys

# ── project root on path ──────────────────────────────────────────────────────
_SYNC_ROOT = Path(__file__).resolve().parents[2]
if str(_SYNC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SYNC_ROOT))

import pandas as pd
import config
from config import DAYS, PERIODS, LAB_PERIODS, SHORT_NAMES, MAX_HOURS, SECTIONS
from config import OUTPUT_DIR, DATA_DIR, resolve_output_path, get_sem_paths
from src.phase5.agent_ops import (
    BACKUPS_DIR, AGENT_OPS_DIR,
    log_operation, list_operations,
)

logger = logging.getLogger(__name__)

# ── required keys in change_dict ─────────────────────────────────────────────
_REQUIRED_KEYS = {"section", "day", "period_start", "period_end",
                  "original_faculty", "new_faculty", "change_type"}


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write DataFrame to CSV atomically via temp-file + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via temp-file + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _faculty_csv_path(faculty_id: str, out: Path = None) -> Path:
    """Canonical path for a faculty timetable CSV."""
    base = out if out is not None else OUTPUT_DIR
    return base / f"faculty_{faculty_id.upper()}_timetable.csv"


def _section_csv_path(section: str, out: Path = None) -> Path:
    """Canonical path for a section timetable CSV."""
    base = out if out is not None else OUTPUT_DIR
    return base / f"section_{section.upper()}_timetable.csv"


def _summary_path(out: Path = None) -> Path:
    base = out if out is not None else OUTPUT_DIR
    return base / "summary_report.txt"


# ── backup helpers ────────────────────────────────────────────────────────────

def _make_backup_dir(op_ts: str) -> Path:
    """Create a timestamped backup sub-directory inside BACKUPS_DIR."""
    d = BACKUPS_DIR / op_ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _backup_file(src: Path, backup_dir: Path) -> Optional[Path]:
    """Copy *src* into *backup_dir*; return dst path, or None if src missing."""
    if not src.exists():
        return None
    dst = backup_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _restore_backup(backup_dir: Path, filenames: List[str], out: Path = None) -> None:
    """Copy all *filenames* from backup_dir back to the output directory."""
    base = out if out is not None else OUTPUT_DIR
    for name in filenames:
        src = backup_dir / name
        if src.exists():
            dst = base / name
            shutil.copy2(src, dst)
            logger.info("Rollback: restored %s", name)
        else:
            logger.warning("Rollback: backup missing for %s", name)


# ── faculty CSV reconstruction ────────────────────────────────────────────────

def _initials(name: str) -> str:
    tokens = [t.strip(".") for t in str(name).split()
              if t and t.lower() != "prof."]
    return "".join(t[0].upper() for t in tokens[:3]) if tokens else "NA"


def _load_faculty_meta(data_dir: Path = None) -> Dict[str, Dict]:
    """Return {faculty_id: {name, designation, initials}}."""
    fac_path = (data_dir if data_dir is not None else DATA_DIR) / "faculty.csv"
    if not fac_path.exists():
        return {}
    df = pd.read_csv(fac_path)
    meta = {}
    for _, row in df.iterrows():
        fid = str(row["faculty_id"]).strip()
        meta[fid] = {
            "name": str(row["name"]),
            "designation": str(row["designation"]),
            "initials": _initials(str(row["name"])),
        }
    return meta


def rebuild_faculty_csv(faculty_id: str, sem_id: str = None) -> Path:
    """
    Reconstruct a faculty timetable CSV by scanning every section CSV on disk.

    Returns the path written.
    """
    out = get_sem_paths(sem_id).output_dir if sem_id else OUTPUT_DIR
    fid = faculty_id.strip().upper()
    period_cols = [f"P{p}" for p in PERIODS]

    # Initialise empty grid
    grid: Dict[str, Dict[str, str]] = {
        day: {f"P{p}": "----" for p in PERIODS}
        for day in DAYS
    }

    for section in SECTIONS:
        sec_path = _section_csv_path(section, out)
        if not sec_path.exists():
            sec_path = out / f"section_{section}_timetable.csv"
        if not sec_path.exists():
            continue
        try:
            df = pd.read_csv(sec_path)
        except Exception as exc:
            logger.warning("Cannot read %s: %s", sec_path.name, exc)
            continue

        for _, row in df.iterrows():
            day = str(row["Day"]).strip()
            if day not in DAYS:
                continue
            for p in PERIODS:
                col = f"P{p}"
                cell = str(row.get(col, "----")).strip()
                if _cell_belongs_to_faculty(cell, fid):
                    base = cell.split("→")[0].strip()
                    grid[day][col] = f"{base} ({section})"

    rows = []
    for day in DAYS:
        row_data = {"Day": day}
        for col in period_cols:
            row_data[col] = grid[day][col]
        rows.append(row_data)

    df_out = pd.DataFrame(rows, columns=["Day"] + period_cols)
    out_path = _faculty_csv_path(fid, out)
    _atomic_write_csv(out_path, df_out)
    logger.info("Rebuilt faculty CSV: %s", out_path.name)
    return out_path


def _cell_belongs_to_faculty(cell: str, faculty_id: str) -> bool:
    """
    Heuristic: a section CSV cell belongs to faculty_id when:
      - The cell contains '→<faculty_id>' (substitution annotation), OR
      - The cell contains '(<initials>)' matching this faculty's initials.

    We load faculty metadata lazily to avoid repeated CSV reads.
    """
    if not cell or cell in ("----", "nan", ""):
        return False
    fid_upper = faculty_id.upper()

    # Direct substitution annotation: "DDCO (XYZ)→F07"
    if f"→{fid_upper}" in cell.upper():
        return True

    # Check initials inside parentheses: "DDCO (ABC)"
    meta = _load_faculty_meta()
    if fid_upper in meta:
        initials = meta[fid_upper]["initials"]
        # Match "(INITIALS)" but not "(INITIALS)→..." for the original faculty
        import re
        pattern = rf"\({re.escape(initials)}\)"
        if re.search(pattern, cell, re.IGNORECASE):
            # Make sure it's not a sub annotation that overrides this faculty
            # i.e. the cell hasn't been reassigned away from this faculty.
            # If there's a →SOMEONE_ELSE annotation, skip.
            arrow_idx = cell.find("→")
            if arrow_idx != -1:
                sub_fid = cell[arrow_idx + 1:].strip().upper()
                if sub_fid != fid_upper:
                    return False  # reassigned away from this faculty
            return True
    return False


# ── summary report reconstruction ────────────────────────────────────────────

def rebuild_summary_report(sem_id: str = None) -> Path:
    """
    Rebuild summary_report.txt from all section CSVs currently on disk.

    Counts theory slots (non-LAB, non-empty) and lab slots (containing "LAB").
    Also recomputes faculty load from faculty CSVs.
    """
    out      = get_sem_paths(sem_id).output_dir if sem_id else OUTPUT_DIR
    data_dir = get_sem_paths(sem_id).data_dir   if sem_id else DATA_DIR
    fac_meta = _load_faculty_meta(data_dir)
    period_cols = [f"P{p}" for p in PERIODS]

    total_theory = 0
    total_lab = 0

    for section in SECTIONS:
        sp = _section_csv_path(section, out)
        if not sp.exists():
            sp = out / f"section_{section}_timetable.csv"
        if not sp.exists():
            continue
        try:
            df = pd.read_csv(sp)
        except Exception:
            continue
        for _, row in df.iterrows():
            for col in period_cols:
                cell = str(row.get(col, "----")).strip()
                if cell in ("----", "", "nan"):
                    continue
                if "LAB" in cell.upper():
                    total_lab += 1
                else:
                    total_theory += 1

    lines = [
        "TIMETABLE SUMMARY REPORT",
        "========================",
        f"Total sections scheduled: {len(SECTIONS)}",
        f"Total theory slots placed: {total_theory}",
        f"Total lab slots placed: {total_lab} (12 pairs x 2 periods x 2 sections = 48)",
        "Soft constraint violations:",
        "  same_subject_same_day: (recomputed after live edit)",
        "  back_to_back_same_subject: (recomputed after live edit)",
        "",
        "Faculty Load Table",
        "faculty_id | name | total_hours | max_hours | status",
    ]

    for fid, meta in fac_meta.items():
        fp = _faculty_csv_path(fid, out)
        if not fp.exists():
            fp = out / f"faculty_{fid}_timetable.csv"
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        total = 0
        for _, row in df.iterrows():
            for col in period_cols:
                cell = str(row.get(col, "----")).strip()
                if cell not in ("----", "", "nan"):
                    total += 1
        max_h = MAX_HOURS.get(meta["designation"], 16)
        status = "OK" if total <= max_h else "OVERLOAD"
        lines.append(
            f"{fid} | {meta['name']} | {total} | {max_h} | {status}"
        )

    content = "\n".join(lines)
    out_path = _summary_path(out)
    _atomic_write_text(out_path, content)
    logger.info("Rebuilt summary_report.txt")
    return out_path


# ── RAG re-index (best-effort) ────────────────────────────────────────────────

def _try_rebuild_rag(sem_id: str = None) -> bool:
    """Attempt to rebuild the FAISS RAG index. Returns True on success."""
    try:
        from src.phase5.rag_indexer import build_index
        result = build_index(sem_id=sem_id)
        if result[0] is not None:
            logger.info("RAG index rebuilt successfully.")
            return True
        logger.warning("RAG index rebuild returned None (deps missing?).")
        return False
    except Exception as exc:
        logger.warning("RAG re-index failed (non-fatal): %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def commit_schedule_change(change_dict: dict, sem_id: str = None) -> dict:
    """
    Atomically commit a schedule change and update ALL derived artifacts.

    Parameters
    ----------
    change_dict : dict
        Required keys:
          section          – str, e.g. "A"
          day              – str, e.g. "Monday"
          period_start     – int
          period_end       – int
          original_faculty – str, faculty_id being replaced/absent, e.g. "F03"
          new_faculty      – str, faculty_id taking over, e.g. "F07"
          change_type      – str, "substitute" or "swap"
        Optional:
          reason           – str

    Returns
    -------
    dict with keys:
      success       – bool
      message       – str
      log_path      – str (path to the JSON op-log)
      rag_refreshed – bool

    Raises
    ------
    ValueError  if required keys are missing or paths don't exist.
    RuntimeError on any write failure (all changes already rolled back).
    """
    # ── 0. Validate ───────────────────────────────────────────────────────────
    missing = _REQUIRED_KEYS - change_dict.keys()
    if missing:
        raise ValueError(f"commit_schedule_change: missing keys {missing}")

    # ── Resolve semester-aware output dir once ────────────────────────────────
    out: Path = get_sem_paths(sem_id).output_dir if sem_id else OUTPUT_DIR

    section        = str(change_dict["section"]).upper()
    day            = str(change_dict["day"]).strip()
    p_start        = int(change_dict["period_start"])
    p_end          = int(change_dict["period_end"])
    orig_fac       = str(change_dict["original_faculty"]).upper()
    new_fac        = str(change_dict["new_faculty"]).upper()
    change_type    = str(change_dict["change_type"])
    reason         = str(change_dict.get("reason", ""))

    sec_path       = _section_csv_path(section, out)
    if not sec_path.exists():
        sec_path = out / f"section_{section}_timetable.csv"
    if not sec_path.exists():
        raise ValueError(
            f"Section CSV not found for section {section}. "
            "Run run_all.py to generate outputs first."
        )

    # ── 1. Backup ─────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Backup dir lives inside the semester-aware output tree
    backup_dir = out / "agent_ops" / "backups" / ts
    backup_dir.mkdir(parents=True, exist_ok=True)

    backed_up: List[str] = []

    def _bk(path: Path) -> None:
        dst = _backup_file(path, backup_dir)
        if dst:
            backed_up.append(path.name)

    _bk(sec_path)
    _bk(_faculty_csv_path(orig_fac, out))
    _bk(_faculty_csv_path(new_fac, out))
    _bk(_summary_path(out))

    logger.info(
        "Backup created at %s for files: %s",
        backup_dir, backed_up
    )

    # ── 2. Patch section CSV ──────────────────────────────────────────────────
    try:
        df = pd.read_csv(sec_path)
        row_mask = df["Day"].str.strip() == day
        if not row_mask.any():
            raise ValueError(
                f"Day '{day}' not found in section {section} timetable."
            )

        pre_state = df[row_mask].to_csv(index=False)

        for p in range(p_start, p_end + 1):
            col = f"P{p}"
            if col in df.columns:
                current = str(df.loc[row_mask, col].values[0]).strip()
                # Annotate cell: "DDCO (XYZ)→F07"
                df.loc[row_mask, col] = f"{current}→{new_fac}"

        post_state = df[row_mask].to_csv(index=False)
        _atomic_write_csv(sec_path, df)
        logger.info("Section CSV patched: %s", sec_path.name)

    except Exception as exc:
        _restore_backup(backup_dir, backed_up, out)
        raise RuntimeError(
            f"Failed to patch section CSV — rolled back. Cause: {exc}"
        ) from exc

    # ── 3. Rebuild faculty CSVs ───────────────────────────────────────────────
    try:
        rebuild_faculty_csv(orig_fac, sem_id)
        rebuild_faculty_csv(new_fac, sem_id)
    except Exception as exc:
        _restore_backup(backup_dir, backed_up, out)
        raise RuntimeError(
            f"Failed to rebuild faculty CSVs — rolled back. Cause: {exc}"
        ) from exc

    # ── 4. Rebuild summary_report.txt ─────────────────────────────────────────
    try:
        rebuild_summary_report(sem_id)
    except Exception as exc:
        _restore_backup(backup_dir, backed_up, out)
        raise RuntimeError(
            f"Failed to rebuild summary report — rolled back. Cause: {exc}"
        ) from exc

    # ── 5. RAG re-index (best-effort) ─────────────────────────────────────────
    rag_ok = _try_rebuild_rag(sem_id)

    # ── 6. Log operation ──────────────────────────────────────────────────────
    try:
        log_path = log_operation(
            action=change_type,
            absent_faculty=orig_fac,
            section_id=section,
            day=day,
            period_range=(p_start, p_end),
            substitute_faculty=new_fac,
            reasoning_chain=[reason] if reason else ["no reason provided"],
            pre_state=pre_state,
            post_state=post_state,
            commit_result="SUCCESS",
            backup_path=str(backup_dir),
        )
    except Exception as exc:
        # Logging failure is non-fatal — don't roll back the actual changes
        logger.error("Failed to log operation (non-fatal): %s", exc)
        log_path = None

    msg = (
        f"Committed {change_type}: {new_fac} covers section {section} "
        f"on {day} P{p_start}-P{p_end} (replacing {orig_fac}). "
        f"Faculty CSVs rebuilt. Summary updated. "
        f"RAG {'refreshed' if rag_ok else 'not refreshed (deps missing)'}."
    )
    logger.info(msg)

    return {
        "success": True,
        "message": msg,
        "log_path": str(log_path) if log_path else None,
        "rag_refreshed": rag_ok,
        "backup_dir": str(backup_dir),
    }


def rollback_change(operation_id: str, sem_id: str = None) -> str:
    """
    Roll back a committed change by operation_id.

    Restores section CSV, faculty CSVs, and summary_report.txt from backup,
    then triggers a best-effort RAG re-index.

    Returns a human-readable status message.
    """
    out = get_sem_paths(sem_id).output_dir if sem_id else OUTPUT_DIR

    AGENT_OPS_DIR.mkdir(parents=True, exist_ok=True)
    files = list(AGENT_OPS_DIR.glob(f"*-{operation_id}.json"))
    if not files:
        return f"Operation '{operation_id}' not found."

    import json
    record = json.loads(files[0].read_text(encoding="utf-8"))
    backup_dir = Path(record.get("backup_path", ""))

    if not backup_dir.exists():
        return (
            f"Backup directory not found: {backup_dir}. "
            "Cannot rollback — manual restoration required."
        )

    # Restore every file that was backed up
    restored = []
    for bk_file in backup_dir.iterdir():
        dst = out / bk_file.name
        shutil.copy2(bk_file, dst)
        restored.append(bk_file.name)
        logger.info("Rollback: restored %s", bk_file.name)

    # Best-effort RAG rebuild after rollback
    rag_ok = _try_rebuild_rag(sem_id)

    summary = (
        f"Rolled back operation '{operation_id}'. "
        f"Restored: {', '.join(restored)}. "
        f"RAG {'refreshed' if rag_ok else 'not refreshed'}."
    )
    logger.info(summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Temp-folder preview API
# ═══════════════════════════════════════════════════════════════════════════════

def _temp_root(sem_id: str = None) -> Path:
    """Return the temp-preview root directory for this semester."""
    out = get_sem_paths(sem_id).output_dir if sem_id else OUTPUT_DIR
    return out / "agent_ops" / "temp"


def _build_preview_faculty_csv(
    faculty_id: str,
    patched_section: str,
    patched_df,   # pd.DataFrame — already-patched section rows
    out: Path,
    temp_dir: Path,
    sem_id: str = None,
) -> None:
    """
    Write a faculty timetable CSV into *temp_dir* using *patched_df* for
    *patched_section* and live CSVs for all other sections.
    Does NOT touch live output files.
    """
    fid = faculty_id.strip().upper()
    period_cols = [f"P{p}" for p in PERIODS]

    grid: Dict[str, Dict[str, str]] = {
        day: {f"P{p}": "----" for p in PERIODS}
        for day in DAYS
    }

    for section in SECTIONS:
        if section == patched_section:
            df_sec = patched_df
        else:
            sec_path = _section_csv_path(section, out)
            if not sec_path.exists():
                sec_path = out / f"section_{section}_timetable.csv"
            if not sec_path.exists():
                continue
            try:
                df_sec = pd.read_csv(sec_path)
            except Exception:
                continue

        for _, row in df_sec.iterrows():
            day = str(row["Day"]).strip()
            if day not in DAYS:
                continue
            for p in PERIODS:
                col = f"P{p}"
                cell = str(row.get(col, "----")).strip()
                if _cell_belongs_to_faculty(cell, fid):
                    base = cell.split("→")[0].strip()
                    grid[day][col] = f"{base} ({section})"

    rows = []
    for day in DAYS:
        row_data = {"Day": day}
        for col in period_cols:
            row_data[col] = grid[day][col]
        rows.append(row_data)

    df_out = pd.DataFrame(rows, columns=["Day"] + period_cols)
    _atomic_write_csv(temp_dir / f"faculty_{fid}_timetable.csv", df_out)
    logger.info("Built preview faculty CSV: faculty_%s_timetable.csv (temp)", fid)


def preview_schedule_change(change_dict: dict, sem_id: str = None) -> dict:
    """
    Write proposed changes to a temp folder WITHOUT touching live outputs.

    Creates  outputs/{sem_id}/agent_ops/temp/{op_id}/  containing:
      section_{X}_timetable.csv        — patched section view
      faculty_{ABSENT}_timetable.csv   — absent faculty (gap shown)
      faculty_{SUB}_timetable.csv      — substitute (new assignment shown)
      .pending                         — JSON metadata marker

    Returns
    -------
    dict with keys: op_id, temp_dir, diff_summary, message
    """
    missing = _REQUIRED_KEYS - change_dict.keys()
    if missing:
        raise ValueError(f"preview_schedule_change: missing keys {missing}")

    out: Path = get_sem_paths(sem_id).output_dir if sem_id else OUTPUT_DIR

    section     = str(change_dict["section"]).upper()
    day         = str(change_dict["day"]).strip()
    p_start     = int(change_dict["period_start"])
    p_end       = int(change_dict["period_end"])
    orig_fac    = str(change_dict["original_faculty"]).upper()
    new_fac     = str(change_dict["new_faculty"]).upper()
    change_type = str(change_dict["change_type"])
    reason      = str(change_dict.get("reason", ""))

    sec_path = _section_csv_path(section, out)
    if not sec_path.exists():
        sec_path = out / f"section_{section}_timetable.csv"
    if not sec_path.exists():
        raise ValueError(
            f"Section CSV not found for section {section}. "
            "Run run_all.py to generate outputs first."
        )

    import uuid as _uuid
    import json as _json
    op_id    = _uuid.uuid4().hex[:8]
    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    temp_dir = _temp_root(sem_id) / op_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ── Patch section CSV in-memory ──────────────────────────────────────────
    df       = pd.read_csv(sec_path)
    row_mask = df["Day"].str.strip() == day
    if not row_mask.any():
        raise ValueError(f"Day '{day}' not found in section {section} timetable.")

    pre_state = df[row_mask].to_csv(index=False)
    for p in range(p_start, p_end + 1):
        col = f"P{p}"
        if col in df.columns:
            current = str(df.loc[row_mask, col].values[0]).strip()
            df.loc[row_mask, col] = f"{current}→{new_fac}"
    post_state = df[row_mask].to_csv(index=False)

    # ── Write preview files to temp dir ──────────────────────────────────────
    _atomic_write_csv(temp_dir / f"section_{section}_timetable.csv", df)
    _build_preview_faculty_csv(orig_fac, section, df, out, temp_dir, sem_id)
    _build_preview_faculty_csv(new_fac,  section, df, out, temp_dir, sem_id)

    # ── Write .pending marker ─────────────────────────────────────────────────
    marker_data = {
        "op_id":       op_id,
        "timestamp":   ts,
        "change_dict": dict(change_dict),
        "pre_state":   pre_state,
        "post_state":  post_state,
        "sem_id":      sem_id,
        "temp_dir":    str(temp_dir),
    }
    (temp_dir / ".pending").write_text(
        _json.dumps(marker_data, indent=2), encoding="utf-8"
    )

    diff_lines = [
        f"[PREVIEW] {change_type.upper()}: {new_fac} covers section {section} "
        f"on {day} P{p_start}–P{p_end} (replacing {orig_fac}).",
        "Patched cells (not yet committed):",
    ]
    for p in range(p_start, p_end + 1):
        col = f"P{p}"
        if col in df.columns:
            diff_lines.append(f"  {col}: annotated →{new_fac}")

    logger.info("Preview created: op_id=%s, temp_dir=%s", op_id, temp_dir)
    return {
        "op_id":        op_id,
        "temp_dir":     str(temp_dir),
        "diff_summary": "\n".join(diff_lines),
        "message": (
            f"Preview generated (op_id={op_id}). No live files changed. "
            "Ask the user to confirm, then call apply_pending_preview."
        ),
    }


def commit_from_preview(op_id: str, sem_id: str = None) -> dict:
    """
    Promote temp-folder preview files to live outputs and finalise.

    Steps:
      1. Backup live files.
      2. Copy preview files → live output dir atomically.
      3. Rebuild summary_report.txt and RAG index.
      4. Log the operation.
      5. Remove the temp folder.

    Returns the same dict shape as commit_schedule_change.
    Raises RuntimeError (with full rollback) on any write failure.
    """
    import json as _json
    temp_dir = _temp_root(sem_id) / op_id
    marker   = temp_dir / ".pending"

    if not temp_dir.exists() or not marker.exists():
        raise ValueError(f"No pending preview found for op_id={op_id!r}.")

    meta        = _json.loads(marker.read_text(encoding="utf-8"))
    change_dict = meta["change_dict"]
    pre_state   = meta["pre_state"]
    post_state  = meta["post_state"]

    out: Path = get_sem_paths(sem_id).output_dir if sem_id else OUTPUT_DIR

    section     = str(change_dict["section"]).upper()
    day         = str(change_dict["day"]).strip()
    p_start     = int(change_dict["period_start"])
    p_end       = int(change_dict["period_end"])
    orig_fac    = str(change_dict["original_faculty"]).upper()
    new_fac     = str(change_dict["new_faculty"]).upper()
    change_type = str(change_dict["change_type"])
    reason      = str(change_dict.get("reason", ""))

    # ── Backup live files first ───────────────────────────────────────────────
    ts         = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = out / "agent_ops" / "backups" / ts
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up: List[str] = []

    def _bk(path: Path) -> None:
        dst = _backup_file(path, backup_dir)
        if dst:
            backed_up.append(path.name)

    _bk(_section_csv_path(section, out))
    _bk(_faculty_csv_path(orig_fac, out))
    _bk(_faculty_csv_path(new_fac,  out))
    _bk(_summary_path(out))
    logger.info("commit_from_preview: backup at %s (%s)", backup_dir, backed_up)

    try:
        # ── Copy preview files → live outputs ─────────────────────────────────
        for preview_file in temp_dir.iterdir():
            if preview_file.name.startswith("."):
                continue  # skip .pending marker
            dst = out / preview_file.name
            shutil.copy2(preview_file, dst)
            logger.info("Promoted preview file: %s", preview_file.name)

        rebuild_summary_report(sem_id)

    except Exception as exc:
        _restore_backup(backup_dir, backed_up, out)
        raise RuntimeError(
            f"commit_from_preview failed — rolled back. Cause: {exc}"
        ) from exc

    # ── RAG re-index (best-effort) ────────────────────────────────────────────
    rag_ok = _try_rebuild_rag(sem_id)

    # ── Log operation ─────────────────────────────────────────────────────────
    try:
        log_path = log_operation(
            action=change_type,
            absent_faculty=orig_fac,
            section_id=section,
            day=day,
            period_range=(p_start, p_end),
            substitute_faculty=new_fac,
            reasoning_chain=[reason] if reason else ["no reason provided"],
            pre_state=pre_state,
            post_state=post_state,
            commit_result="SUCCESS",
            backup_path=str(backup_dir),
        )
    except Exception as exc:
        logger.error("Failed to log operation (non-fatal): %s", exc)
        log_path = None

    # ── Delete temp dir ───────────────────────────────────────────────────────
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Temp dir cleaned up: %s", temp_dir)
    except Exception:
        pass

    msg = (
        f"Committed {change_type}: {new_fac} covers section {section} "
        f"on {day} P{p_start}-P{p_end} (replacing {orig_fac}). "
        f"Faculty CSVs rebuilt. Summary updated. "
        f"RAG {'refreshed' if rag_ok else 'not refreshed (deps missing)'}."
    )
    logger.info(msg)
    return {
        "success":       True,
        "message":       msg,
        "log_path":      str(log_path) if log_path else None,
        "rag_refreshed": rag_ok,
        "backup_dir":    str(backup_dir),
    }


def discard_preview(op_id: str, sem_id: str = None) -> None:
    """Delete a pending preview temp folder without touching live files."""
    temp_dir = _temp_root(sem_id) / op_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Preview discarded: op_id=%s", op_id)


def get_active_preview(sem_id: str = None) -> Optional[dict]:
    """
    Return metadata of the active preview for *sem_id*, or None.
    Scans outputs/{sem_id}/agent_ops/temp/ for .pending marker files.
    Returns the first valid metadata dict found, or None if nothing pending.
    """
    import json as _json
    root = _temp_root(sem_id)
    if not root.exists():
        return None
    for marker in sorted(root.glob("*/.pending")):
        try:
            return _json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None
