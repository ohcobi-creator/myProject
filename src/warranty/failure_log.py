"""Persistent failure logging for the warranty-analysis pipeline.

Every module is expected to call `record(...)` from inside an `except` block
(or wherever a failure/lesson should be captured). Entries are appended in
English Markdown to `docs/FAILURE_LOG.md` at the repository root, so lessons
accumulate across runs and can be consulted later.

Why a file-based log? We want lessons to survive restart of the notebook
kernel and to be reviewable via plain git diff. Using a structured Markdown
format keeps it human-readable while still being machine-parseable.
"""

from __future__ import annotations

import datetime as _dt
import threading
from pathlib import Path
from typing import Any

# A single module-level lock so concurrent workers (Step-4/5/6/7 use threads)
# do not interleave their writes and corrupt the log file.
_LOG_LOCK = threading.Lock()


def _resolve_log_path() -> Path:
    """Return the absolute path to docs/FAILURE_LOG.md.

    We walk up from this file to find the repo root (first parent containing
    a `docs/` folder). Fallback: CWD / docs / FAILURE_LOG.md.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "docs" / "FAILURE_LOG.md"
        if candidate.exists():
            return candidate
    # Fallback — create under CWD so we never silently drop entries.
    fallback = Path.cwd() / "docs" / "FAILURE_LOG.md"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    if not fallback.exists():
        fallback.write_text("# Failure Log\n\n", encoding="utf-8")
    return fallback


def record(
    step: str,
    context: str,
    exc: BaseException | str,
    lesson: str,
    status: str = "Open",
    title: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a single failure entry to docs/FAILURE_LOG.md.

    Args:
        step: Which pipeline step the failure occurred in
            (e.g., "Cell 5 (DTC extraction)", "io_utils.load_warranty_xlsx").
        context: Input context (row index, VIN, filename, ...). Short string.
        exc: The exception instance or a string describing the error.
        lesson: What to change / guard against next time.
        status: "Open" or "Resolved". Defaults to "Open".
        title: Optional short title; defaults to the exception class name.
        extra: Optional mapping of extra key/value pairs appended as bullets.

    The function swallows any I/O error while writing the log so that logging
    itself never breaks the caller's flow.
    """
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    err_type = type(exc).__name__ if isinstance(exc, BaseException) else "Error"
    err_msg = str(exc)
    headline = title or err_type

    # Build a Markdown entry matching the schema documented in
    # docs/FAILURE_LOG.md so it stays consistent with what tooling expects.
    lines = [
        f"\n## {ts} \u2014 {headline}",
        f"- **Step**: {step}",
        f"- **Input context**: {context}",
        f"- **Error**: {err_type}: {err_msg}",
        f"- **Root cause**: {_truncate(err_msg, 500)}",
        f"- **Lesson / Fix**: {lesson}",
        f"- **Status**: {status}",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")
    entry = "\n".join(lines) + "\n"

    try:
        path = _resolve_log_path()
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(entry)
    except Exception:  # pragma: no cover - best-effort logger
        # Never raise from the logger. Print to stderr as last resort.
        import sys

        print(f"[failure_log] Could not write entry: {entry}", file=sys.stderr)


def _truncate(s: str, n: int) -> str:
    """Clip long strings so log entries stay readable."""
    return s if len(s) <= n else s[: n - 3] + "..."
