"""DTC extraction and match-verdict logic (Step 5).

1. Regex-extract DTC codes from technician comments (P/C/B/U + 4 hex).
2. Optional typo correction (O->0, l->1) before matching.
3. LLM judges whether the DTCs are consistent with the customer
   complaint and the technician comment.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from tqdm import tqdm

from . import failure_log
from .claude_client import ClaudeClient
from .io_utils import ColumnMap, load_settings
from .prompts import dtc_system
from .schemas import DtcMatchVerdict

# ---------------------------------------------------------------------------
# Regex extraction
# ---------------------------------------------------------------------------

# Standard OBD-II pattern: one letter + four hex digits.
_DTC_PATTERN = re.compile(r"\b[PCBUpbcu][0-9A-Fa-f]{4}\b")


def _correct_typos(text: str) -> str:
    """Replace common OCR / typo characters before regex matching."""
    return text.replace("O", "0").replace("o", "0").replace("l", "1")


def extract_dtcs_regex(text: str, typo_correction: bool = True) -> list[str]:
    """Extract and normalize DTC codes from free text.

    Returns a deduplicated list of uppercase DTC strings.
    """
    if typo_correction:
        # Run regex on the typo-corrected copy but keep the original codes
        # for any matches found in the uncorrected text too.
        corrected = _correct_typos(text)
        matches = set(_DTC_PATTERN.findall(text)) | set(_DTC_PATTERN.findall(corrected))
    else:
        matches = set(_DTC_PATTERN.findall(text))
    return sorted({m.upper() for m in matches})


# ---------------------------------------------------------------------------
# LLM match verdict
# ---------------------------------------------------------------------------


def match_dtc_to_context(
    dtcs: list[str],
    complaint: str,
    tech_comment: str,
    client: ClaudeClient,
) -> DtcMatchVerdict:
    """Ask Claude whether the extracted DTCs match the complaint+comment."""
    if not dtcs:
        return DtcMatchVerdict(
            dtc_codes=[],
            dtc_match_verdict="NoDTC",
            reasoning="No DTC codes found in the technician comment.",
        )

    user_msg = (
        f"DTC codes found: {', '.join(dtcs)}\n\n"
        f"Customer complaint: {complaint}\n\n"
        f"Technician comment: {tech_comment}\n\n"
        "Judge whether the DTCs are consistent with the complaint and comment."
    )

    return client.chat_json(
        messages=[{"role": "user", "content": user_msg}],
        schema=DtcMatchVerdict,
        system=dtc_system(),
    )


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------


def extract_and_match_dtcs(
    df: pd.DataFrame,
    col_map: ColumnMap,
    client: ClaudeClient,
) -> pd.DataFrame:
    """Process every row: regex-extract DTCs then LLM-judge match.

    Returns a DataFrame with columns: dtc_codes, dtc_match_verdict.
    """
    settings = load_settings()
    typo_flag = settings.get("dtc", {}).get("enable_typo_correction", True)

    complaint_col = col_map.customer_complaint
    tech_col = col_map.tech_comment

    codes_list = []
    verdicts = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Step 5: DTC extraction"):
        complaint = str(row.get(complaint_col, "")) if complaint_col else ""
        tech = str(row.get(tech_col, "")) if tech_col else ""

        # Regex extraction.
        dtcs = extract_dtcs_regex(tech, typo_correction=typo_flag)

        # LLM verdict.
        try:
            verdict = match_dtc_to_context(dtcs, complaint, tech, client)
            codes_list.append("; ".join(verdict.dtc_codes) if verdict.dtc_codes else "")
            verdicts.append(
                f"{verdict.dtc_match_verdict} — {verdict.reasoning}"
            )
        except Exception as e:
            failure_log.record(
                step="dtc_extract_match.extract_and_match_dtcs",
                context=f"row={idx}, dtcs={dtcs}",
                exc=e,
                lesson="DTC match LLM call failed; row gets regex-only codes.",
            )
            codes_list.append("; ".join(dtcs))
            verdicts.append("Error — LLM unavailable")

    return pd.DataFrame({
        "dtc_codes": codes_list,
        "dtc_match_verdict": verdicts,
    }, index=df.index)
