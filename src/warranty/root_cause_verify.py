"""Root-cause system/part determination (Step 8).

Synthesizes all prior evidence (symptom category, DTC codes and match
verdict, DTC web verification, verified repair method, customer
complaint, and technician comment) to determine the SPECIFIC failing
system and part for each warranty claim.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from . import failure_log
from .claude_client import ClaudeClient
from .prompts import root_cause_system
from .schemas import RootCause


def determine_root_cause_row(
    row: dict,
    client: ClaudeClient,
) -> RootCause:
    """Determine root cause for a single row using Claude Opus."""
    user_msg = (
        f"Customer complaint: {row.get('customer_complaint', 'N/A')}\n"
        f"Technician comment: {row.get('tech_comment', 'N/A')}\n"
        f"Symptom category: {row.get('symptom_category', 'N/A')}\n"
        f"DTC codes: {row.get('dtc_codes', 'None')}\n"
        f"DTC match verdict: {row.get('dtc_match_verdict', 'N/A')}\n"
        f"DTC web-verified case: {row.get('dtc_web_verified_case', 'N/A')}\n"
        f"Repair web-verified: {row.get('repair_web_verified', 'N/A')}\n\n"
        "Determine the specific failing system and part."
    )

    return client.chat_json(
        messages=[{"role": "user", "content": user_msg}],
        schema=RootCause,
        system=root_cause_system(),
        model_override=client.heavy_model,
    )


def determine_root_cause(
    df: pd.DataFrame,
    client: ClaudeClient,
    complaint_col: str = "customer_complaint",
    tech_col: str = "tech_comment",
) -> pd.DataFrame:
    """Run root-cause determination on every row.

    Expects that columns from Steps 4-7 are already present in df:
    symptom_category, dtc_codes, dtc_match_verdict,
    dtc_web_verified_case, repair_web_verified.

    Returns a DataFrame with 4 columns: root_cause_system,
    root_cause_part, root_cause_confidence, root_cause_reasoning.
    """
    systems = []
    parts = []
    confidences = []
    reasonings = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Step 8: Root cause"):
        row_dict = {
            "customer_complaint": str(row.get(complaint_col, "")),
            "tech_comment": str(row.get(tech_col, "")),
            "symptom_category": str(row.get("symptom_category", "")),
            "dtc_codes": str(row.get("dtc_codes", "")),
            "dtc_match_verdict": str(row.get("dtc_match_verdict", "")),
            "dtc_web_verified_case": str(row.get("dtc_web_verified_case", "")),
            "repair_web_verified": str(row.get("repair_web_verified", "")),
        }

        try:
            rc = determine_root_cause_row(row_dict, client)
            systems.append(rc.root_cause_system)
            parts.append(rc.root_cause_part)
            confidences.append(rc.root_cause_confidence)
            reasonings.append(rc.root_cause_reasoning)
        except Exception as e:
            failure_log.record(
                step="root_cause_verify.determine_root_cause",
                context=f"row={idx}",
                exc=e,
                lesson="Root-cause LLM call failed; row gets UNKNOWN.",
            )
            systems.append("UNKNOWN")
            parts.append("UNKNOWN")
            confidences.append(0.0)
            reasonings.append(f"Error: {e}")

    return pd.DataFrame({
        "root_cause_system": systems,
        "root_cause_part": parts,
        "root_cause_confidence": confidences,
        "root_cause_reasoning": reasonings,
    }, index=df.index)
