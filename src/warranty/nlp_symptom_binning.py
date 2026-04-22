"""NLP-based symptom classification for technician comments (Step 4).

Each warranty row's customer complaint + technician comment is classified
into exactly one of the 13 standard automotive system categories defined
in config/taxonomy.yaml. The LLM system prompt (including the full
taxonomy) is cached so per-row calls only transmit the user message.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from . import failure_log
from .claude_client import ClaudeClient
from .io_utils import ColumnMap, load_settings
from .prompts import symptom_system
from .schemas import SymptomResult


def classify_symptom_row(
    complaint: str,
    tech_comment: str,
    client: ClaudeClient,
) -> SymptomResult:
    """Classify a single row into a symptom category via Claude.

    The system prompt (with taxonomy) is sent with cache_control so
    repeated calls within the same batch reuse the cached prefix.
    """
    user_msg = (
        f"Customer complaint: {complaint}\n\n"
        f"Technician comment: {tech_comment}\n\n"
        "Classify into one symptom category."
    )
    return client.chat_json(
        messages=[{"role": "user", "content": user_msg}],
        schema=SymptomResult,
        system=symptom_system(),
    )


def classify_symptoms_batch(
    df: pd.DataFrame,
    col_map: ColumnMap,
    client: ClaudeClient,
) -> pd.DataFrame:
    """Classify every row in the dataframe, returning 3 new columns.

    Adds: symptom_category, symptom_confidence, symptom_evidence.
    Rows with missing/empty comments are marked UNKNOWN without an
    LLM call. Comments longer than the configured limit are truncated.
    """
    settings = load_settings()
    symptom_cfg = settings.get("symptom", {})
    unknown_label = symptom_cfg.get("unknown_label", "UNKNOWN")
    char_limit = symptom_cfg.get("long_comment_char_limit", 4000)
    keep_chars = symptom_cfg.get("long_comment_keep", 2000)

    complaint_col = col_map.customer_complaint
    tech_col = col_map.tech_comment

    categories = []
    confidences = []
    evidences = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Step 4: Symptom binning"):
        complaint = str(row.get(complaint_col, "")) if complaint_col else ""
        tech = str(row.get(tech_col, "")) if tech_col else ""

        # Skip LLM call for empty comments.
        if not tech.strip() or tech.strip().lower() == "nan":
            categories.append(unknown_label)
            confidences.append(0.0)
            evidences.append("No technician comment available.")
            continue

        # Truncate very long comments.
        if len(tech) > char_limit:
            tech = tech[:keep_chars] + " ... " + tech[-keep_chars:]

        try:
            result = classify_symptom_row(complaint, tech, client)
            categories.append(result.symptom_category)
            confidences.append(result.symptom_confidence)
            evidences.append(result.symptom_evidence)
        except Exception as e:
            failure_log.record(
                step="nlp_symptom_binning.classify_symptoms_batch",
                context=f"row={idx}",
                exc=e,
                lesson="LLM classification failed; row marked UNKNOWN.",
            )
            categories.append(unknown_label)
            confidences.append(0.0)
            evidences.append(f"Classification error: {e}")

    result_df = pd.DataFrame({
        "symptom_category": categories,
        "symptom_confidence": confidences,
        "symptom_evidence": evidences,
    }, index=df.index)

    return result_df
