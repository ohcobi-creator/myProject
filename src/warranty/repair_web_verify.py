"""Repair-method web-search verification (Step 7).

Reads the technician comment, extracts the described repair action,
and searches the web for the standard procedure to verify alignment
with industry best practice.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from . import failure_log
from .claude_client import ClaudeClient
from .io_utils import ColumnMap
from .prompts import repair_web_search_system


def verify_repair_web(
    df: pd.DataFrame,
    col_map: ColumnMap,
    client: ClaudeClient,
) -> pd.DataFrame:
    """Verify the repair method for each row via web search.

    Returns a single-column DataFrame: repair_web_verified.
    """
    tech_col = col_map.tech_comment
    make_col = col_map.make
    model_col = col_map.model

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Step 7: Repair verify"):
        tech = str(row.get(tech_col, "")) if tech_col else ""
        if not tech.strip() or tech.strip().lower() == "nan":
            results.append("")
            continue

        make = str(row.get(make_col, "")) if make_col else ""
        model = str(row.get(model_col, "")) if model_col else ""

        query = (
            f"Verify the following automotive repair procedure. "
            f"Vehicle: {make} {model}. "
            f"Technician's description: {tech}\n\n"
            f"Search for the standard repair method and summarize whether "
            f"the technician's approach is correct."
        )

        try:
            summary = client.chat_with_web_search(
                messages=[{"role": "user", "content": query}],
                system=repair_web_search_system(),
            )
            results.append(summary)
        except Exception as e:
            failure_log.record(
                step="repair_web_verify.verify_repair_web",
                context=f"row={idx}",
                exc=e,
                lesson="Repair web search failed; row gets empty string.",
            )
            results.append("")

    return pd.DataFrame({"repair_web_verified": results}, index=df.index)
