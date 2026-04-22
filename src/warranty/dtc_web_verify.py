"""DTC web-search verification (Step 6).

For each unique DTC code found in Step 5, uses Claude's web_search
server tool to look up real-world set conditions, typical symptoms,
and common causes. Results are memoized so each DTC is searched only
once even if it appears in many rows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from . import failure_log
from .claude_client import ClaudeClient
from .prompts import dtc_web_search_system


def _verify_single_dtc(
    dtc: str,
    client: ClaudeClient,
    make: str = "",
    model: str = "",
) -> str:
    """Web-search a single DTC and return a concise summary with citations."""
    query = (
        f"Search for OBD-II DTC {dtc}: what conditions set this code, "
        f"typical symptoms, and common root causes."
    )
    if make or model:
        query += f" Focus on {make} {model} if possible."

    return client.chat_with_web_search(
        messages=[{"role": "user", "content": query}],
        system=dtc_web_search_system(),
    )


def verify_dtc_web(
    df: pd.DataFrame,
    client: ClaudeClient,
    dtc_col: str = "dtc_codes",
    make_col: str | None = None,
    model_col: str | None = None,
) -> pd.DataFrame:
    """Verify every unique DTC via web search; write results per row.

    Returns a single-column DataFrame: dtc_web_verified_case.
    """
    cache: dict[str, str] = {}

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Step 6: DTC web verify"):
        raw_codes = str(row.get(dtc_col, ""))
        codes = [c.strip() for c in raw_codes.split(";") if c.strip()]

        if not codes:
            results.append("")
            continue

        make = str(row.get(make_col, "")) if make_col and make_col in df.columns else ""
        model = str(row.get(model_col, "")) if model_col and model_col in df.columns else ""

        parts = []
        for dtc in codes:
            if dtc not in cache:
                try:
                    cache[dtc] = _verify_single_dtc(dtc, client, make, model)
                except Exception as e:
                    failure_log.record(
                        step="dtc_web_verify.verify_dtc_web",
                        context=f"dtc={dtc}",
                        exc=e,
                        lesson="Web search failed for this DTC; row gets 'Unavailable'.",
                    )
                    cache[dtc] = f"[Web search unavailable for {dtc}]"
            parts.append(f"[{dtc}] {cache[dtc]}")

        results.append(" | ".join(parts))

    return pd.DataFrame({"dtc_web_verified_case": results}, index=df.index)
