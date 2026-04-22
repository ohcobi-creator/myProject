"""English system prompts with Anthropic prompt-caching directives.

Prompts are structured as lists of content blocks so the heavy,
reusable parts (role definition + taxonomy) can carry
`cache_control: {"type": "ephemeral"}` for >=90% cache-hit rates
inside per-row loops.

Usage in notebook cells:
    from warranty.prompts import SYMPTOM_SYSTEM, DTC_SYSTEM, ...
    result = client.chat_json(messages, schema, system=SYMPTOM_SYSTEM)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .io_utils import _repo_root

# ---------------------------------------------------------------------------
# Load taxonomy once at import time.
# ---------------------------------------------------------------------------

_TAXONOMY_TEXT: str | None = None


def _load_taxonomy() -> str:
    global _TAXONOMY_TEXT
    if _TAXONOMY_TEXT is None:
        path = _repo_root() / "config" / "taxonomy.yaml"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        lines = ["# Automotive Symptom Taxonomy (13 categories)\n"]
        for cat in data.get("categories", []):
            lines.append(f"## {cat['name']}")
            lines.append(f"Definition: {cat['definition']}")
            lines.append(f"Keywords: {', '.join(cat['keywords'])}")
            lines.append(f"Examples: {'; '.join(cat['examples'])}")
            lines.append("")
        _TAXONOMY_TEXT = "\n".join(lines)
    return _TAXONOMY_TEXT


# ---------------------------------------------------------------------------
# Reusable content-block builders (with cache_control).
# ---------------------------------------------------------------------------

# The role block is short and sets behavioral constraints.
_ROLE_BLOCK: dict[str, Any] = {
    "type": "text",
    "text": (
        "You are an expert automotive warranty engineer. You analyze dealer "
        "repair records to identify defect patterns, classify symptoms, "
        "extract DTCs, and determine root causes. Always respond in English. "
        "When asked for JSON, output ONLY valid JSON — no markdown fences, "
        "no extra commentary."
    ),
}

# The taxonomy block is large (~2 KB) and reused across every row in Step 4.
# Marking it with cache_control so the API caches it.
_TAXONOMY_BLOCK_TEMPLATE: dict[str, Any] = {
    "type": "text",
    "cache_control": {"type": "ephemeral"},
    # "text" is filled at runtime via _taxonomy_block().
}


def _taxonomy_block() -> dict[str, Any]:
    block = _TAXONOMY_BLOCK_TEMPLATE.copy()
    block["text"] = _load_taxonomy()
    return block


# ---------------------------------------------------------------------------
# Per-step system prompts (exported as module-level constants).
# ---------------------------------------------------------------------------


def symptom_system() -> list[dict[str, Any]]:
    """System prompt for Step 4 — symptom classification."""
    return [
        _ROLE_BLOCK,
        _taxonomy_block(),
        {
            "type": "text",
            "text": (
                "Given a customer complaint and technician comment, classify the "
                "issue into exactly ONE of the 13 taxonomy categories above. "
                "If insufficient information, use 'UNKNOWN'. Rate your confidence "
                "0.0-1.0 and quote the evidence from the technician comment."
            ),
        },
    ]


def dtc_system() -> list[dict[str, Any]]:
    """System prompt for Step 5 — DTC extraction and match verdict."""
    return [
        _ROLE_BLOCK,
        {
            "type": "text",
            "cache_control": {"type": "ephemeral"},
            "text": (
                "DTC code conventions:\n"
                "- P = Powertrain, C = Chassis, B = Body, U = Network/Communication\n"
                "- Format: one letter + four hex digits (e.g., P0171, C0035, B1000, U0100)\n"
                "- Codes may appear with typos (O instead of 0, l instead of 1)\n\n"
                "Given a customer complaint and technician comment:\n"
                "1. List all DTC codes found in the text (normalized uppercase).\n"
                "2. Judge whether the DTC(s) are consistent with the customer complaint "
                "AND the technician's findings.\n"
                "3. Verdict: 'Yes' (fully consistent), 'No' (contradictory), "
                "'Partial' (some match), or 'NoDTC' (no codes found).\n"
                "4. Provide 1-3 sentences of reasoning."
            ),
        },
    ]


def dtc_web_search_system() -> list[dict[str, Any]]:
    """System prompt for Step 6 — DTC web verification."""
    return [
        _ROLE_BLOCK,
        {
            "type": "text",
            "cache_control": {"type": "ephemeral"},
            "text": (
                "You have access to a web_search tool. For the given DTC code:\n"
                "1. Search for real-world conditions that set this DTC.\n"
                "2. Identify typical symptoms and common causes.\n"
                "3. Cite your sources (URLs).\n"
                "Return a concise summary in English."
            ),
        },
    ]


def repair_web_search_system() -> list[dict[str, Any]]:
    """System prompt for Step 7 — repair method verification."""
    return [
        _ROLE_BLOCK,
        {
            "type": "text",
            "cache_control": {"type": "ephemeral"},
            "text": (
                "You have access to a web_search tool. Given a technician's "
                "repair description, search for the standard repair procedure "
                "and verify whether the described actions align with industry "
                "best practice. Summarize the verified procedure and cite URLs."
            ),
        },
    ]


def root_cause_system() -> list[dict[str, Any]]:
    """System prompt for Step 8 — root-cause determination."""
    return [
        _ROLE_BLOCK,
        _taxonomy_block(),
        {
            "type": "text",
            "text": (
                "Synthesize ALL available evidence for a single warranty claim:\n"
                "- Symptom category (from Step 4)\n"
                "- DTC codes and match verdict (from Step 5)\n"
                "- DTC real-world verification (from Step 6)\n"
                "- Verified repair method (from Step 7)\n"
                "- Customer complaint text\n"
                "- Technician comment text\n\n"
                "Determine the SPECIFIC system and part that failed. Be precise — "
                "e.g., 'Powertrain/Engine — Ignition coil (cylinder 3)', not just "
                "'Engine problem'. Rate confidence 0.0-1.0 and explain reasoning."
            ),
        },
    ]


def briefing_system() -> list[dict[str, Any]]:
    """System prompt for Step 10 — final briefing generation."""
    return [
        _ROLE_BLOCK,
        {
            "type": "text",
            "cache_control": {"type": "ephemeral"},
            "text": (
                "Generate a structured English briefing for a warranty-engineering "
                "audience. Include:\n"
                "1. Executive summary (2-3 sentences)\n"
                "2. Key trends (significant at 95% CI)\n"
                "3. Top regression drivers (OR/coefficients with 95% CI)\n"
                "4. Symptom distribution highlights\n"
                "5. Top DTC findings with verified real-world cases\n"
                "6. Root-cause hypotheses ranked by evidence strength\n"
                "7. Recommended actions\n"
                "8. Limitations / caveats\n\n"
                "Use markdown headers. Reference chart filenames where relevant."
            ),
        },
    ]
