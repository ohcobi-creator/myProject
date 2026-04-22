"""Pydantic schemas for structured LLM outputs and internal reports.

Every LLM call that returns a JSON payload is validated against one of the
models in this module. This gives us typed access to fields downstream and
lets us catch malformed responses early (the LLM is then asked to self-
correct once before we log to docs/FAILURE_LOG.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Step 4 — Symptom binning
# ---------------------------------------------------------------------------


class SymptomResult(BaseModel):
    """Structured output of the per-row symptom classifier (Step 4)."""

    symptom_category: str = Field(
        ...,
        description="One of the 13 taxonomy categories, or 'UNKNOWN' if insufficient info.",
    )
    symptom_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model-estimated confidence in [0,1]."
    )
    symptom_evidence: str = Field(
        ..., description="Short quote/summary from the tech comment supporting the choice."
    )


# ---------------------------------------------------------------------------
# Step 5 — DTC extraction & match verdict
# ---------------------------------------------------------------------------

MatchLabel = Literal["Yes", "No", "Partial", "NoDTC"]


class DtcMatchVerdict(BaseModel):
    """Whether the extracted DTC(s) are consistent with the complaint+comment."""

    dtc_codes: list[str] = Field(default_factory=list, description="Normalized uppercase DTCs.")
    dtc_match_verdict: MatchLabel = Field(..., description="Yes / No / Partial / NoDTC.")
    reasoning: str = Field(..., description="1-3 sentence justification in English.")


# ---------------------------------------------------------------------------
# Step 6 — DTC web verification
# ---------------------------------------------------------------------------


class DtcWebVerification(BaseModel):
    """Web-search summary for a single DTC code."""

    dtc_code: str
    set_conditions: str = Field(..., description="Real-world conditions that set this DTC.")
    typical_symptoms: str
    common_causes: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list, description="URLs backing the summary.")


# ---------------------------------------------------------------------------
# Step 7 — Repair verification
# ---------------------------------------------------------------------------


class RepairVerification(BaseModel):
    """Verified repair-method summary for a technician comment."""

    repair_actions: list[str] = Field(default_factory=list)
    web_verified_procedure: str = Field(..., description="Concise verified procedure summary.")
    citations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 8 — Root cause
# ---------------------------------------------------------------------------


class RootCause(BaseModel):
    """Final system/part determination synthesizing all prior evidence."""

    root_cause_system: str
    root_cause_part: str
    root_cause_confidence: float = Field(..., ge=0.0, le=1.0)
    root_cause_reasoning: str


# ---------------------------------------------------------------------------
# Step 2a — Variable role report (non-LLM dataclass)
# ---------------------------------------------------------------------------


@dataclass
class VarSpec:
    """Single column's role assignment + rationale."""

    name: str
    role: Literal["input", "output", "excluded"]
    dtype: str
    reason: str
    n_unique: int
    pct_missing: float
    notes: str = ""


@dataclass
class VariableRoleReport:
    """Bundle of all variable assignments for Step 2a."""

    inputs: list[VarSpec] = field(default_factory=list)
    outputs: list[VarSpec] = field(default_factory=list)
    excluded: list[VarSpec] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the report as a Markdown table for notebook + Summary sheet."""
        lines = [
            "# Variable Role & Rationale",
            "",
            "Each column in the input data was auto-classified as an **input (X)**,",
            "**output (Y)**, or **excluded** (id/meta/leakage). The rationale column",
            "explains *why*.",
            "",
            "| Column | Role | dtype | n_unique | % missing | Reason | Notes |",
            "|---|---|---|---:|---:|---|---|",
        ]
        for v in self.inputs + self.outputs + self.excluded:
            lines.append(
                f"| `{v.name}` | {v.role} | {v.dtype} | {v.n_unique} | "
                f"{v.pct_missing:.1%} | {v.reason} | {v.notes} |"
            )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Step 2b — Significance table
# ---------------------------------------------------------------------------


@dataclass
class SignificanceRow:
    """Single row of the significance table used by Step 2b."""

    test_name: str
    metric: str
    statistic: float
    p_value: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    effect_size: Optional[float]
    significant_at_95pct: bool
    note: str = ""


@dataclass
class TrendReport:
    """Aggregated result of Step 2b trend analysis."""

    significance_rows: list[SignificanceRow] = field(default_factory=list)
    chart_paths: list[str] = field(default_factory=list)
    summaries: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 2c — Defect regression report
# ---------------------------------------------------------------------------


@dataclass
class RegressionModelResult:
    """One fitted regression model with key metrics for reporting."""

    name: str  # e.g., "Logistic: is_defect ~ X"
    family: str  # "logistic" | "poisson" | "negbin" | "ols" | "cox" | "multinomial"
    target: str
    coef_table: Any  # pandas.DataFrame with columns: term, estimate, ci_low, ci_high, p_value
    cv_metrics: dict[str, float] = field(default_factory=dict)
    chart_paths: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DefectRegressionReport:
    """Bundle of all Step 2c regression models."""

    models: list[RegressionModelResult] = field(default_factory=list)
    vif_table: Any | None = None  # pandas.DataFrame
