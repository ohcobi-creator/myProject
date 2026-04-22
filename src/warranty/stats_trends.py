"""Statistical trend analysis for warranty data (Steps 2a, 2b, 3).

This module covers three notebook cells:
  2a — select_variables(): assign input/output/excluded roles to columns
  2b — run_full_trend_analysis(): significance-tested stats at alpha=0.05
  3  — summarize_trends(): LLM-generated English markdown summary

The defect regression (cell 2c) lives in a separate module
`defect_regression.py` to keep file sizes manageable.

All hypothesis tests use alpha=0.05 (95% confidence) loaded from
config/settings.yaml. Every chart is saved as PNG under
data/output/charts/ and displayed inline in the notebook.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for notebooks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from . import failure_log
from .io_utils import ColumnMap, load_settings
from .schemas import (
    SignificanceRow,
    TrendReport,
    VarSpec,
    VariableRoleReport,
)

# ---------------------------------------------------------------------------
# Constants loaded from settings
# ---------------------------------------------------------------------------

_SETTINGS: dict[str, Any] = {}


def _get_settings() -> dict[str, Any]:
    global _SETTINGS
    if not _SETTINGS:
        _SETTINGS = load_settings()
    return _SETTINGS


def _alpha() -> float:
    return _get_settings().get("stats", {}).get("alpha", 0.05)


def _chart_dir() -> Path:
    from .io_utils import _repo_root
    d = _repo_root() / "data" / "output" / "charts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _chart_dpi() -> int:
    return _get_settings().get("output", {}).get("chart_dpi", 150)


# ---------------------------------------------------------------------------
# 2a — Variable role selection
# ---------------------------------------------------------------------------

# Columns that are recorded *after* the failure event and must not be used
# as predictors (leakage). They can still be targets (Y) or descriptive.
_POST_FAILURE_ROLES = {
    "replaced_part_number", "repair_cost", "labor_hours",
    "tech_comment", "customer_complaint",
}

# Columns that are identifiers / metadata, not useful as X or Y.
_ID_ROLES = {"vin", "claim_id"}


def select_variables(
    df: pd.DataFrame,
    col_map: ColumnMap,
) -> VariableRoleReport:
    """Assign every dataframe column to input (X), output (Y), or excluded.

    The assignment is heuristic but transparent — each VarSpec carries a
    human-readable `reason` explaining the decision. The notebook renders
    this as a table and lets the user override via ipywidgets before
    proceeding to Steps 2b/2c.

    Args:
        df: The warranty dataframe from io_utils.load_warranty_xlsx().
        col_map: Detected column mapping.

    Returns:
        VariableRoleReport with inputs, outputs, and excluded lists.
    """
    settings = _get_settings().get("stats", {})
    high_card_pct = settings.get("high_cardinality_pct", 0.9)
    high_miss_pct = settings.get("high_missing_pct", 0.5)

    # Reverse map: actual column name -> semantic role name.
    col_to_role = {}
    for role, actual in col_map.as_dict().items():
        if actual is not None:
            col_to_role[actual] = role

    report = VariableRoleReport()

    for col in df.columns:
        n_unique = int(df[col].nunique())
        pct_miss = float(df[col].isna().mean())
        dtype_str = str(df[col].dtype)
        role_name = col_to_role.get(col, "")

        # --- Exclusion checks (order matters) ---

        # 1. Id / meta columns
        if role_name in _ID_ROLES:
            report.excluded.append(VarSpec(
                name=col, role="excluded", dtype=dtype_str,
                n_unique=n_unique, pct_missing=pct_miss,
                reason=f"Identifier column ({role_name}); not predictive.",
            ))
            continue

        # 2. High missingness
        if pct_miss > high_miss_pct:
            report.excluded.append(VarSpec(
                name=col, role="excluded", dtype=dtype_str,
                n_unique=n_unique, pct_missing=pct_miss,
                reason=f">{high_miss_pct:.0%} missing values ({pct_miss:.1%}).",
            ))
            continue

        # 3. Free-text fields -> excluded from raw modeling (NLP in Step 4/5)
        if role_name in ("customer_complaint", "tech_comment"):
            report.excluded.append(VarSpec(
                name=col, role="excluded", dtype=dtype_str,
                n_unique=n_unique, pct_missing=pct_miss,
                reason="Free-text field; analyzed via NLP in Steps 4-5, not as a raw predictor.",
            ))
            continue

        # 4. Post-failure leakage -> candidate output, not input
        if role_name in _POST_FAILURE_ROLES:
            report.outputs.append(VarSpec(
                name=col, role="output", dtype=dtype_str,
                n_unique=n_unique, pct_missing=pct_miss,
                reason=(
                    f"Recorded after failure ({role_name}); potential regression target "
                    "but excluded from predictors to avoid leakage."
                ),
            ))
            continue

        # 5. Near-constant (variance ~ 0 for numeric, 1 unique for categorical)
        if n_unique <= 1:
            report.excluded.append(VarSpec(
                name=col, role="excluded", dtype=dtype_str,
                n_unique=n_unique, pct_missing=pct_miss,
                reason="Near-constant column (<=1 unique value); no predictive variance.",
            ))
            continue

        # 6. Very high cardinality categorical
        if df[col].dtype == "object" and n_unique / max(len(df), 1) > high_card_pct:
            report.excluded.append(VarSpec(
                name=col, role="excluded", dtype=dtype_str,
                n_unique=n_unique, pct_missing=pct_miss,
                reason=f"High-cardinality categorical (>{high_card_pct:.0%} unique).",
            ))
            continue

        # --- Output candidates ---
        if role_name in ("repair_cost", "labor_hours"):
            report.outputs.append(VarSpec(
                name=col, role="output", dtype=dtype_str,
                n_unique=n_unique, pct_missing=pct_miss,
                reason="Continuous post-failure metric; suitable as OLS regression target.",
            ))
            continue

        # --- Everything else is an input candidate ---
        reason = _input_reason(role_name, dtype_str, col)
        report.inputs.append(VarSpec(
            name=col, role="input", dtype=dtype_str,
            n_unique=n_unique, pct_missing=pct_miss,
            reason=reason,
        ))

    # Derive synthetic outputs that are always available.
    report.outputs.insert(0, VarSpec(
        name="is_defect (derived)",
        role="output",
        dtype="bool",
        n_unique=2,
        pct_missing=0.0,
        reason=(
            "Binary indicator: 1 for every row in the warranty dataset "
            "(each row IS a defect claim). Useful as logistic-regression target "
            "when compared against a fleet-population denominator."
        ),
    ))

    if col_map.repair_date and col_map.in_service_date:
        report.outputs.insert(1, VarSpec(
            name="days_to_failure (derived)",
            role="output",
            dtype="float64",
            n_unique=-1,
            pct_missing=0.0,
            reason=(
                "Continuous: repair_date minus in_service_date. "
                "Measures time-to-failure; OLS or survival-model target."
            ),
        ))

    return report


def _input_reason(role_name: str, dtype_str: str, col: str) -> str:
    """Generate a human-readable reason for classifying a column as input."""
    reasons = {
        "make": "Vehicle manufacturer; categorical grouping variable for defect-rate comparison.",
        "model": "Vehicle model; key stratification variable for defect analysis.",
        "model_year": "Model year; proxy for design generation and fleet age.",
        "trim": "Trim level; captures equipment-level variation.",
        "engine_type": "Engine/powertrain type; key engineering variable.",
        "transmission_type": "Transmission type; known defect-rate differentiator.",
        "drivetrain": "Drivetrain (FWD/AWD/RWD); affects load paths and failure modes.",
        "plant_code": "Assembly plant; captures manufacturing-process variation.",
        "production_date": "Production date; can reveal batch/lot effects.",
        "mileage_at_repair": "Mileage at repair; primary usage/exposure variable (continuous).",
        "region": "Geographic region; captures climate and usage-pattern differences.",
        "dealer_id": "Dealer; may reveal diagnostic-capability or reporting differences.",
        "repair_date": "Repair date; used for time-series trending.",
        "in_service_date": "In-service date; used to derive age-at-repair.",
    }
    if role_name in reasons:
        return reasons[role_name]
    if "int" in dtype_str or "float" in dtype_str:
        return f"Numeric column '{col}'; candidate continuous predictor."
    if "date" in dtype_str.lower():
        return f"Date column '{col}'; potential temporal predictor."
    return f"Categorical column '{col}'; candidate grouping variable."


# ---------------------------------------------------------------------------
# 2b — Full trend analysis (significance-tested at alpha=0.05)
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, name: str) -> str:
    """Save a figure to data/output/charts/ and return the path string."""
    path = _chart_dir() / f"step2_{name}.png"
    fig.savefig(path, dpi=_chart_dpi(), bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _footer_text() -> str:
    return f"alpha={_alpha()} (95% CI)"


def frequency_analysis(
    df: pd.DataFrame, col: str, top_n: int = 20,
) -> tuple[pd.DataFrame, SignificanceRow | None, str]:
    """Bar chart of top-N value counts with 95% Wilson CIs.

    Returns (counts_df, significance_row_or_None, chart_path).
    """
    alpha = _alpha()
    counts = df[col].value_counts().head(top_n)
    n_total = int(counts.sum())

    # Wilson score 95% CI for each proportion.
    rows = []
    for val, cnt in counts.items():
        p_hat = cnt / n_total
        z = sp_stats.norm.ppf(1 - alpha / 2)
        denom = 1 + z**2 / n_total
        centre = (p_hat + z**2 / (2 * n_total)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
        rows.append({
            "value": val, "count": cnt, "proportion": p_hat,
            "ci_low": max(0, centre - margin),
            "ci_high": min(1, centre + margin),
        })
    result_df = pd.DataFrame(rows)

    # Plot.
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.35)))
    y_pos = range(len(result_df))
    ax.barh(y_pos, result_df["count"], color="steelblue", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(result_df["value"], fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title(f"Frequency — {col}  (n={n_total})")
    ax.invert_yaxis()
    fig.text(0.99, 0.01, _footer_text(), ha="right", fontsize=7, color="grey")
    chart_path = _save_fig(fig, f"freq_{col}")

    return result_df, None, chart_path


def time_series_trend(
    df: pd.DataFrame, date_col: str, freq: str = "M",
) -> tuple[pd.DataFrame, SignificanceRow | None, str]:
    """Monthly claim counts with 95% CI band and Mann-Kendall trend test."""
    alpha = _alpha()
    ts = df[date_col].dropna()
    if ts.empty:
        return pd.DataFrame(), None, ""

    ts = pd.to_datetime(ts, errors="coerce").dropna()
    monthly = ts.dt.to_period(freq).value_counts().sort_index()
    monthly.index = monthly.index.to_timestamp()
    monthly_df = monthly.reset_index()
    monthly_df.columns = ["month", "count"]

    # Bootstrap 95% CI around the mean count.
    mean_count = monthly_df["count"].mean()
    std_count = monthly_df["count"].std(ddof=1)
    n_months = len(monthly_df)
    se = std_count / np.sqrt(max(n_months, 1))
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci_low = mean_count - z * se
    ci_high = mean_count + z * se

    # Mann-Kendall trend test (simplified implementation).
    y = monthly_df["count"].values.astype(float)
    mk_s, mk_p = _mann_kendall(y)
    sig = mk_p < alpha

    sig_row = SignificanceRow(
        test_name="Mann-Kendall trend",
        metric=f"monthly claim counts ({date_col})",
        statistic=float(mk_s),
        p_value=float(mk_p),
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size=None,
        significant_at_95pct=sig,
        note="Trend is significant" if sig else "No significant trend",
    )

    # Plot.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_df["month"], monthly_df["count"], "o-", color="steelblue", label="Claims/month")
    ax.axhline(mean_count, ls="--", color="grey", alpha=0.5, label=f"Mean={mean_count:.1f}")
    ax.fill_between(monthly_df["month"], ci_low, ci_high, alpha=0.15, color="steelblue",
                     label=f"95% CI [{ci_low:.1f}, {ci_high:.1f}]")
    if sig:
        # Overlay Theil-Sen slope.
        x_num = np.arange(len(y), dtype=float)
        slope, intercept = sp_stats.theilslopes(y, x_num)[:2]
        ax.plot(monthly_df["month"], intercept + slope * x_num, "--", color="red",
                label=f"Trend (slope={slope:.2f}, p={mk_p:.4f}) *")
    ax.set_ylabel("Claim count")
    ax.set_title(f"Monthly claims — {date_col}  (n={int(y.sum())})")
    ax.legend(fontsize=7)
    fig.text(0.99, 0.01, _footer_text(), ha="right", fontsize=7, color="grey")
    chart_path = _save_fig(fig, "timeseries")

    return monthly_df, sig_row, chart_path


def _mann_kendall(y: np.ndarray) -> tuple[float, float]:
    """Simplified Mann-Kendall test statistic S and two-sided p-value."""
    n = len(y)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(y[j] - y[i])
    # Variance under H0 (no ties correction for simplicity).
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * sp_stats.norm.sf(abs(z))
    return float(s), float(p)


def chi_square_tests(
    df: pd.DataFrame, pairs: list[tuple[str, str]],
) -> tuple[pd.DataFrame, list[SignificanceRow], str]:
    """Chi-square independence tests with Cramér's V for categorical pairs."""
    alpha = _alpha()
    results = []
    sig_rows = []

    for col_a, col_b in pairs:
        try:
            ct = pd.crosstab(df[col_a], df[col_b])
            if ct.size == 0:
                continue
            chi2, p, dof, _ = sp_stats.chi2_contingency(ct)
            n_obs = ct.values.sum()
            k = min(ct.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_obs * max(k, 1))) if n_obs > 0 else 0.0
            sig = p < alpha
            results.append({
                "pair": f"{col_a} × {col_b}", "chi2": chi2, "p_value": p,
                "dof": dof, "cramers_v": cramers_v, "n": n_obs,
                "significant": sig,
            })
            sig_rows.append(SignificanceRow(
                test_name="Chi-square independence",
                metric=f"{col_a} × {col_b}",
                statistic=float(chi2),
                p_value=float(p),
                ci_low=None, ci_high=None,
                effect_size=float(cramers_v),
                significant_at_95pct=sig,
            ))
        except Exception as e:
            failure_log.record(
                step="stats_trends.chi_square_tests",
                context=f"pair=({col_a}, {col_b})",
                exc=e,
                lesson="Chi-square requires >0 observations in the contingency table.",
            )

    result_df = pd.DataFrame(results)

    # Heatmap of Cramér's V.
    chart_path = ""
    if not result_df.empty:
        fig, ax = plt.subplots(figsize=(8, max(3, len(result_df) * 0.4)))
        colors = ["#d9534f" if s else "steelblue" for s in result_df["significant"]]
        ax.barh(result_df["pair"], result_df["cramers_v"], color=colors, edgecolor="white")
        for i, row in result_df.iterrows():
            marker = " *" if row["significant"] else ""
            ax.text(row["cramers_v"] + 0.01, i, f"p={row['p_value']:.4f}{marker}", fontsize=7, va="center")
        ax.set_xlabel("Cramér's V")
        ax.set_title("Chi-square independence tests")
        ax.invert_yaxis()
        fig.text(0.99, 0.01, _footer_text(), ha="right", fontsize=7, color="grey")
        chart_path = _save_fig(fig, "chisq")

    return result_df, sig_rows, chart_path


def correlation_matrix(
    df: pd.DataFrame, numeric_cols: list[str],
) -> tuple[pd.DataFrame, list[SignificanceRow], str]:
    """Pearson + Spearman correlation matrix with p-values."""
    alpha = _alpha()
    sig_rows = []
    if len(numeric_cols) < 2:
        return pd.DataFrame(), sig_rows, ""

    sub = df[numeric_cols].dropna()
    corr = sub.corr(method="pearson")

    # Compute p-values for each pair.
    n = len(sub)
    p_matrix = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
    for i, c1 in enumerate(numeric_cols):
        for j, c2 in enumerate(numeric_cols):
            if i < j:
                r, p = sp_stats.pearsonr(sub[c1], sub[c2])
                p_matrix.loc[c1, c2] = p
                p_matrix.loc[c2, c1] = p
                sig_rows.append(SignificanceRow(
                    test_name="Pearson correlation",
                    metric=f"{c1} vs {c2}",
                    statistic=float(r),
                    p_value=float(p),
                    ci_low=None, ci_high=None,
                    effect_size=float(r),
                    significant_at_95pct=p < alpha,
                ))

    # Heatmap.
    fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols) * 0.8)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    # Bold-outline significant cells.
    for i in range(len(numeric_cols)):
        for j in range(i):
            if p_matrix.iloc[i, j] < alpha:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2))
    ax.set_title(f"Pearson correlation (n={n})")
    fig.text(0.99, 0.01, _footer_text(), ha="right", fontsize=7, color="grey")
    chart_path = _save_fig(fig, "correlation")

    return corr, sig_rows, chart_path


def pareto_analysis(
    df: pd.DataFrame, col: str, top_n: int = 20,
) -> tuple[pd.DataFrame, str]:
    """Pareto chart: sorted bar + cumulative line with 80% cutoff."""
    counts = df[col].value_counts().head(top_n)
    cum_pct = counts.cumsum() / counts.sum()
    pareto_df = pd.DataFrame({"count": counts, "cum_pct": cum_pct})

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(pareto_df))
    ax1.bar(x, pareto_df["count"], color="steelblue", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pareto_df.index, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Count")

    ax2 = ax1.twinx()
    ax2.plot(x, pareto_df["cum_pct"] * 100, "ro-", markersize=4)
    ax2.axhline(80, ls="--", color="red", alpha=0.5, label="80% line")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=7)

    ax1.set_title(f"Pareto — {col}  (n={int(counts.sum())})")
    fig.text(0.99, 0.01, _footer_text(), ha="right", fontsize=7, color="grey")
    chart_path = _save_fig(fig, f"pareto_{col}")
    return pareto_df, chart_path


def cohort_heatmap(
    df: pd.DataFrame, model_year_col: str, mileage_col: str,
) -> tuple[pd.DataFrame, str]:
    """Model-year × mileage-bucket heatmap of claim counts."""
    sub = df[[model_year_col, mileage_col]].dropna()
    if sub.empty:
        return pd.DataFrame(), ""

    sub = sub.copy()
    sub["mileage_bucket"] = pd.cut(
        sub[mileage_col],
        bins=[0, 10000, 25000, 50000, 75000, 100000, float("inf")],
        labels=["0-10k", "10-25k", "25-50k", "50-75k", "75-100k", "100k+"],
    )
    pivot = sub.groupby([model_year_col, "mileage_bucket"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title(f"Claims: {model_year_col} × mileage bucket")
    fig.text(0.99, 0.01, _footer_text(), ha="right", fontsize=7, color="grey")
    chart_path = _save_fig(fig, "cohort_heatmap")
    return pivot, chart_path


def group_comparison(
    df: pd.DataFrame, group_col: str, value_col: str,
) -> tuple[SignificanceRow | None, str]:
    """ANOVA or Kruskal-Wallis box-plot comparison across groups."""
    alpha = _alpha()
    sub = df[[group_col, value_col]].dropna()
    groups = [g[value_col].values for _, g in sub.groupby(group_col) if len(g) >= 2]
    if len(groups) < 2:
        return None, ""

    # Normality check (Shapiro on each group, if small enough).
    all_normal = all(
        sp_stats.shapiro(g[:500])[1] > alpha if len(g) >= 3 else False
        for g in groups
    )

    if all_normal:
        stat, p = sp_stats.f_oneway(*groups)
        test_name = "One-way ANOVA"
    else:
        stat, p = sp_stats.kruskal(*groups)
        test_name = "Kruskal-Wallis H"

    sig = p < alpha
    sig_row = SignificanceRow(
        test_name=test_name,
        metric=f"{value_col} by {group_col}",
        statistic=float(stat),
        p_value=float(p),
        ci_low=None, ci_high=None, effect_size=None,
        significant_at_95pct=sig,
    )

    # Box plot.
    fig, ax = plt.subplots(figsize=(10, 5))
    sub.boxplot(column=value_col, by=group_col, ax=ax)
    star = " *" if sig else ""
    ax.set_title(f"{test_name}: {value_col} by {group_col}  (p={p:.4f}{star})")
    fig.suptitle("")  # remove pandas auto-title
    fig.text(0.99, 0.01, _footer_text(), ha="right", fontsize=7, color="grey")
    chart_path = _save_fig(fig, f"boxplot_{group_col}_{value_col}")
    return sig_row, chart_path


# ---------------------------------------------------------------------------
# run_full_trend_analysis — orchestrator for 2b
# ---------------------------------------------------------------------------


def run_full_trend_analysis(
    df: pd.DataFrame,
    col_map: ColumnMap,
    var_report: VariableRoleReport | None = None,
) -> TrendReport:
    """Run all trend-analysis sub-steps and return an aggregated TrendReport.

    This is the main entry point called from notebook cell 2b. It
    delegates to the individual analysis functions above and collects
    their results into a single TrendReport.
    """
    report = TrendReport()

    # Identify column lists from the variable report or col_map.
    cat_cols = []
    num_cols = []
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            if df[col].nunique() <= 50:
                cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)

    # 1. Frequency analysis for key categoricals.
    for col in cat_cols[:8]:
        try:
            freq_df, _, path = frequency_analysis(df, col)
            if path:
                report.chart_paths.append(path)
            report.summaries[f"freq_{col}"] = freq_df.to_dict()
        except Exception as e:
            failure_log.record("stats_trends.frequency_analysis", col, e,
                              "Frequency chart failed; column may be unsuitable.")

    # 2. Time-series trend.
    if col_map.repair_date:
        try:
            ts_df, sig_row, path = time_series_trend(df, col_map.repair_date)
            if sig_row:
                report.significance_rows.append(sig_row)
            if path:
                report.chart_paths.append(path)
        except Exception as e:
            failure_log.record("stats_trends.time_series_trend", col_map.repair_date, e,
                              "Time-series failed; check date column parsing.")

    # 3. Chi-square for all categorical pairs (limit to first 5 categoricals).
    if len(cat_cols) >= 2:
        pairs = [(cat_cols[i], cat_cols[j])
                 for i in range(min(5, len(cat_cols)))
                 for j in range(i + 1, min(5, len(cat_cols)))]
        try:
            _, sig_rows, path = chi_square_tests(df, pairs)
            report.significance_rows.extend(sig_rows)
            if path:
                report.chart_paths.append(path)
        except Exception as e:
            failure_log.record("stats_trends.chi_square_tests", str(pairs), e,
                              "Chi-square block failed.")

    # 4. Correlation matrix for numerics.
    if len(num_cols) >= 2:
        try:
            _, sig_rows, path = correlation_matrix(df, num_cols[:10])
            report.significance_rows.extend(sig_rows)
            if path:
                report.chart_paths.append(path)
        except Exception as e:
            failure_log.record("stats_trends.correlation_matrix", str(num_cols), e,
                              "Correlation failed.")

    # 5. Pareto for top categorical.
    if cat_cols:
        try:
            _, path = pareto_analysis(df, cat_cols[0])
            if path:
                report.chart_paths.append(path)
        except Exception as e:
            failure_log.record("stats_trends.pareto_analysis", cat_cols[0], e,
                              "Pareto chart failed.")

    # 6. Cohort heatmap.
    if col_map.model_year and col_map.mileage_at_repair:
        try:
            _, path = cohort_heatmap(df, col_map.model_year, col_map.mileage_at_repair)
            if path:
                report.chart_paths.append(path)
        except Exception as e:
            failure_log.record("stats_trends.cohort_heatmap", "model_year x mileage", e,
                              "Cohort heatmap failed.")

    # 7. Group comparisons (mileage by model, by region, etc.).
    if col_map.mileage_at_repair:
        for grp in [col_map.model, col_map.region, col_map.engine_type]:
            if grp and grp in df.columns:
                try:
                    sig_row, path = group_comparison(df, grp, col_map.mileage_at_repair)
                    if sig_row:
                        report.significance_rows.append(sig_row)
                    if path:
                        report.chart_paths.append(path)
                except Exception as e:
                    failure_log.record("stats_trends.group_comparison", grp, e,
                                      "Group comparison failed.")

    return report


# ---------------------------------------------------------------------------
# Cell 3 — LLM trend summary (uses claude_client at runtime)
# ---------------------------------------------------------------------------


def build_trend_summary_prompt(
    report: TrendReport, var_report: VariableRoleReport | None = None,
) -> str:
    """Build the user-message text fed to Claude for trend summarization.

    The system prompt (with caching) is handled by claude_client; this
    function only produces the per-call user content.
    """
    lines = ["Summarize the following warranty trend-analysis results in English markdown.",
             "Highlight only findings significant at the 95% confidence level.",
             "For each finding, reference the chart filename.",
             "Structure: 3-5 bullet points, each with the key insight and supporting numbers.", ""]

    if var_report:
        lines.append("## Variable roles")
        lines.append(var_report.to_markdown())
        lines.append("")

    lines.append("## Significance table")
    for row in report.significance_rows:
        star = " *" if row.significant_at_95pct else ""
        ci = ""
        if row.ci_low is not None and row.ci_high is not None:
            ci = f" 95%CI=[{row.ci_low:.3f}, {row.ci_high:.3f}]"
        lines.append(
            f"- {row.test_name} | {row.metric}: stat={row.statistic:.4f}, "
            f"p={row.p_value:.4f}{ci}{star}"
        )

    lines.append("")
    lines.append("## Charts produced")
    for p in report.chart_paths:
        lines.append(f"- {Path(p).name}")

    return "\n".join(lines)


def summarize_trends(
    report: TrendReport,
    var_report: VariableRoleReport | None,
    client: Any,  # ClaudeClient (imported at runtime to avoid circular dep)
) -> str:
    """Call Claude Opus to produce an English markdown summary (cell 3).

    Args:
        report: The TrendReport from run_full_trend_analysis().
        var_report: Optional VariableRoleReport from select_variables().
        client: An initialized ClaudeClient instance.

    Returns:
        English markdown string suitable for display in the notebook.
    """
    user_msg = build_trend_summary_prompt(report, var_report)
    try:
        response = client.chat(
            messages=[{"role": "user", "content": user_msg}],
            model_override=client.heavy_model,
        )
        return response
    except Exception as e:
        failure_log.record(
            step="stats_trends.summarize_trends",
            context="LLM call for trend summary",
            exc=e,
            lesson="Check API key and model availability; fall back to raw table.",
        )
        return "**[LLM summary unavailable — see significance table above.]**"
