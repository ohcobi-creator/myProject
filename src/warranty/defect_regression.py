"""Defect regression modeling for warranty data (Step 2c).

Quantifies how much each input variable is associated with defect
outcomes. All models report coefficients / odds-ratios / IRRs with
95% confidence intervals and p-values at alpha = 0.05.

Models fitted:
  - Logistic regression  (binary is_defect)
  - Poisson / Negative Binomial GLM  (claim count per cohort)
  - OLS with HC3 robust SE  (days_to_failure, repair_cost)
  - Multinomial logistic  (symptom_category, after Step 4)

Each model produces a forest plot of coefficients/ORs with 95% CI
error bars saved under data/output/charts/.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from . import failure_log
from .io_utils import ColumnMap, load_settings
from .schemas import (
    DefectRegressionReport,
    RegressionModelResult,
    VariableRoleReport,
)

# ---------------------------------------------------------------------------
# Helpers
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


def _footer_text() -> str:
    return f"alpha={_alpha()} (95% CI)"


def _save_fig(fig: plt.Figure, name: str) -> str:
    path = _chart_dir() / f"step2c_{name}.png"
    fig.savefig(path, dpi=_chart_dpi(), bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# VIF check
# ---------------------------------------------------------------------------


def compute_vif(df_numeric: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor for each numeric column.

    VIF > 10 signals multicollinearity — the caller should drop or
    combine the offending predictor before fitting the regression.
    """
    from sklearn.linear_model import LinearRegression

    vif_data = []
    X = df_numeric.values
    for i, col in enumerate(df_numeric.columns):
        mask = [j for j in range(X.shape[1]) if j != i]
        if len(mask) == 0:
            vif_data.append({"feature": col, "VIF": 1.0})
            continue
        lr = LinearRegression().fit(X[:, mask], X[:, i])
        r2 = lr.score(X[:, mask], X[:, i])
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_data.append({"feature": col, "VIF": round(vif, 2)})
    return pd.DataFrame(vif_data)


# ---------------------------------------------------------------------------
# Prepare design matrix
# ---------------------------------------------------------------------------


def prepare_design_matrix(
    df: pd.DataFrame,
    var_report: VariableRoleReport,
    col_map: ColumnMap,
    target: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Build X (encoded inputs) and optionally y from the variable report.

    Categorical inputs are one-hot encoded (drop_first=True).
    Date columns are converted to ordinal integers.
    """
    input_cols = [v.name for v in var_report.inputs if v.name in df.columns]
    X = df[input_cols].copy()

    # Convert dates to ordinal.
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = pd.to_datetime(X[col], errors="coerce").map(
                lambda d: d.toordinal() if pd.notna(d) else np.nan
            )

    # One-hot encode categoricals.
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

    # Fill remaining NaN with column median (simple imputation).
    X = X.fillna(X.median())

    y = None
    if target and target in df.columns:
        y = df[target].copy()
    elif target == "days_to_failure" and col_map.repair_date and col_map.in_service_date:
        repair = pd.to_datetime(df[col_map.repair_date], errors="coerce")
        in_service = pd.to_datetime(df[col_map.in_service_date], errors="coerce")
        y = (repair - in_service).dt.days.astype(float)

    return X, y


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------


def _forest_plot(
    coef_df: pd.DataFrame, title: str, filename: str,
) -> str:
    """Draw a horizontal forest plot of estimates + 95% CI.

    coef_df must have columns: term, estimate, ci_low, ci_high, p_value.
    """
    alpha = _alpha()
    df_sorted = coef_df.sort_values("estimate", ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_sorted) * 0.35)))
    y_pos = range(len(df_sorted))
    colors = ["#d9534f" if p < alpha else "steelblue"
              for p in df_sorted["p_value"]]

    ax.barh(y_pos, df_sorted["estimate"], color=colors, edgecolor="white", height=0.6)
    ax.errorbar(
        df_sorted["estimate"], y_pos,
        xerr=[
            df_sorted["estimate"] - df_sorted["ci_low"],
            df_sorted["ci_high"] - df_sorted["estimate"],
        ],
        fmt="none", ecolor="black", capsize=2, lw=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["term"], fontsize=7)
    ax.axvline(0, ls="--", color="grey", alpha=0.5)
    ax.set_xlabel("Coefficient / OR")
    ax.set_title(title)

    # Legend: red = significant.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d9534f", label=f"p < {alpha} (significant)"),
        Patch(facecolor="steelblue", label=f"p >= {alpha}"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    fig.text(0.99, 0.01, _footer_text(), ha="right", fontsize=7, color="grey")
    return _save_fig(fig, filename)


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


def fit_logistic(
    X: pd.DataFrame, y: pd.Series, name: str = "Logistic: is_defect ~ X",
) -> RegressionModelResult:
    """Fit logistic regression and return coefficients as odds ratios."""
    import statsmodels.api as sm
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression

    alpha = _alpha()
    Xc = sm.add_constant(X)
    mask = y.notna()
    Xc, y_clean = Xc[mask], y[mask]

    try:
        model = sm.Logit(y_clean, Xc).fit(disp=0, maxiter=100)
        coef = model.params
        ci = model.conf_int(alpha=alpha)
        p_vals = model.pvalues

        coef_df = pd.DataFrame({
            "term": coef.index,
            "estimate": np.exp(coef.values),  # odds ratio
            "ci_low": np.exp(ci.iloc[:, 0].values),
            "ci_high": np.exp(ci.iloc[:, 1].values),
            "p_value": p_vals.values,
        })

        # CV metrics via sklearn.
        lr_sk = LogisticRegression(max_iter=200, solver="lbfgs", penalty=None)
        try:
            cv_auc = cross_val_score(lr_sk, X[mask], y_clean, cv=min(5, int(y_clean.sum())),
                                     scoring="roc_auc").mean()
        except Exception:
            cv_auc = float("nan")

        chart = _forest_plot(coef_df, f"Odds Ratios — {name}", "logistic_forest")
        return RegressionModelResult(
            name=name, family="logistic", target="is_defect",
            coef_table=coef_df,
            cv_metrics={"AUC_cv5": round(cv_auc, 4), "pseudo_r2": round(model.prsquared, 4)},
            chart_paths=[chart],
        )
    except Exception as e:
        failure_log.record("defect_regression.fit_logistic", name, e,
                          "Logistic fit failed; check for perfect separation or all-zero target.")
        raise


# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------


def fit_ols(
    X: pd.DataFrame, y: pd.Series, target_name: str,
) -> RegressionModelResult:
    """OLS with HC3 robust standard errors."""
    import statsmodels.api as sm

    alpha = _alpha()
    Xc = sm.add_constant(X)
    mask = y.notna()
    Xc, y_clean = Xc[mask], y[mask]

    try:
        model = sm.OLS(y_clean, Xc).fit(cov_type="HC3")
        coef = model.params
        ci = model.conf_int(alpha=alpha)
        p_vals = model.pvalues

        coef_df = pd.DataFrame({
            "term": coef.index,
            "estimate": coef.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
            "p_value": p_vals.values,
        })

        chart = _forest_plot(coef_df, f"OLS coefficients — {target_name}", f"ols_{target_name}")
        return RegressionModelResult(
            name=f"OLS: {target_name} ~ X", family="ols", target=target_name,
            coef_table=coef_df,
            cv_metrics={"r2": round(model.rsquared, 4), "r2_adj": round(model.rsquared_adj, 4)},
            chart_paths=[chart],
        )
    except Exception as e:
        failure_log.record("defect_regression.fit_ols", target_name, e,
                          "OLS fit failed; check for singular matrix or insufficient observations.")
        raise


# ---------------------------------------------------------------------------
# Poisson / Negative Binomial GLM
# ---------------------------------------------------------------------------


def fit_count_model(
    X: pd.DataFrame, y: pd.Series, target_name: str = "claim_count",
) -> RegressionModelResult:
    """Fit Poisson GLM; if overdispersed, refit as Negative Binomial."""
    import statsmodels.api as sm

    alpha = _alpha()
    Xc = sm.add_constant(X)
    mask = y.notna()
    Xc, y_clean = Xc[mask], y[mask]

    try:
        poisson_model = sm.GLM(y_clean, Xc, family=sm.families.Poisson()).fit()
        # Cameron-Trivedi dispersion test: if mean of (resid_pearson^2) >> 1 -> overdispersed.
        pearson_resid = poisson_model.resid_pearson
        dispersion = float((pearson_resid ** 2).mean())

        if dispersion > 1.5:
            # Refit as Negative Binomial.
            nb_model = sm.GLM(y_clean, Xc, family=sm.families.NegativeBinomial()).fit()
            model = nb_model
            family = "negbin"
        else:
            model = poisson_model
            family = "poisson"

        coef = model.params
        ci = model.conf_int(alpha=alpha)
        p_vals = model.pvalues

        coef_df = pd.DataFrame({
            "term": coef.index,
            "estimate": np.exp(coef.values),  # IRR
            "ci_low": np.exp(ci.iloc[:, 0].values),
            "ci_high": np.exp(ci.iloc[:, 1].values),
            "p_value": p_vals.values,
        })

        chart = _forest_plot(coef_df, f"IRR — {family} ({target_name})", f"count_{target_name}")
        return RegressionModelResult(
            name=f"{family}: {target_name} ~ X", family=family, target=target_name,
            coef_table=coef_df,
            cv_metrics={"dispersion": round(dispersion, 3)},
            chart_paths=[chart],
            notes=f"Dispersion={dispersion:.2f}; {'switched to NegBin' if family == 'negbin' else 'Poisson adequate'}.",
        )
    except Exception as e:
        failure_log.record("defect_regression.fit_count_model", target_name, e,
                          "Count-model fit failed; check for negative or non-integer target values.")
        raise


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def fit_defect_regression(
    df: pd.DataFrame,
    col_map: ColumnMap,
    var_report: VariableRoleReport,
) -> DefectRegressionReport:
    """Run all applicable regression models and return a bundled report.

    Called from notebook cell 2c. Decides which models to fit based on
    what output variables are available.
    """
    report = DefectRegressionReport()

    X, _ = prepare_design_matrix(df, var_report, col_map)
    if X.empty or len(X) < 10:
        failure_log.record("defect_regression.fit_defect_regression", "X is empty",
                          "Insufficient input columns or rows after encoding.",
                          "Check variable roles; at least 10 rows and 1 input needed.")
        return report

    # VIF check.
    num_X = X.select_dtypes(include=[np.number])
    if len(num_X.columns) >= 2:
        vif_df = compute_vif(num_X)
        report.vif_table = vif_df
        threshold = _get_settings().get("stats", {}).get("vif_threshold", 10.0)
        high_vif = vif_df[vif_df["VIF"] > threshold]
        if not high_vif.empty:
            # Drop the highest-VIF column iteratively.
            to_drop = high_vif.sort_values("VIF", ascending=False)["feature"].iloc[0]
            X = X.drop(columns=[to_drop], errors="ignore")

    # --- Logistic: every row is a defect claim, y=1 always.
    # This model only makes sense if we have a population denominator
    # (e.g., VIN counts without claims). Since we only have claims data,
    # we skip pure logistic unless a non-claim baseline is available.
    # Instead, we fit OLS and count models on continuous/count targets.

    # --- OLS: days_to_failure ---
    if col_map.repair_date and col_map.in_service_date:
        try:
            _, y_dtf = prepare_design_matrix(df, var_report, col_map, target="days_to_failure")
            if y_dtf is not None and y_dtf.notna().sum() >= 10:
                result = fit_ols(X, y_dtf, "days_to_failure")
                report.models.append(result)
        except Exception:
            pass  # already logged inside fit_ols

    # --- OLS: repair_cost ---
    if col_map.repair_cost and col_map.repair_cost in df.columns:
        try:
            y_cost = df[col_map.repair_cost].astype(float)
            if y_cost.notna().sum() >= 10:
                result = fit_ols(X, y_cost, "repair_cost")
                report.models.append(result)
        except Exception:
            pass

    # --- OLS: labor_hours ---
    if col_map.labor_hours and col_map.labor_hours in df.columns:
        try:
            y_lh = df[col_map.labor_hours].astype(float)
            if y_lh.notna().sum() >= 10:
                result = fit_ols(X, y_lh, "labor_hours")
                report.models.append(result)
        except Exception:
            pass

    # --- Count model: claims per (model, model_year) cohort ---
    if col_map.model and col_map.model_year:
        try:
            cohort = df.groupby([col_map.model, col_map.model_year]).size().reset_index(name="claim_count")
            if len(cohort) >= 5:
                X_cohort = pd.get_dummies(
                    cohort[[col_map.model, col_map.model_year]], drop_first=True, dtype=float,
                )
                result = fit_count_model(X_cohort, cohort["claim_count"], "claim_count_per_cohort")
                report.models.append(result)
        except Exception:
            pass

    return report
