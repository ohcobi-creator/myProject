"""Final briefing generation (Step 10).

Produces an English markdown briefing summarizing the entire analysis:
trend insights, regression drivers, symptom distribution, top DTCs,
root-cause hypotheses, and recommendations. The briefing is:
  - Displayed inline in the notebook.
  - Written to data/output/briefing_<timestamp>.md.
  - Inserted as a Summary sheet in the output Excel.
  - Optionally rendered as PDF via reportlab.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from . import failure_log
from .claude_client import ClaudeClient
from .io_utils import _repo_root
from .prompts import briefing_system
from .schemas import DefectRegressionReport, TrendReport, VariableRoleReport


def _build_briefing_prompt(
    df: pd.DataFrame,
    trend_report: Optional[TrendReport],
    var_report: Optional[VariableRoleReport],
    reg_report: Optional[DefectRegressionReport],
    chart_paths: list[str],
) -> str:
    """Assemble the user message for the briefing LLM call."""
    lines = ["Generate the final warranty-analysis briefing.\n"]

    # Dataset summary.
    lines.append(f"## Dataset: {len(df)} claims, {len(df.columns)} columns")

    # Symptom distribution.
    if "symptom_category" in df.columns:
        dist = df["symptom_category"].value_counts().head(10)
        lines.append("\n## Symptom distribution (top 10)")
        for cat, cnt in dist.items():
            lines.append(f"- {cat}: {cnt} ({cnt / len(df):.1%})")

    # Top DTCs.
    if "dtc_codes" in df.columns:
        all_codes = df["dtc_codes"].dropna().str.split(";").explode().str.strip()
        all_codes = all_codes[all_codes != ""]
        if not all_codes.empty:
            top_dtcs = all_codes.value_counts().head(10)
            lines.append("\n## Top DTCs")
            for dtc, cnt in top_dtcs.items():
                lines.append(f"- {dtc}: {cnt} occurrences")

    # Root-cause summary.
    if "root_cause_system" in df.columns:
        rc_dist = df["root_cause_system"].value_counts().head(10)
        lines.append("\n## Root-cause systems (top 10)")
        for sys, cnt in rc_dist.items():
            lines.append(f"- {sys}: {cnt}")

    if "root_cause_part" in df.columns:
        rp_dist = df["root_cause_part"].value_counts().head(10)
        lines.append("\n## Root-cause parts (top 10)")
        for part, cnt in rp_dist.items():
            lines.append(f"- {part}: {cnt}")

    # Significance table from trend analysis.
    if trend_report and trend_report.significance_rows:
        lines.append("\n## Significant findings (p < 0.05)")
        for row in trend_report.significance_rows:
            if row.significant_at_95pct:
                lines.append(
                    f"- {row.test_name}: {row.metric} — "
                    f"stat={row.statistic:.3f}, p={row.p_value:.4f}"
                )

    # Regression highlights.
    if reg_report:
        for model in reg_report.models:
            if model.coef_table is not None and len(model.coef_table) > 0:
                lines.append(f"\n## Regression: {model.name}")
                sig_coefs = model.coef_table[model.coef_table["p_value"] < 0.05]
                for _, c in sig_coefs.head(5).iterrows():
                    lines.append(
                        f"- {c['term']}: est={c['estimate']:.3f} "
                        f"95%CI=[{c['ci_low']:.3f}, {c['ci_high']:.3f}], "
                        f"p={c['p_value']:.4f}"
                    )

    # Charts list.
    if chart_paths:
        lines.append("\n## Charts")
        for p in chart_paths:
            lines.append(f"- {Path(p).name}")

    return "\n".join(lines)


def generate_briefing(
    df: pd.DataFrame,
    client: ClaudeClient,
    trend_report: Optional[TrendReport] = None,
    var_report: Optional[VariableRoleReport] = None,
    reg_report: Optional[DefectRegressionReport] = None,
    chart_paths: Optional[list[str]] = None,
) -> str:
    """Generate the final English markdown briefing via Claude Opus.

    Returns the markdown string. The caller (notebook cell 10) is
    responsible for saving it to disk and injecting it into the Excel
    Summary sheet.
    """
    chart_paths = chart_paths or []
    if trend_report:
        chart_paths = list(set(chart_paths + trend_report.chart_paths))
    if reg_report:
        for m in reg_report.models:
            chart_paths = list(set(chart_paths + m.chart_paths))

    prompt = _build_briefing_prompt(df, trend_report, var_report, reg_report, chart_paths)

    try:
        md = client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=briefing_system(),
            model_override=client.heavy_model,
            max_tokens=4096,
        )
        return md
    except Exception as e:
        failure_log.record(
            step="briefing.generate_briefing",
            context="Final briefing LLM call",
            exc=e,
            lesson="Briefing generation failed; returning a raw data summary instead.",
        )
        return prompt  # fallback: return the raw data summary


def save_briefing_md(markdown: str, output_dir: Optional[Path] = None) -> Path:
    """Write the briefing markdown to data/output/briefing_<ts>.md."""
    output_dir = output_dir or (_repo_root() / "data" / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"briefing_{ts}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def save_briefing_pdf(markdown: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """Render the briefing as a simple PDF. Returns None if reportlab missing."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        return None

    output_dir = output_dir or (_repo_root() / "data" / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"briefing_{ts}.pdf"

    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph(f"• {line[2:]}", styles["BodyText"]))
        elif line.strip():
            story.append(Paragraph(line, styles["BodyText"]))
        else:
            story.append(Spacer(1, 6))

    try:
        doc.build(story)
        return path
    except Exception as e:
        failure_log.record(
            step="briefing.save_briefing_pdf",
            context="PDF rendering",
            exc=e,
            lesson="PDF generation failed; markdown is still available.",
        )
        return None
