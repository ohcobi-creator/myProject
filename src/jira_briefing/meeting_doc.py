"""Meeting document renderer.

Produces a structured Markdown (or optionally HTML/PDF) document from
analysis results and AI-generated summaries.  The Markdown flavour is
GitHub-compatible and Confluence-friendly.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import AnalysisResult, JiraIssue, MeetingConfig, RiskItem


class MeetingDocGenerator:
    """Render a complete meeting-preparation document.

    Args:
        analysis: AnalysisResult from jira_analyzer.analyze().
        issues: Full list of JiraIssue objects.
        ai_output: Dict with keys "executive_summary" and "action_items".
        config: MeetingConfig controlling title, sections, output format.
    """

    _SEV_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡"}
    _RISK_LABEL = {
        "blocked": "BLOCKED",
        "overdue": "OVERDUE",
        "unassigned": "UNASSIGNED",
        "stale": "STALE",
    }

    def __init__(
        self,
        analysis: AnalysisResult,
        issues: List[JiraIssue],
        ai_output: Dict[str, str],
        config: MeetingConfig,
    ) -> None:
        self.analysis = analysis
        self.issues = issues
        self.ai = ai_output
        self.config = config
        self._date = config.date or datetime.now().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return the full document as a string in the configured format."""
        md = self._render_markdown()
        if self.config.output_format == "markdown":
            return md
        if self.config.output_format == "html":
            return self._md_to_html(md)
        if self.config.output_format == "pdf":
            return self._md_to_pdf(md)
        return md

    def save(self, path: Optional[str] = None) -> Path:
        """Render and write to disk.  Returns the output Path."""
        out = Path(path or self.config.output_path or self._default_path())
        out.parent.mkdir(parents=True, exist_ok=True)
        content = self.render()
        out.write_text(content, encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Markdown renderer
    # ------------------------------------------------------------------

    def _render_markdown(self) -> str:
        sections = self.config.sections
        parts: list[str] = []

        parts.append(self._header())

        if "executive_summary" in sections:
            parts.append(self._section_executive_summary())

        if "sprint_dashboard" in sections and self.analysis.sprint_summary:
            parts.append(self._section_sprint_dashboard())

        if "issue_table" in sections:
            parts.append(self._section_issue_table())

        if "blockers_risks" in sections:
            parts.append(self._section_blockers_risks())

        if "workload" in sections:
            parts.append(self._section_workload())

        if "overdue" in sections and self.analysis.overdue_issues:
            parts.append(self._section_overdue())

        if "action_items" in sections:
            parts.append(self._section_action_items())

        if "appendix" in sections:
            parts.append(self._section_appendix())

        parts.append(f"\n---\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by Jira Meeting Briefing*")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self) -> str:
        return (
            f"# {self.config.title}\n\n"
            f"| | |\n"
            f"|---|---|\n"
            f"| **Date** | {self._date} |\n"
            f"| **Author** | {self.config.author} |\n"
            f"| **Issues analysed** | {self.analysis.total_issues} |\n"
            + (
                f"| **Sprint** | {self.analysis.sprint_summary.name} |\n"
                if self.analysis.sprint_summary else ""
            )
        )

    def _section_executive_summary(self) -> str:
        text = self.ai.get("executive_summary", "*Not generated.*")
        return f"## Executive Summary\n\n{text}"

    def _section_sprint_dashboard(self) -> str:
        s = self.analysis.sprint_summary
        if not s:
            return ""

        bar = _progress_bar(s.completion_pct)
        sp_bar = _progress_bar(s.sp_completion_pct) if s.total_story_points > 0 else ""

        lines = [
            "## Sprint Dashboard",
            "",
            f"**Sprint:** {s.name}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Done | {s.done_issues} |",
            f"| In Progress | {s.in_progress_issues} |",
            f"| To Do | {s.todo_issues} |",
            f"| Blocked | {s.blocked_issues} |",
            f"| **Completion** | **{s.completion_pct}%** {bar} |",
        ]
        if s.total_story_points > 0:
            lines.append(
                f"| Story Points | {s.done_story_points:.0f} / "
                f"{s.total_story_points:.0f} ({s.sp_completion_pct}%) {sp_bar} |"
            )
        return "\n".join(lines)

    def _section_issue_table(self) -> str:
        a = self.analysis
        lines = [
            "## Issue Status Overview",
            "",
            "### By Status",
            "",
            "| Status | Count |",
            "|--------|-------|",
        ]
        for status, count in sorted(a.by_status.items(), key=lambda x: -x[1]):
            lines.append(f"| {status} | {count} |")

        lines += [
            "",
            "### By Priority",
            "",
            "| Priority | Count |",
            "|----------|-------|",
        ]
        for prio, count in sorted(a.by_priority.items(), key=lambda x: -x[1]):
            lines.append(f"| {prio} | {count} |")

        lines += [
            "",
            "### By Type",
            "",
            "| Type | Count |",
            "|------|-------|",
        ]
        for itype, count in sorted(a.by_type.items(), key=lambda x: -x[1]):
            lines.append(f"| {itype} | {count} |")

        return "\n".join(lines)

    def _section_blockers_risks(self) -> str:
        risks = self.analysis.risks
        if not risks:
            return "## Blockers & Risks\n\n*No significant risks detected.*"

        lines = [
            "## Blockers & Risks",
            "",
            "| Sev | Type | Key | Summary | Assignee | Detail |",
            "|-----|------|-----|---------|----------|--------|",
        ]
        for r in risks[:20]:
            sev_icon = self._SEV_EMOJI.get(r.severity, "⚪")
            label = self._RISK_LABEL.get(r.risk_type, r.risk_type.upper())
            key_link = f"[{r.issue_key}]({r.url})" if r.url else r.issue_key
            summary = _trunc(r.summary, 60)
            detail = _trunc(r.detail, 70)
            lines.append(
                f"| {sev_icon} {r.severity} | {label} | {key_link} | {summary} | "
                f"{r.assignee or '—'} | {detail} |"
            )
        return "\n".join(lines)

    def _section_workload(self) -> str:
        workload = self.analysis.by_assignee
        if not workload:
            return ""

        lines = [
            "## Team Workload",
            "",
            "| Assignee | Total | Story Pts | Overdue |",
            "|----------|-------|-----------|---------|",
        ]
        for w in workload[:15]:
            overdue = f"**{w.overdue}** ⚠️" if w.overdue else "0"
            sp = f"{w.story_points:.0f}" if w.story_points else "—"
            lines.append(f"| {w.assignee} | {w.total} | {sp} | {overdue} |")
        return "\n".join(lines)

    def _section_overdue(self) -> str:
        issues = self.analysis.overdue_issues
        if not issues:
            return ""

        lines = [
            "## ⚠️ Overdue Issues",
            "",
            "| Key | Summary | Due | Status | Assignee |",
            "|-----|---------|-----|--------|----------|",
        ]
        for i in issues[:20]:
            key_link = f"[{i.key}]({i.url})" if i.url else i.key
            due = i.due_date.strftime("%Y-%m-%d") if i.due_date else "—"
            lines.append(
                f"| {key_link} | {_trunc(i.summary, 60)} | {due} | {i.status} | {i.assignee or '—'} |"
            )
        return "\n".join(lines)

    def _section_action_items(self) -> str:
        text = self.ai.get("action_items", "*Not generated.*")
        return f"## Action Items\n\n{text}"

    def _section_appendix(self) -> str:
        lines = [
            "## Appendix — Full Issue List",
            "",
            "| Key | Type | Priority | Status | Assignee | Summary |",
            "|-----|------|----------|--------|----------|---------|",
        ]
        for i in sorted(self.issues, key=lambda x: x.key):
            key_link = f"[{i.key}]({i.url})" if i.url else i.key
            lines.append(
                f"| {key_link} | {i.issue_type} | {i.priority or '—'} | "
                f"{i.status} | {i.assignee or '—'} | {_trunc(i.summary, 70)} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML / PDF conversion
    # ------------------------------------------------------------------

    def _md_to_html(self, md: str) -> str:
        try:
            import markdown  # type: ignore
        except ImportError:
            raise ImportError("pip install markdown to use HTML output format.")

        body = markdown.markdown(md, extensions=["tables", "fenced_code"])
        return textwrap.dedent(f"""\
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <title>{self.config.title}</title>
              <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #333; }}
                h1 {{ border-bottom: 2px solid #0052cc; padding-bottom: 8px; }}
                h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 32px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
                th {{ background: #f4f5f7; font-weight: 600; }}
                tr:nth-child(even) {{ background: #fafafa; }}
                code {{ background: #f4f5f7; padding: 2px 4px; border-radius: 3px; }}
              </style>
            </head>
            <body>
            {body}
            </body>
            </html>
        """)

    def _md_to_pdf(self, md: str) -> str:
        raise NotImplementedError(
            "PDF output requires WeasyPrint. "
            "Install it with: pip install weasyprint\n"
            "Then call: MeetingDocGenerator.save_pdf(path)"
        )

    def save_pdf(self, path: str) -> Path:
        """Render HTML and convert to PDF via WeasyPrint."""
        try:
            from weasyprint import HTML  # type: ignore
        except ImportError:
            raise ImportError("pip install weasyprint to use PDF output.")

        html = self._md_to_html(self._render_markdown())
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html).write_pdf(str(out))
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_path(self) -> str:
        ext = {"markdown": "md", "html": "html", "pdf": "pdf"}.get(
            self.config.output_format, "md"
        )
        slug = self.config.title.lower().replace(" ", "_")[:30]
        return f"data/output/meeting_{self._date}_{slug}.{ext}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _progress_bar(pct: float, width: int = 10) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled) + f" {pct:.0f}%"


def _trunc(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"
