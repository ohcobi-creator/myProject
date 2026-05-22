"""Claude prompt templates for Jira analysis and meeting preparation."""

from __future__ import annotations

from typing import List

from .schemas import AnalysisResult, JiraIssue, MeetingConfig

SYSTEM_PROMPT = (
    "You are an expert engineering program manager and agile coach. "
    "Analyze Jira project data and generate concise, actionable meeting preparation materials. "
    "Focus on: blockers, risks, progress against plan, workload balance, and clear action items. "
    "Be specific — reference issue keys (e.g. PROJ-123) when relevant. "
    "Avoid vague generalities. Output clean, well-structured English prose or lists."
)


def build_analysis_prompt(
    analysis: AnalysisResult,
    issues: List[JiraIssue],
    config: MeetingConfig,
) -> str:
    """Build the main analysis prompt sent to Claude."""

    lines = [
        f"## Meeting: {config.title}",
        f"Date: {config.date or 'Today'}",
        f"Author: {config.author}",
        "",
        "---",
        "## Jira Data Summary",
        f"Total issues retrieved: {analysis.total_issues}",
        "",
        "### Status Distribution",
    ]
    for status, count in sorted(analysis.by_status.items(), key=lambda x: -x[1]):
        lines.append(f"- {status}: {count}")

    lines += ["", "### Priority Distribution"]
    for prio, count in sorted(analysis.by_priority.items(), key=lambda x: -x[1]):
        lines.append(f"- {prio}: {count}")

    if analysis.sprint_summary:
        s = analysis.sprint_summary
        lines += [
            "",
            "### Sprint Summary",
            f"Sprint: {s.name}",
            f"Issues: {s.done_issues} done / {s.in_progress_issues} in progress / "
            f"{s.todo_issues} to-do / {s.blocked_issues} blocked  ({s.completion_pct}% complete)",
        ]
        if s.total_story_points > 0:
            lines.append(
                f"Story Points: {s.done_story_points:.0f} / {s.total_story_points:.0f} "
                f"({s.sp_completion_pct}% SP done)"
            )

    lines += ["", "### Top Risks & Blockers"]
    if analysis.risks:
        for r in analysis.risks[:15]:
            lines.append(
                f"- [{r.severity.upper()}] {r.issue_key} ({r.risk_type}): "
                f"{r.summary} — {r.detail}"
                + (f" Assignee: {r.assignee}" if r.assignee else " (unassigned)")
            )
    else:
        lines.append("- No significant risks detected.")

    lines += ["", "### Workload by Assignee (top 10)"]
    for w in analysis.by_assignee[:10]:
        overdue_note = f", {w.overdue} overdue" if w.overdue else ""
        sp_note = f", {w.story_points:.0f} SP" if w.story_points else ""
        lines.append(f"- {w.assignee}: {w.total} issues{sp_note}{overdue_note}")

    lines += ["", "### Overdue Issues"]
    if analysis.overdue_issues:
        for i in analysis.overdue_issues[:10]:
            lines.append(
                f"- {i.key}: {i.summary} | due {i.due_date} | "
                f"status: {i.status} | assignee: {i.assignee or 'none'}"
            )
    else:
        lines.append("- None.")

    lines += [
        "",
        "### Sample Issues (most recently updated)",
    ]
    for i in analysis.recently_updated[:8]:
        lines.append(
            f"- {i.key} [{i.status}] {i.priority or '-'} | {i.summary[:80]} | "
            f"assignee: {i.assignee or 'none'}"
        )

    lines += [
        "",
        "---",
        "## Task",
        "Using the data above, write an **Executive Summary** for the upcoming meeting.",
        "Cover:",
        "1. Overall progress (on-track / at-risk / off-track) and why.",
        "2. Top 3 concerns that need discussion or decision.",
        "3. Any workload imbalances or single points of failure.",
        "4. Sprint forecast (will the team finish on time?).",
        "",
        "Keep it under 250 words. Use bullet points for items 2-4.",
    ]

    return "\n".join(lines)


def build_action_items_prompt(analysis: AnalysisResult, executive_summary: str) -> str:
    """Build the follow-up prompt to extract concrete action items."""
    risk_lines = []
    for r in analysis.risks[:12]:
        risk_lines.append(
            f"- {r.issue_key} [{r.severity}] {r.risk_type}: {r.detail} "
            f"(assignee: {r.assignee or 'TBD'})"
        )

    return (
        "Based on the executive summary and risk list below, produce a prioritised "
        "**Action Items** table for the meeting.\n\n"
        "Format each action as a Markdown table row:\n"
        "| # | Action | Owner | Due | Priority |\n\n"
        f"Risks:\n" + "\n".join(risk_lines or ["(none)"]) + "\n\n"
        f"Executive Summary:\n{executive_summary}\n\n"
        "Rules:\n"
        "- List at most 10 action items.\n"
        "- Owner = the Jira assignee if known, else 'TBD'.\n"
        "- Due = 'ASAP' for critical items, otherwise 'This Sprint' or 'Next Sprint'.\n"
        "- Output ONLY the Markdown table, no extra prose."
    )
