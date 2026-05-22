"""Statistical aggregation, workload analysis, and risk detection.

Given a list of JiraIssue objects this module produces an AnalysisResult
that summarises the current state of the work and flags blockers, overdue
items, unassigned high-priority issues, and stale in-progress tasks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import List, Optional

from .schemas import (
    AnalysisResult,
    JiraIssue,
    RiskItem,
    SearchConfig,
    SprintSummary,
    WorkloadEntry,
)

# Status groupings (lower-cased for comparison)
_DONE = {"done", "closed", "resolved", "won't do", "wont do", "duplicate", "complete", "completed"}
_IN_PROGRESS = {
    "in progress", "in review", "in development", "testing",
    "under review", "code review", "qa", "in qa",
}
_BLOCKED = {"blocked", "impediment", "on hold"}


def analyze(
    issues: List[JiraIssue],
    config: Optional[SearchConfig] = None,
) -> AnalysisResult:
    """Compute statistics and risk items from a flat list of Jira issues.

    Args:
        issues: Normalised JiraIssue objects from JiraClient.search().
        config: The SearchConfig used to retrieve issues (attached to result).

    Returns:
        AnalysisResult with aggregated stats, workload, and risk lists.
    """
    now = datetime.now(tz=timezone.utc)

    by_status = Counter(i.status for i in issues)
    by_priority = Counter(i.priority or "None" for i in issues)
    by_type = Counter(i.issue_type for i in issues)

    # ---- workload by assignee ----------------------------------------
    assignee_map: dict[str, list[JiraIssue]] = defaultdict(list)
    for issue in issues:
        assignee_map[issue.assignee or "Unassigned"].append(issue)

    workload: list[WorkloadEntry] = []
    for assignee, a_issues in assignee_map.items():
        overdue_count = sum(
            1 for i in a_issues
            if i.due_date and _make_utc(i.due_date) < now
            and i.status.lower() not in _DONE
        )
        workload.append(
            WorkloadEntry(
                assignee=assignee,
                total=len(a_issues),
                by_status=dict(Counter(i.status for i in a_issues)),
                by_priority=dict(Counter(i.priority or "None" for i in a_issues)),
                overdue=overdue_count,
                story_points=sum(i.story_points or 0 for i in a_issues),
            )
        )
    workload.sort(key=lambda w: w.total, reverse=True)

    # ---- sprint summary -----------------------------------------------
    sprint_summary: Optional[SprintSummary] = _build_sprint_summary(issues)

    # ---- risk detection ----------------------------------------------
    risks: list[RiskItem] = []
    overdue_issues: list[JiraIssue] = []

    for issue in issues:
        status_lower = issue.status.lower()

        # Blocked
        if status_lower in _BLOCKED or "blocker" in [l.lower() for l in issue.labels]:
            risks.append(
                RiskItem(
                    issue_key=issue.key,
                    summary=issue.summary,
                    risk_type="blocked",
                    severity="critical",
                    detail=f"Status is '{issue.status}' — active blocker.",
                    assignee=issue.assignee,
                    due_date=_fmt_date(issue.due_date),
                    url=issue.url,
                )
            )

        # Overdue
        if issue.due_date and status_lower not in _DONE:
            due_utc = _make_utc(issue.due_date)
            if due_utc < now:
                days_late = (now - due_utc).days
                severity = "critical" if days_late > 7 else "high" if days_late > 3 else "medium"
                risks.append(
                    RiskItem(
                        issue_key=issue.key,
                        summary=issue.summary,
                        risk_type="overdue",
                        severity=severity,
                        detail=f"Overdue by {days_late} day(s) (due {_fmt_date(issue.due_date)}).",
                        assignee=issue.assignee,
                        due_date=_fmt_date(issue.due_date),
                        url=issue.url,
                    )
                )
                overdue_issues.append(issue)

        # Unassigned high-priority
        if not issue.assignee and (issue.priority or "").lower() in (
            "blocker", "critical", "highest", "high"
        ):
            risks.append(
                RiskItem(
                    issue_key=issue.key,
                    summary=issue.summary,
                    risk_type="unassigned",
                    severity="high",
                    detail=f"Priority '{issue.priority}' but has no assignee.",
                    assignee=None,
                    url=issue.url,
                )
            )

        # Stale in-progress (not updated in >5 days)
        if status_lower in _IN_PROGRESS and issue.updated:
            days_stale = (now - _make_utc(issue.updated)).days
            if days_stale > 5:
                risks.append(
                    RiskItem(
                        issue_key=issue.key,
                        summary=issue.summary,
                        risk_type="stale",
                        severity="medium",
                        detail=f"In-progress but not updated for {days_stale} day(s).",
                        assignee=issue.assignee,
                        url=issue.url,
                    )
                )

    # Deduplicate risks by (key, type) and sort by severity
    _sev = {"critical": 0, "high": 1, "medium": 2}
    seen: set[tuple] = set()
    deduped: list[RiskItem] = []
    for r in risks:
        key = (r.issue_key, r.risk_type)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    deduped.sort(key=lambda r: _sev.get(r.severity, 3))

    # Recently updated (top 8 by updated timestamp)
    recently_updated = sorted(
        [i for i in issues if i.updated],
        key=lambda i: i.updated,  # type: ignore[arg-type]
        reverse=True,
    )[:8]

    return AnalysisResult(
        total_issues=len(issues),
        by_status=dict(by_status),
        by_priority=dict(by_priority),
        by_type=dict(by_type),
        by_assignee=workload,
        sprint_summary=sprint_summary,
        risks=deduped,
        overdue_issues=overdue_issues,
        recently_updated=recently_updated,
        search_config=config,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _fmt_date(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime("%Y-%m-%d") if dt else None


def _build_sprint_summary(issues: List[JiraIssue]) -> Optional[SprintSummary]:
    """Build a sprint summary from issues that carry a sprint name."""
    sprinted = [i for i in issues if i.sprint]
    if not sprinted:
        return None

    # Use the most common sprint name (handles mixed-sprint searches)
    sprint_name = Counter(i.sprint for i in sprinted).most_common(1)[0][0]
    subset = [i for i in sprinted if i.sprint == sprint_name]

    done = sum(1 for i in subset if i.status.lower() in _DONE)
    blocked = sum(1 for i in subset if i.status.lower() in _BLOCKED)
    in_prog = sum(1 for i in subset if i.status.lower() in _IN_PROGRESS)
    todo = len(subset) - done - blocked - in_prog

    total_sp = sum(i.story_points or 0 for i in subset)
    done_sp = sum(i.story_points or 0 for i in subset if i.status.lower() in _DONE)

    return SprintSummary(
        name=sprint_name,
        total_issues=len(subset),
        done_issues=done,
        in_progress_issues=in_prog,
        todo_issues=max(todo, 0),
        blocked_issues=blocked,
        total_story_points=total_sp,
        done_story_points=done_sp,
    )
