"""Pydantic data models for the Jira briefing pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JiraIssue(BaseModel):
    key: str
    summary: str
    description: Optional[str] = None
    status: str
    issue_type: str
    priority: Optional[str] = None
    assignee: Optional[str] = None
    reporter: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    due_date: Optional[datetime] = None
    labels: List[str] = Field(default_factory=list)
    components: List[str] = Field(default_factory=list)
    sprint: Optional[str] = None
    story_points: Optional[float] = None
    comments: List[str] = Field(default_factory=list)
    url: Optional[str] = None


class SprintSummary(BaseModel):
    name: str
    state: str = "unknown"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_issues: int = 0
    done_issues: int = 0
    in_progress_issues: int = 0
    todo_issues: int = 0
    blocked_issues: int = 0
    total_story_points: float = 0.0
    done_story_points: float = 0.0

    @property
    def completion_pct(self) -> float:
        if self.total_issues == 0:
            return 0.0
        return round(self.done_issues / self.total_issues * 100, 1)

    @property
    def sp_completion_pct(self) -> float:
        if self.total_story_points == 0:
            return 0.0
        return round(self.done_story_points / self.total_story_points * 100, 1)


class SearchConfig(BaseModel):
    """Parameters that drive the JQL query builder.

    Set ``jql`` for a raw override; otherwise the builder assembles a query
    from the structured fields below.
    """

    projects: List[str] = Field(default_factory=list)
    jql: Optional[str] = None
    assignees: List[str] = Field(default_factory=list)
    statuses: List[str] = Field(default_factory=list)
    priorities: List[str] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    issue_types: List[str] = Field(default_factory=list)
    sprint: Optional[str] = None
    sprint_state: Optional[str] = None  # "active", "closed", "future"
    created_days: Optional[int] = None   # issues created in last N days
    updated_days: Optional[int] = None   # issues updated in last N days
    due_within_days: Optional[int] = None  # due within next N days
    keywords: Optional[str] = None
    max_results: int = 200
    include_comments: bool = False
    order_by: str = "updated DESC"


class WorkloadEntry(BaseModel):
    assignee: str
    total: int
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_priority: Dict[str, int] = Field(default_factory=dict)
    overdue: int = 0
    story_points: float = 0.0


class RiskItem(BaseModel):
    issue_key: str
    summary: str
    risk_type: str   # "blocked" | "overdue" | "unassigned" | "stale"
    severity: str    # "critical" | "high" | "medium"
    detail: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    url: Optional[str] = None


class AnalysisResult(BaseModel):
    total_issues: int
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_priority: Dict[str, int] = Field(default_factory=dict)
    by_type: Dict[str, int] = Field(default_factory=dict)
    by_assignee: List[WorkloadEntry] = Field(default_factory=list)
    sprint_summary: Optional[SprintSummary] = None
    risks: List[RiskItem] = Field(default_factory=list)
    overdue_issues: List[JiraIssue] = Field(default_factory=list)
    recently_updated: List[JiraIssue] = Field(default_factory=list)
    search_config: Optional[SearchConfig] = None
    analysed_at: datetime = Field(default_factory=datetime.now)


class MeetingConfig(BaseModel):
    title: str = "Sprint Status Review"
    date: Optional[str] = None
    author: str = "Kunwoo Lee"
    output_format: str = "markdown"   # "markdown" | "html" | "pdf"
    output_path: Optional[str] = None
    language: str = "en"
    sections: List[str] = Field(
        default_factory=lambda: [
            "executive_summary",
            "sprint_dashboard",
            "issue_table",
            "blockers_risks",
            "workload",
            "overdue",
            "action_items",
            "appendix",
        ]
    )
