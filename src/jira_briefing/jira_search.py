"""Jira REST API client with a flexible JQL query builder.

Authentication reads from environment variables:
  JIRA_URL        e.g. https://your-domain.atlassian.net
  JIRA_EMAIL      Atlassian account email
  JIRA_API_TOKEN  API token (not your password)

The ``JiraClient.search()`` method handles pagination automatically and
returns a list of normalised ``JiraIssue`` objects.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Optional

from dotenv import load_dotenv
from jira import JIRA
from jira.exceptions import JIRAError

from . import failure_log
from .schemas import JiraIssue, SearchConfig

# Custom fields where Jira typically stores sprint and story-point data.
# Cloud instances most commonly use these; add others if your instance differs.
_SPRINT_FIELDS = ("customfield_10020", "customfield_10021", "customfield_10014")
_SP_FIELDS = ("customfield_10016", "customfield_10028", "story_points")

_DEFAULT_FIELDS = [
    "summary",
    "description",
    "status",
    "issuetype",
    "priority",
    "assignee",
    "reporter",
    "created",
    "updated",
    "duedate",
    "labels",
    "components",
    "customfield_10016",  # story points (cloud)
    "customfield_10028",  # story points (alt)
    "customfield_10020",  # sprint (cloud)
]


class JiraClient:
    """Wrapper around the python-jira library."""

    def __init__(
        self,
        url: Optional[str] = None,
        email: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        load_dotenv()
        self.url = (url or os.environ.get("JIRA_URL", "")).rstrip("/")
        _email = email or os.environ.get("JIRA_EMAIL", "")
        _token = token or os.environ.get("JIRA_API_TOKEN", "")

        if not all([self.url, _email, _token]):
            raise EnvironmentError(
                "Set JIRA_URL, JIRA_EMAIL, and JIRA_API_TOKEN in your .env file."
            )

        self._jira = JIRA(server=self.url, basic_auth=(_email, _token))

    # ------------------------------------------------------------------
    # JQL builder
    # ------------------------------------------------------------------

    def build_jql(self, config: SearchConfig) -> str:
        """Translate a SearchConfig into a JQL string.

        If ``config.jql`` is set it is returned unchanged (raw override).
        """
        if config.jql:
            return config.jql

        parts: list[str] = []

        if config.projects:
            proj_list = ", ".join(f'"{p}"' for p in config.projects)
            parts.append(f"project in ({proj_list})")

        if config.assignees:
            a_list = ", ".join(f'"{a}"' for a in config.assignees)
            parts.append(f"assignee in ({a_list})")

        if config.statuses:
            s_list = ", ".join(f'"{s}"' for s in config.statuses)
            parts.append(f"status in ({s_list})")

        if config.priorities:
            p_list = ", ".join(f'"{p}"' for p in config.priorities)
            parts.append(f"priority in ({p_list})")

        if config.issue_types:
            t_list = ", ".join(f'"{t}"' for t in config.issue_types)
            parts.append(f"issuetype in ({t_list})")

        if config.labels:
            for label in config.labels:
                parts.append(f'labels = "{label}"')

        if config.sprint:
            parts.append(f'sprint = "{config.sprint}"')
        elif config.sprint_state:
            parts.append(f"sprint in {config.sprint_state}Sprints()")

        if config.created_days:
            parts.append(f"created >= -{config.created_days}d")

        if config.updated_days:
            parts.append(f"updated >= -{config.updated_days}d")

        if config.due_within_days is not None:
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=config.due_within_days)).strftime("%Y-%m-%d")
            parts.append(f"due >= '{today}' AND due <= '{future}'")

        if config.keywords:
            # Escape double quotes inside keyword string
            safe_kw = config.keywords.replace('"', '\\"')
            parts.append(f'text ~ "{safe_kw}"')

        if not parts:
            return "ORDER BY updated DESC"

        jql = " AND ".join(parts)
        if config.order_by:
            jql += f" ORDER BY {config.order_by}"
        return jql

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, config: SearchConfig) -> List[JiraIssue]:
        """Execute a Jira search and return normalised issues.

        Handles pagination transparently up to ``config.max_results``.
        """
        jql = self.build_jql(config)
        fields = list(_DEFAULT_FIELDS)
        if config.include_comments:
            fields.append("comment")

        issues: list[JiraIssue] = []
        start_at = 0
        batch_size = min(100, config.max_results)

        try:
            while len(issues) < config.max_results:
                batch = self._jira.search_issues(
                    jql,
                    startAt=start_at,
                    maxResults=batch_size,
                    fields=fields,
                )
                if not batch:
                    break
                for raw in batch:
                    issues.append(self._parse_issue(raw, config.include_comments))
                start_at += len(batch)
                if start_at >= batch.total:
                    break
        except JIRAError as exc:
            failure_log.record(
                step="jira_search.JiraClient.search",
                context=f"jql={jql[:200]}",
                exc=exc,
                lesson="Check that the JQL is valid and that the API token has read access.",
            )
            raise

        return issues[: config.max_results]

    def get_projects(self) -> list[dict]:
        """Return a list of {key, name} dicts for all visible projects."""
        return [{"key": p.key, "name": p.name} for p in self._jira.projects()]

    def get_active_sprints(self, board_id: int) -> list[dict]:
        """Return active sprint info for a given board ID."""
        sprints = self._jira.sprints(board_id, state="active")
        return [{"id": s.id, "name": s.name, "state": s.state} for s in sprints]

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_issue(self, raw, include_comments: bool) -> JiraIssue:
        f = raw.fields

        # Sprint: try each candidate custom field
        sprint_name: Optional[str] = None
        for attr in _SPRINT_FIELDS:
            val = getattr(f, attr, None)
            if val:
                if isinstance(val, list) and val:
                    last = val[-1]
                    sprint_name = getattr(last, "name", None) or str(last)
                elif hasattr(val, "name"):
                    sprint_name = val.name
                if sprint_name:
                    break

        # Story points: try each candidate custom field
        sp: Optional[float] = None
        for attr in _SP_FIELDS:
            sp_raw = getattr(f, attr, None)
            if sp_raw is not None:
                try:
                    sp = float(sp_raw)
                except (TypeError, ValueError):
                    pass
                if sp is not None:
                    break

        # Comments (last 5)
        comments: list[str] = []
        if include_comments and hasattr(f, "comment") and f.comment:
            for c in f.comment.comments[-5:]:
                author = getattr(getattr(c, "author", None), "displayName", "Unknown")
                body = (c.body or "")[:300]
                comments.append(f"[{author}] {body}")

        def _dt(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            try:
                return datetime.fromisoformat(val[:19])
            except ValueError:
                return None

        return JiraIssue(
            key=raw.key,
            summary=f.summary or "",
            description=str(f.description or "")[:600] if f.description else None,
            status=f.status.name if f.status else "Unknown",
            issue_type=f.issuetype.name if hasattr(f, "issuetype") and f.issuetype else "Unknown",
            priority=f.priority.name if f.priority else None,
            assignee=f.assignee.displayName if f.assignee else None,
            reporter=f.reporter.displayName if f.reporter else None,
            created=_dt(getattr(f, "created", None)),
            updated=_dt(getattr(f, "updated", None)),
            due_date=_dt(getattr(f, "duedate", None)),
            labels=list(f.labels) if f.labels else [],
            components=[c.name for c in f.components] if f.components else [],
            sprint=sprint_name,
            story_points=sp,
            comments=comments,
            url=f"{self.url}/browse/{raw.key}",
        )
