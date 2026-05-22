#!/usr/bin/env python3
"""CLI entry point: Jira search → analysis → meeting document.

Examples
--------
# Active sprint, all issues
python scripts/jira_to_meeting.py --sprint-state active --projects QUAL

# Specific sprint, specific assignee
python scripts/jira_to_meeting.py --sprint "Sprint 24" --projects QUAL ENG \\
    --assignee "Kunwoo Lee" --format markdown

# Last 14 days, high priority only, with AI summary
python scripts/jira_to_meeting.py --projects QUAL --days 14 \\
    --priority High Critical --title "Weekly Quality Review"

# Raw JQL override
python scripts/jira_to_meeting.py --jql "project = QUAL AND status != Done ORDER BY priority" \\
    --title "Open Issues Review"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure src/ is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jira_briefing import (
    JiraClient,
    JiraSummarizer,
    MeetingDocGenerator,
    MeetingConfig,
    SearchConfig,
    analyze,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Search Jira, analyse results, and generate a meeting document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Search options ----
    search = p.add_argument_group("Search / Filter")
    search.add_argument("--projects", nargs="+", metavar="KEY",
                        help="Jira project keys, e.g. QUAL ENG")
    search.add_argument("--jql", metavar="JQL",
                        help="Raw JQL override (ignores other search flags)")
    search.add_argument("--sprint", metavar="NAME",
                        help='Sprint name, e.g. "Sprint 24"')
    search.add_argument("--sprint-state", choices=["active", "closed", "future"],
                        help="Sprint state filter (used when --sprint is omitted)")
    search.add_argument("--assignee", nargs="+", metavar="NAME",
                        help="Filter by assignee display name(s)")
    search.add_argument("--status", nargs="+", metavar="STATUS",
                        help='Filter by status, e.g. "In Progress" Blocked')
    search.add_argument("--priority", nargs="+", metavar="PRIO",
                        help="Filter by priority, e.g. High Critical")
    search.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="Filter by labels")
    search.add_argument("--days", type=int, metavar="N",
                        help="Issues updated in the last N days")
    search.add_argument("--keywords", metavar="TEXT",
                        help="Full-text keyword search (JQL text ~)")
    search.add_argument("--max-results", type=int, default=200, metavar="N",
                        help="Maximum issues to retrieve (default: 200)")
    search.add_argument("--include-comments", action="store_true",
                        help="Fetch last 5 comments per issue")

    # ---- Output options ----
    out = p.add_argument_group("Output")
    out.add_argument("--title", default="Sprint Status Review",
                     help="Meeting document title")
    out.add_argument("--author", default="Kunwoo Lee",
                     help="Author shown in the document header")
    out.add_argument("--format", dest="output_format",
                     choices=["markdown", "html", "pdf"], default="markdown",
                     help="Output format (default: markdown)")
    out.add_argument("--output", metavar="PATH",
                     help="Output file path (auto-generated if omitted)")
    out.add_argument("--no-ai", action="store_true",
                     help="Skip Claude AI summarisation (faster, no API cost)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")

    # ---- Build SearchConfig ----
    config = SearchConfig(
        projects=args.projects or [],
        jql=args.jql,
        assignees=args.assignee or [],
        statuses=args.status or [],
        priorities=args.priority or [],
        labels=args.labels or [],
        sprint=args.sprint,
        sprint_state=args.sprint_state,
        updated_days=args.days,
        keywords=args.keywords,
        max_results=args.max_results,
        include_comments=args.include_comments,
    )

    if not config.projects and not config.jql:
        parser.error("Provide at least --projects or --jql.")

    # ---- Search ----
    print(f"[1/4] Connecting to Jira and searching issues…")
    client = JiraClient()
    jql = client.build_jql(config)
    print(f"      JQL: {jql}")
    issues = client.search(config)
    print(f"      Retrieved {len(issues)} issues.")

    if not issues:
        print("No issues found. Check your search parameters.")
        sys.exit(0)

    # ---- Analyse ----
    print("[2/4] Analysing data…")
    result = analyze(issues, config=config)
    print(
        f"      Risks: {len(result.risks)}  |  "
        f"Overdue: {len(result.overdue_issues)}  |  "
        f"Blocked: {result.by_status.get('Blocked', 0)}"
    )

    # ---- AI Summary ----
    ai_output: dict = {"executive_summary": "", "action_items": ""}
    if not args.no_ai:
        print("[3/4] Generating AI summary with Claude…")
        meeting_cfg_tmp = MeetingConfig(
            title=args.title,
            author=args.author,
            date=today,
        )
        try:
            summarizer = JiraSummarizer()
            ai_output = summarizer.summarize(result, issues, meeting_cfg_tmp)
            print("      Done.")
        except Exception as exc:
            print(f"      Warning: AI summarisation failed ({exc}). Continuing without it.")
    else:
        print("[3/4] AI summary skipped (--no-ai).")

    # ---- Render document ----
    print("[4/4] Rendering meeting document…")
    meeting_config = MeetingConfig(
        title=args.title,
        author=args.author,
        date=today,
        output_format=args.output_format,
        output_path=args.output,
    )
    generator = MeetingDocGenerator(result, issues, ai_output, meeting_config)
    out_path = generator.save()
    print(f"\n✅  Meeting document saved to: {out_path}")


if __name__ == "__main__":
    main()
