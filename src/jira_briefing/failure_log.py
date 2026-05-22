"""Re-export the shared failure_log from the warranty module.

Import this inside jira_briefing so callers write:
    from . import failure_log
    failure_log.record(...)
"""

from warranty.failure_log import record  # noqa: F401
