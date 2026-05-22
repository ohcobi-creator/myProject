"""Claude-powered summarizer for Jira analysis results.

Uses prompt caching on the system prompt to reduce token cost across the
two-pass generation (executive summary → action items). Retries transient
API errors with exponential back-off, consistent with the rest of this repo.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from anthropic import Anthropic, APIStatusError
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from . import failure_log
from .prompts import SYSTEM_PROMPT, build_action_items_prompt, build_analysis_prompt
from .schemas import AnalysisResult, JiraIssue, MeetingConfig


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, APIStatusError) and exc.status_code in (429, 503, 529)


class JiraSummarizer:
    """Two-pass Claude summarizer: executive summary then action items.

    Args:
        model: Anthropic model ID. Defaults to claude-sonnet-4-6.
        api_key: Explicit key; falls back to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ) -> None:
        load_dotenv()
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError("Set ANTHROPIC_API_KEY in your .env file.")
        self._client = Anthropic(api_key=key)
        self.model = model

    def summarize(
        self,
        analysis: AnalysisResult,
        issues: List[JiraIssue],
        config: MeetingConfig,
    ) -> Dict[str, str]:
        """Generate executive summary and action items.

        Returns:
            {"executive_summary": str, "action_items": str}
        """
        # System prompt is cached — only charged once per cache TTL (5 min).
        system = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # Pass 1: executive summary
        analysis_prompt = build_analysis_prompt(analysis, issues, config)
        try:
            exec_summary = self._chat(
                system=system,
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=800,
            )
        except Exception as exc:
            failure_log.record(
                step="jira_summarizer.pass1_executive_summary",
                context=f"total_issues={analysis.total_issues}",
                exc=exc,
                lesson="Check Claude API key validity and model name.",
            )
            raise

        # Pass 2: action items (continues the conversation for coherence)
        action_prompt = build_action_items_prompt(analysis, exec_summary)
        try:
            action_items = self._chat(
                system=system,
                messages=[
                    {"role": "user", "content": analysis_prompt},
                    {"role": "assistant", "content": exec_summary},
                    {"role": "user", "content": action_prompt},
                ],
                max_tokens=600,
            )
        except Exception as exc:
            failure_log.record(
                step="jira_summarizer.pass2_action_items",
                context=f"total_issues={analysis.total_issues}",
                exc=exc,
                lesson="Pass 2 failed after Pass 1 succeeded; check token limits.",
            )
            raise

        return {
            "executive_summary": exec_summary.strip(),
            "action_items": action_items.strip(),
        }

    @retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=16),
        reraise=True,
    )
    def _chat(self, system, messages, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(parts)
