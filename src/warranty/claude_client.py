"""Anthropic Claude API wrapper with prompt caching, retries, and web search.

This module centralizes all LLM calls so that:
  - The expensive system prompt (taxonomy, rules) is cached across
    per-row loops via Anthropic's prompt-caching feature.
  - Transient API errors (429, 529, 503) are retried with exponential
    backoff using tenacity.
  - The web_search server tool is wired up for Steps 6-7.
  - Structured JSON output is validated against Pydantic schemas;
    on parse failure a single self-correcting retry is attempted
    before logging to docs/FAILURE_LOG.md.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Type

from anthropic import Anthropic
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from . import failure_log
from .io_utils import load_settings

# ---------------------------------------------------------------------------
# Retry predicate: only retry on transient server / rate-limit errors.
# ---------------------------------------------------------------------------


def _is_transient(exc: BaseException) -> bool:
    """Return True for HTTP 429, 503, 529 from the Anthropic SDK."""
    from anthropic import APIStatusError
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 503, 529)
    return False


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------


class ClaudeClient:
    """Thin wrapper around the Anthropic Messages API.

    Attributes:
        default_model: Model for per-row NLP calls (Sonnet).
        heavy_model: Model for heavy-reasoning calls (Opus).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        settings: Optional[dict] = None,
    ) -> None:
        settings = settings or load_settings()
        ant_cfg = settings.get("anthropic", {})

        self.default_model: str = ant_cfg.get("default_model", "claude-sonnet-4-6")
        self.heavy_model: str = ant_cfg.get("heavy_model", "claude-opus-4-6")
        self._max_tokens: int = ant_cfg.get("max_output_tokens", 2048)
        self._max_retries: int = ant_cfg.get("retries", 5)
        self._web_search_max: int = ant_cfg.get("web_search_max_uses", 3)

        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = Anthropic(api_key=key)

    # ---- core chat ----

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: Optional[list[dict[str, Any]]] = None,
        model_override: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a Messages API request and return the assistant text.

        Args:
            messages: Conversation messages (role/content dicts).
            system: System-prompt blocks (with cache_control where applicable).
            model_override: Use a different model for this call.
            tools: Tool definitions (e.g., web_search server tool).
            max_tokens: Override max output tokens for this call.

        Returns:
            The assistant's text response as a plain string.
        """
        model = model_override or self.default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self._call_with_retry(**kwargs)
        # Extract text from content blocks.
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        schema: Type[BaseModel],
        system: Optional[list[dict[str, Any]]] = None,
        model_override: Optional[str] = None,
    ) -> BaseModel:
        """Send a request expecting structured JSON that validates against a Pydantic schema.

        On parse failure, one self-correcting retry is attempted where the
        model is shown its own malformed output and asked to fix it. If the
        retry also fails, the error is logged and re-raised.
        """
        # Append JSON instruction to the last user message.
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        json_instruction = (
            "\n\nRespond ONLY with a JSON object matching this schema "
            "(no markdown fences, no extra text):\n" + schema_json
        )
        enriched = _append_to_last_user(messages, json_instruction)

        raw = self.chat(enriched, system=system, model_override=model_override)
        try:
            return _parse_json(raw, schema)
        except Exception as first_err:
            # Self-correcting retry: show the bad output and ask to fix.
            fix_msg = enriched + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"Your previous response failed JSON validation: {first_err}. "
                    "Please output ONLY the corrected JSON object."
                )},
            ]
            raw2 = self.chat(fix_msg, system=system, model_override=model_override)
            try:
                return _parse_json(raw2, schema)
            except Exception as second_err:
                failure_log.record(
                    step="claude_client.chat_json",
                    context=f"schema={schema.__name__}",
                    exc=second_err,
                    lesson=(
                        "LLM returned invalid JSON even after self-correction retry. "
                        "Check the schema complexity or add examples to the prompt."
                    ),
                )
                raise

    def chat_with_web_search(
        self,
        messages: list[dict[str, Any]],
        system: Optional[list[dict[str, Any]]] = None,
        model_override: Optional[str] = None,
        max_uses: Optional[int] = None,
    ) -> str:
        """Chat with the web_search server tool enabled."""
        tools = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_uses or self._web_search_max,
            }
        ]
        return self.chat(
            messages, system=system, model_override=model_override, tools=tools,
        )

    # ---- internal ----

    @retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=16),
        reraise=True,
    )
    def _call_with_retry(self, **kwargs: Any) -> Any:
        return self._client.messages.create(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_to_last_user(
    messages: list[dict[str, Any]], suffix: str,
) -> list[dict[str, Any]]:
    """Return a copy of messages with suffix appended to the last user msg."""
    msgs = [m.copy() for m in messages]
    for m in reversed(msgs):
        if m.get("role") == "user":
            m["content"] = str(m.get("content", "")) + suffix
            break
    return msgs


def _parse_json(raw: str, schema: Type[BaseModel]) -> BaseModel:
    """Strip markdown fences if present, then validate against schema."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return schema.model_validate_json(text)
