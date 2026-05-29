"""Synthetic data generation (Phase 14).

Uses a teacher LLM to expand a small source dataset into a larger one
of (prompt, completion) pairs. Provider is pluggable; default backends
talk to Anthropic and OpenAI via HTTP.
"""

from .runner import (
    AnthropicProvider,
    FatalHTTPError,
    MockProvider,
    OpenAIProvider,
    RetriableHTTPError,
    SynthAborted,
    SynthProvider,
    generate_synthetic,
)

__all__ = [
    "AnthropicProvider",
    "FatalHTTPError",
    "MockProvider",
    "OpenAIProvider",
    "RetriableHTTPError",
    "SynthAborted",
    "SynthProvider",
    "generate_synthetic",
]
