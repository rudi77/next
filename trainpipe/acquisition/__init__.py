"""Agentic data acquisition (Phase 22).

Builds a training dataset *from-scratch* out of a natural-language brief —
intake → research → acquire/synthesize → curate → register. Distinct from
``synth`` (which expands an existing source dataset): see
docs/spec/agentic-data-acquisition.md.

The MVP ships the phase machinery, the in-process driver/manager, and the
synthesize/curate/register phases with mock + real LLM providers. Real web
research/acquisition lands in a later phase.
"""
