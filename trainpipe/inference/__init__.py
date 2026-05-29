"""Inference / playground service (Phase 8).

Loads a base model + optional adapter on demand, caches up to N backends
LRU-style, and serves single + streaming + compare predictions.
"""

from .service import InferenceService, ModelRef, resolve_model_ref

__all__ = ["InferenceService", "ModelRef", "resolve_model_ref"]
