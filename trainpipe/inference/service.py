"""Inference service: model-ref resolution + LRU-cached backends.

A *model ref* is a single string the API accepts in lieu of the user
having to know the underlying experiment id / adapter path. Supported
forms (resolved against the DB):

* ``base:<hf-id-or-path>`` — just the base model, no adapter.
* ``exp:<experiment_id>`` — load the adapter trained by that experiment.
* ``<name>@<alias>``       — registered model alias (e.g.
  ``invoice-extractor@production``).
* ``<name>@<version:int>`` — explicit registered version
  (``invoice-extractor@3``).

``InferenceService`` keeps an LRU cache of opened :class:`InferenceBackend`
instances keyed by the resolved ``(base_model, adapter_path)`` tuple, so
two refs that point at the same artifact share a load.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from ..core import repository
from ..core.db import Database
from ..evals.inference import InferenceBackend
from ..settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRef:
    """Resolved model reference: what to actually load."""

    base_model: str
    adapter_path: str | None  # None = bare base model

    @property
    def cache_key(self) -> tuple[str, str | None]:
        return (self.base_model, self.adapter_path)


class UnknownModelRef(Exception):
    """Raised when a string ref cannot be resolved."""

    def __init__(self, raw: str, reason: str) -> None:
        super().__init__(f"unknown model ref {raw!r}: {reason}")
        self.raw = raw
        self.reason = reason


async def resolve_model_ref(
    raw: str, conn: aiosqlite.Connection
) -> ModelRef:
    """Translate a user-supplied ref string into a concrete (base, adapter).

    Returns :class:`ModelRef`. Raises :class:`UnknownModelRef` for missing
    experiments / registered models / unknown prefixes.
    """
    raw = raw.strip()
    if not raw:
        raise UnknownModelRef(raw, "empty")

    if raw.startswith("base:"):
        return ModelRef(base_model=raw[len("base:") :], adapter_path=None)

    if raw.startswith("exp:"):
        exp_id = raw[len("exp:") :]
        exp = await repository.get_experiment(conn, exp_id)
        if exp is None:
            raise UnknownModelRef(raw, "experiment not found")
        adapter = exp.spec.output_dir or str(
            settings.output_base_dir / exp_id
        )
        return ModelRef(base_model=exp.spec.model, adapter_path=adapter)

    if "@" in raw:
        name, sep, ref = raw.partition("@")
        _ = sep  # always "@" because "in" returned True
        if not name or not ref:
            raise UnknownModelRef(raw, "expected <name>@<alias-or-version>")
        if ref.isdigit():
            model = await repository.get_model_by_name_version(
                conn, name, int(ref)
            )
        else:
            model = await repository.resolve_model_alias(conn, name, ref)
        if model is None:
            raise UnknownModelRef(raw, "registered model not found")
        return ModelRef(
            base_model=model.base_model, adapter_path=model.adapter_path
        )

    raise UnknownModelRef(
        raw,
        "expected prefix 'base:', 'exp:', or '<name>@<alias|version>'",
    )


BackendFactory = "callable[[ModelRef], InferenceBackend]"


class InferenceService:
    """Process-wide cache of opened inference backends.

    The cache is bounded LRU (``max_loaded``); evicting an entry calls
    its :meth:`InferenceBackend.close` to free GPU memory before loading
    the replacement.

    ``backend_factory`` lets tests inject :class:`MockInferenceBackend`;
    the default factory builds a :class:`TransformersInferenceBackend`
    (lazy-import — see ``evals/inference.py``).
    """

    def __init__(
        self,
        db: Database,
        *,
        max_loaded: int = 2,
        backend_factory=None,
    ) -> None:
        self.db = db
        self.max_loaded = max_loaded
        self._cache: OrderedDict[tuple[str, str | None], InferenceBackend] = OrderedDict()
        # Short-held lock: protects cache map + per-key load lock map. NOT
        # held across slow backend.open() — that would queue every cache
        # hit behind a cold load.
        self._map_lock = asyncio.Lock()
        # Per-key load locks: only the cold-load path takes one. Two
        # concurrent gets for the same uncached ref serialize; gets for
        # different refs (or cache hits) don't.
        self._load_locks: dict[tuple[str, str | None], asyncio.Lock] = {}
        self._factory = backend_factory or _default_factory

    async def _get_load_lock(self, key: tuple[str, str | None]) -> asyncio.Lock:
        async with self._map_lock:
            lock = self._load_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._load_locks[key] = lock
            return lock

    async def get(self, ref: ModelRef) -> InferenceBackend:
        """Return an opened backend for ``ref``.

        Cache hits are O(1) and contention-free; cold loads serialize per
        ``ref`` so two concurrent requests for the same uncached model
        share one load instead of double-allocating.
        """
        key = ref.cache_key

        # Fast path: cache hit, just bump LRU order.
        async with self._map_lock:
            backend = self._cache.get(key)
            if backend is not None:
                self._cache.move_to_end(key)
                return backend

        # Cold path: per-key load lock. A second waiter for the same key
        # finds the cache populated when it acquires the lock.
        load_lock = await self._get_load_lock(key)
        async with load_lock:
            async with self._map_lock:
                backend = self._cache.get(key)
                if backend is not None:
                    self._cache.move_to_end(key)
                    return backend
                # Evict before we allocate the new entry so peak GPU usage
                # stays bounded by ``max_loaded`` even mid-load.
                to_close: list[InferenceBackend] = []
                while len(self._cache) >= self.max_loaded and self._cache:
                    _ek, ev = self._cache.popitem(last=False)
                    to_close.append(ev)
            # Close evicted backends OUTSIDE the map lock — their close
            # may also do slow GPU work and we don't want to block the
            # cache map on it.
            for ev in to_close:
                try:
                    await ev.close()
                except Exception:
                    logger.exception("eviction close failed")

            new_backend = self._factory(ref)
            try:
                await new_backend.open()
            except Exception:
                # Don't leak a half-initialized backend if open() failed
                # — call close() best-effort to free whatever it did
                # acquire, then re-raise so the caller sees the load error.
                logger.exception("backend.open() failed for %s", key)
                try:
                    await new_backend.close()
                except Exception:
                    logger.exception("post-failure close also raised")
                raise
            async with self._map_lock:
                self._cache[key] = new_backend
            return new_backend

    async def invalidate(self, ref: ModelRef) -> None:
        """Drop ``ref`` from the cache and close its backend.

        Use after a ``predict()`` failure where the backend may be in a
        broken state (corrupted CUDA context, etc.), so the next request
        triggers a fresh load instead of reusing the bad one.
        """
        key = ref.cache_key
        async with self._map_lock:
            backend = self._cache.pop(key, None)
        if backend is not None:
            try:
                await backend.close()
            except Exception:
                logger.exception("invalidate close failed")

    async def resolve(self, raw: str) -> ModelRef:
        async with self.db.connect() as conn:
            return await resolve_model_ref(raw, conn)

    async def close_all(self) -> None:
        async with self._map_lock:
            backends = list(self._cache.values())
            self._cache.clear()
        for backend in backends:
            try:
                await backend.close()
            except Exception:
                logger.exception("close_all failed for one backend")

    def cache_keys(self) -> list[tuple[str, str | None]]:
        """Snapshot of currently-loaded backend keys (oldest → newest)."""
        return list(self._cache.keys())


def _default_factory(ref: ModelRef) -> InferenceBackend:
    """Build a TransformersInferenceBackend for production use.

    Heavy deps are lazy-imported inside the backend's ``open()`` so this
    factory itself is cheap even on machines without torch installed.
    """
    from ..evals.inference import TransformersInferenceBackend

    adapter = Path(ref.adapter_path) if ref.adapter_path else None
    return TransformersInferenceBackend(
        base_model=ref.base_model,
        adapter_path=adapter,
        gpu_indices=[],  # service-level: caller env controls visibility
    )
