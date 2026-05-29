"""Quantization driver.

The actual heavy quantization step is delegated to a :class:`QuantizeBackend`
implementation — production uses :class:`SubprocessSwiftQuantizer` to
spawn ``swift export --quant_method <awq|gptq>``; tests inject a noop
backend that just produces a placeholder file.

The driver writes the quantized weights into a per-model output dir
under the registry, and the route layer wires the result into the
model registry as a new version of the same family with a synthetic
description (``"quantized awq:4bit from <parent_model_id>"``).
"""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

QuantMethod = Literal["awq", "gptq"]


@dataclass
class QuantizationResult:
    method: QuantMethod
    bits: int
    output_dir: str
    log_tail: str = ""


class QuantizeBackend(ABC):
    @abstractmethod
    def quantize(
        self,
        *,
        source_adapter_path: str,
        out_dir: Path,
        method: QuantMethod,
        bits: int,
    ) -> QuantizationResult: ...


class SubprocessSwiftQuantizer(QuantizeBackend):
    """Default backend: shells out to ``swift export``.

    ms-swift's ``swift export`` accepts ``--quant_method`` and
    ``--quant_bits``; the output goes to ``--output_dir``. We capture
    stdout/stderr so the route can surface failure detail.
    """

    # Default 4-hour cap. A wedged ``swift export`` on a worker thread
    # is uncancellable via the asyncio loop (``asyncio.to_thread`` does
    # not propagate cancellation into the subprocess), so the timeout
    # at the ``subprocess.run`` layer is the only line of defense.
    DEFAULT_TIMEOUT_SEC = 4 * 60 * 60

    def __init__(self, timeout_sec: int | None = None) -> None:
        self.timeout_sec = (
            timeout_sec if timeout_sec is not None else self.DEFAULT_TIMEOUT_SEC
        )

    def quantize(
        self,
        *,
        source_adapter_path: str,
        out_dir: Path,
        method: QuantMethod,
        bits: int,
    ) -> QuantizationResult:
        # Defensive: refuse adapter paths that look like flags so a future
        # caller can't sneak ``--evil`` past swift's argparse.
        if source_adapter_path.startswith("-"):
            raise ValueError(
                f"adapter path looks like a flag (refusing): {source_adapter_path!r}"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            "swift",
            "export",
            "--model",
            source_adapter_path,
            "--quant_method",
            method,
            "--quant_bits",
            str(bits),
            "--output_dir",
            str(out_dir),
        ]
        logger.info("quantize: spawning %s", argv)
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"swift export timed out after {self.timeout_sec}s"
            ) from e
        log_tail = (proc.stdout + "\n" + proc.stderr)[-4096:]
        if proc.returncode != 0:
            raise RuntimeError(
                f"swift export failed rc={proc.returncode}: {log_tail}"
            )
        return QuantizationResult(
            method=method, bits=bits, output_dir=str(out_dir), log_tail=log_tail
        )


def quantize_model(
    *,
    source_adapter_path: str,
    out_dir: Path,
    method: QuantMethod,
    bits: int,
    backend: QuantizeBackend | None = None,
) -> QuantizationResult:
    """Top-level convenience wrapper."""
    backend = backend or SubprocessSwiftQuantizer()
    return backend.quantize(
        source_adapter_path=source_adapter_path,
        out_dir=out_dir,
        method=method,
        bits=bits,
    )


# For tests: easy noop backend.
class MockQuantizeBackend(QuantizeBackend):
    def __init__(
        self,
        on_call: Callable[[Path], None] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.on_call = on_call

    def quantize(
        self,
        *,
        source_adapter_path: str,
        out_dir: Path,
        method: QuantMethod,
        bits: int,
    ) -> QuantizationResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Drop a marker file so the route's existence check passes.
        (out_dir / "QUANTIZED_MOCK").write_text(
            f"{method}:{bits}", encoding="utf-8"
        )
        self.calls.append(
            {"src": source_adapter_path, "method": method, "bits": bits}
        )
        if self.on_call is not None:
            self.on_call(out_dir)
        return QuantizationResult(
            method=method, bits=bits, output_dir=str(out_dir), log_tail="ok"
        )
