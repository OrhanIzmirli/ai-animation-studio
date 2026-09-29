"""Run one BenchmarkRequest on one backend and log the result.

The runner knows only the VideoEditingBackend contract. It never inspects
backend_parameters or result.parameters; it copies them into the record.

Stages: validation -> prepare -> generation -> verification -> logging.
Each stage is timed separately (prepare = cold start, generation = inference).

Failed runs are benchmark data: if a stage raises, a record with
status="failed" is written (whenever that is possible without overwriting
anything) and BenchmarkRunError is raised, chained to the original exception.
Exceptions are never swallowed. Not recorded, and raised as-is:
  - FileExistsError: the run_id or its output directory already exists
  - BackendNotInstalledError: a configuration problem, not a model result
  - KeyboardInterrupt / SystemExit (only Exception subclasses are caught)
Timeouts are the backend's job (only it can stop its own inference); a
TimeoutError raised by the backend is recorded like any other failure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_studio.benchmarking.environment import collect_environment
from ai_studio.benchmarking.logger import BenchmarkLogger, new_run_id
from ai_studio.benchmarking.request import BenchmarkRequest
from ai_studio.benchmarking.schema import BenchmarkResult, check_file_name_safe
from ai_studio.generation.backends.base import (
    BackendMetadata,
    BackendNotInstalledError,
    VideoEditingBackend,
)
from ai_studio.generation.result import GenerationResult
from ai_studio.paths import project_root, resolve_path, to_project_relative

MAX_ERROR_MESSAGE_CHARS = 4000


class BenchmarkRunError(RuntimeError):
    """A benchmark run failed. `record` is the failed-run record, or None if it
    could not be saved; the original exception is available as __cause__."""

    def __init__(self, stage: str, record: BenchmarkResult | None, cause: BaseException) -> None:
        saved = "recorded" if record is not None else "NOT recorded"
        super().__init__(f"benchmark run failed during {stage} ({saved}): {type(cause).__name__}: {cause}")
        self.stage = stage
        self.record = record


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error_type(exc: BaseException) -> str:
    cls = type(exc)
    return cls.__qualname__ if cls.__module__ == "builtins" else f"{cls.__module__}.{cls.__qualname__}"


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=repr))


class BenchmarkRunner:
    def __init__(
        self,
        logger: BenchmarkLogger,
        outputs_dir: str | Path,
        environment_collector: Callable[[], dict[str, Any]] = collect_environment,
        root: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve() if root is not None else project_root()
        self.logger = logger
        self.outputs_dir = resolve_path(outputs_dir, self.root)
        self.environment_collector = environment_collector

    def run(
        self,
        request: BenchmarkRequest,
        backend: VideoEditingBackend,
        *,
        run_id: str | None = None,
        estimated_cost_usd: float | None = None,
        notes: str = "",
    ) -> BenchmarkResult:
        run_id = run_id or new_run_id()
        check_file_name_safe("run_id", run_id)
        output_dir = self.outputs_dir / request.benchmark_id / run_id
        if self.logger.path_for(request.benchmark_id, run_id).exists() or output_dir.exists():
            raise FileExistsError(f"run {request.benchmark_id}/{run_id} already exists")

        # Backends always receive absolute paths, independent of the cwd.
        resolved = dataclasses.replace(
            request,
            source_video=str(resolve_path(request.source_video, self.root)),
            reference_images=[str(resolve_path(p, self.root)) for p in request.reference_images],
        )
        meta = self._metadata(backend)

        timings: dict[str, float] = {}
        checksums: dict[str, str] = {}
        result: GenerationResult | None = None
        stage = "validation"
        run_start = stage_start = time.perf_counter()

        def enter(next_stage: str) -> None:
            nonlocal stage, stage_start
            now = time.perf_counter()
            timings[stage] = now - stage_start
            stage, stage_start = next_stage, now

        try:
            missing = [p for p in resolved.input_paths if not Path(p).is_file()]
            if missing:
                raise FileNotFoundError(f"input files not found: {missing}")
            backend.validate_inputs(resolved)
            checksums = {p: sha256_file(p) for p in resolved.input_paths}

            enter("prepare")
            backend.prepare()

            enter("generation")
            output_dir.mkdir(parents=True, exist_ok=False)  # never reuse an output dir
            result = backend.generate(resolved, output_dir)

            enter("verification")
            self._verify(result, output_dir, resolved, checksums, self.root)
            enter("done")
        except BackendNotInstalledError:
            raise
        except Exception as exc:
            timings[stage] = time.perf_counter() - stage_start
            timings["total"] = time.perf_counter() - run_start
            record = self._save_failure(
                exc, stage, request, run_id, meta, result, checksums, timings,
                estimated_cost_usd, notes,
            )
            raise BenchmarkRunError(stage, record, exc) from exc

        timings["total"] = time.perf_counter() - run_start
        try:
            record = self._build_record(
                request, run_id, meta, result, checksums, timings, estimated_cost_usd, notes,
            )
            self.logger.log(record)
        except FileExistsError:
            raise
        except Exception as exc:
            failed = self._save_failure(
                exc, "logging", request, run_id, meta, result, checksums, timings,
                estimated_cost_usd, notes,
            )
            raise BenchmarkRunError("logging", failed, exc) from exc
        return record

    # --- helpers ---

    @staticmethod
    def _metadata(backend: VideoEditingBackend) -> BackendMetadata:
        try:
            return backend.metadata()
        except Exception:
            unknown = "unknown"
            return BackendMetadata(
                backend_name=type(backend).__name__, backend_version=unknown,
                model_name=unknown, model_version=unknown, workflow_version=unknown,
            )

    @staticmethod
    def _verify(
        result: Any,
        output_dir: Path,
        request: BenchmarkRequest,
        checksums: dict[str, str],
        root: Path,
    ) -> None:
        if not isinstance(result, GenerationResult):
            raise TypeError(f"backend returned {type(result).__name__}, expected GenerationResult")
        output = resolve_path(result.output_video, root)
        if not output.is_relative_to(output_dir.resolve()) or not output.is_file():
            raise RuntimeError(f"backend output must be a file inside {output_dir}: {output}")
        if {p: sha256_file(p) for p in request.input_paths} != checksums:
            raise RuntimeError("backend modified an input file; original inputs must be preserved")

    def _collect_environment(self) -> dict[str, Any]:
        try:
            return dict(self.environment_collector())
        except Exception as exc:
            return {"collection_error": f"{_error_type(exc)}: {exc}"}

    def _build_record(
        self,
        request: BenchmarkRequest,
        run_id: str,
        meta: BackendMetadata,
        result: GenerationResult | None,
        checksums: dict[str, str],
        timings: dict[str, float],
        estimated_cost_usd: float | None,
        notes: str,
        failure: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        valid_result = result if isinstance(result, GenerationResult) else None
        effective = valid_result.parameters if valid_result else {}
        model_parameters = {**request.backend_parameters, **effective}

        note_lines = [notes]
        if valid_result:
            note_lines += [f"warning: {w}" for w in valid_result.warnings]
            for key, requested in request.backend_parameters.items():
                if key in effective and effective[key] != requested:
                    note_lines.append(
                        f"warning: backend changed parameter {key!r}: requested "
                        f"{requested!r}, used {effective[key]!r}"
                    )
            if valid_result.generation_time_seconds is not None:
                note_lines.append(
                    f"backend-reported generation time: {valid_result.generation_time_seconds:.3f}s"
                )

        environment = self._collect_environment()
        if valid_result:
            environment.update(valid_result.environment)

        return BenchmarkResult(
            benchmark_id=request.benchmark_id,
            run_id=run_id,
            **(failure or {}),
            backend=valid_result.backend_name if valid_result else meta.backend_name,
            backend_version=valid_result.backend_version if valid_result else meta.backend_version,
            model=valid_result.model_name if valid_result else meta.model_name,
            model_version=valid_result.model_version if valid_result else meta.model_version,
            workflow_version=meta.workflow_version,
            seed=request.seed,
            inference_steps=valid_result.inference_steps if valid_result else None,
            guidance=valid_result.guidance if valid_result else None,
            model_parameters=model_parameters,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            source_video=to_project_relative(request.source_video, self.root),
            output_video=(
                to_project_relative(valid_result.output_video, self.root) if valid_result else None
            ),
            reference_assets=[to_project_relative(p, self.root) for p in request.reference_images],
            input_checksums={to_project_relative(p, self.root): h for p, h in checksums.items()},
            resolution=request.output_resolution,
            fps=valid_result.fps if valid_result else None,
            frame_count=valid_result.frame_count if valid_result else None,
            gpu=environment.get("gpu_model"),
            vram_gb=environment.get("vram_gb"),
            validation_time_seconds=timings.get("validation"),
            prepare_time_seconds=timings.get("prepare"),
            generation_time_seconds=timings.get("generation"),
            verification_time_seconds=timings.get("verification"),
            total_time_seconds=timings.get("total"),
            estimated_cost_usd=estimated_cost_usd,
            environment=environment,
            notes="\n".join(line for line in note_lines if line),
        )

    def _save_failure(
        self,
        exc: Exception,
        stage: str,
        request: BenchmarkRequest,
        run_id: str,
        meta: BackendMetadata,
        result: GenerationResult | None,
        checksums: dict[str, str],
        timings: dict[str, float],
        estimated_cost_usd: float | None,
        notes: str,
    ) -> BenchmarkResult | None:
        """Write a status="failed" record. Returns None if that is not possible."""
        failure = {
            "status": "failed",
            "failure_stage": stage,
            "error_type": _error_type(exc),
            "error_message": str(exc)[:MAX_ERROR_MESSAGE_CHARS],
        }
        try:
            record = self._build_record(
                request, run_id, meta, result, checksums, timings,
                estimated_cost_usd, notes, failure,
            )
            # A failure record must be writable even when the original problem
            # was unserializable data, so force everything to plain JSON.
            record = dataclasses.replace(
                record,
                model_parameters=_json_safe(record.model_parameters),
                environment=_json_safe(record.environment),
            )
            self.logger.log(record)  # "x" mode: never overwrites
            return record
        except Exception as save_exc:
            exc.add_note(f"failed-run record could not be saved: {_error_type(save_exc)}: {save_exc}")
            return None
