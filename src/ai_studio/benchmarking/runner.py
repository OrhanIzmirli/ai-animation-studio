"""Run one task request on one backend and log the result.

One runner for every capability. It knows only the VideoBackend contract and
the small task-request protocol shared by BenchmarkRequest (editing) and
VideoGenerationRequest (generation):

    capability              which Capability the request needs
    input_paths             files that must exist and must not be modified
    with_resolved_paths(f)  copy with every input path mapped through f
    task_parameters()       task-level fields to record (e.g. shot controls)
    prompt, negative_prompt, reference_images, output_resolution, seed,
    backend_parameters      generic fields copied into the record

Every record also gets task_parameters["run_purpose"]: "benchmark" (default),
"smoke" or "experiment", given by the caller. It is metadata about the run,
not part of the request.

It never inspects backend_parameters or result.parameters; it copies them
into the record.

Stages: validation -> prepare -> generation -> verification -> logging.
Each stage is timed separately (prepare = cold start, generation = inference).

Verification checks that the output is a file inside the run's output
directory, that no input was modified and that start-frame conditioning was
reported honestly. Then it MEASURES the output video with `media_probe`
(default: ffprobe) instead of trusting the backend's fps / frame_count.
Verification fails when:
  - no probe is available but the request pins fps or a duration tolerance
    ("cannot verify"); without such pins it is only a warning
  - backend-reported fps differs from the measured fps by more than 0.01
  - backend-reported frame_count differs from the measured frame count
  - the request's fps differs from the measured fps by more than 0.01
  - duration_seconds and duration_tolerance_seconds are set and the measured
    duration (frame_count / fps) is further from duration_seconds than that
A measured resolution different from output_resolution is only a warning.
The record stores the measurement in output_media, and its fps / frame_count
columns hold the measured values (backend-reported only when nothing was
measured). Runner warnings go into notes as "warning: ...". These checks use
only generic request fields; the runner stays model-agnostic.

Failed runs are benchmark data: if a stage raises, a record with
status="failed" is written (whenever that is possible without overwriting
anything) and BenchmarkRunError is raised, chained to the original exception.
Exceptions are never swallowed. Not recorded, and raised as-is:
  - FileExistsError: the run_id or its output directory already exists
  - UnsupportedCapabilityError: backend lacks the request's capability
  - BackendNotInstalledError: a configuration problem, not a model result
  - KeyboardInterrupt / SystemExit (only Exception subclasses are caught)
After a successful run is recorded, its output video and derived start frame
are marked read-only (ai_studio.protection), so they cannot be deleted or
overwritten by accident.

Timeouts are the backend's job (only it can stop its own inference); a
TimeoutError raised by the backend is recorded like any other failure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_studio.benchmarking.definitions import BenchmarkDefinition, DraftBenchmarkError
from ai_studio.benchmarking.environment import collect_environment
from ai_studio.benchmarking.logger import BenchmarkLogger, new_run_id
from ai_studio.benchmarking.request import BenchmarkRequest
from ai_studio.benchmarking.schema import BenchmarkResult, Resolution, check_file_name_safe
from ai_studio.generation.backends.base import (
    BackendMetadata,
    BackendNotInstalledError,
    VideoBackend,
)
from ai_studio.generation.capabilities import UnsupportedCapabilityError
from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.generation.result import GenerationResult
from ai_studio.media import image_dimensions
from ai_studio.media_probe import ProbeUnavailableError, VideoInfo, ffprobe_video
from ai_studio.paths import project_root, resolve_path, to_project_relative
from ai_studio.protection import protect_run_files

MAX_ERROR_MESSAGE_CHARS = 4000
# Why a run was made; stored as task_parameters["run_purpose"]. Only
# "benchmark" runs are meant to be rated and compared.
RUN_PURPOSES = ("benchmark", "smoke", "experiment")
FPS_TOLERANCE = 0.01  # 23.976 != 24

TaskRequest = BenchmarkRequest | VideoGenerationRequest


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


@dataclass
class _Run:
    """Everything known about one run so far."""

    benchmark_id: str
    run_id: str
    request: TaskRequest  # as given (project-relative paths)
    meta: BackendMetadata
    estimated_cost_usd: float | None
    notes: str
    purpose: str = "benchmark"
    result: GenerationResult | None = None
    checksums: dict[str, str] = dataclasses.field(default_factory=dict)
    timings: dict[str, float] = dataclasses.field(default_factory=dict)
    media: VideoInfo | None = None  # measured output; None if not measured
    warnings: list[str] = dataclasses.field(default_factory=list)  # runner's own


class BenchmarkRunner:
    def __init__(
        self,
        logger: BenchmarkLogger,
        outputs_dir: str | Path,
        environment_collector: Callable[[], dict[str, Any]] = collect_environment,
        root: str | Path | None = None,
        media_probe: Callable[[Path], VideoInfo] = ffprobe_video,
    ) -> None:
        self.root = Path(root).resolve() if root is not None else project_root()
        self.logger = logger
        self.outputs_dir = resolve_path(outputs_dir, self.root)
        self.environment_collector = environment_collector
        self.media_probe = media_probe

    def run_definition(
        self,
        definition: BenchmarkDefinition,
        backend: VideoBackend,
        *,
        seed: int,
        output_resolution: Resolution,
        fps: float | None = None,
        backend_parameters: dict[str, Any] | None = None,
        run_id: str | None = None,
        estimated_cost_usd: float | None = None,
        notes: str = "",
        purpose: str = "benchmark",
    ) -> BenchmarkResult:
        """Run a benchmark definition. Refuses drafts before anything else
        happens: no backend call, no output directory, no record."""
        if definition.status != "ready":
            raise DraftBenchmarkError(
                f"benchmark {definition.benchmark_id!r} is a draft and cannot be run; "
                "set 'status: ready' in its definition (explicit human decision)"
            )
        request = definition.to_request(
            seed=seed, output_resolution=output_resolution, fps=fps,
            backend_parameters=backend_parameters,
        )
        return self.run(
            request, backend, benchmark_id=definition.benchmark_id, run_id=run_id,
            estimated_cost_usd=estimated_cost_usd, notes=notes, purpose=purpose,
        )

    def run(
        self,
        request: TaskRequest,
        backend: VideoBackend,
        *,
        benchmark_id: str | None = None,
        run_id: str | None = None,
        estimated_cost_usd: float | None = None,
        notes: str = "",
        purpose: str = "benchmark",
    ) -> BenchmarkResult:
        """Run a request directly. `benchmark_id` is required for requests that
        do not carry one (VideoGenerationRequest); for BenchmarkRequest it may
        be omitted. Benchmark definitions should go through run_definition(),
        which enforces the draft/ready lifecycle."""
        benchmark_id = self._benchmark_id(request, benchmark_id)
        if purpose not in RUN_PURPOSES:
            raise ValueError(f"purpose must be one of {RUN_PURPOSES}, got {purpose!r}")
        if not backend.supports(request.capability):
            raise UnsupportedCapabilityError(
                f"{type(backend).__name__} supports "
                f"{sorted(c.value for c in backend.capabilities)}, "
                f"not {request.capability.value!r}"
            )
        if getattr(request, "start_frame", None) and not getattr(backend, "supports_start_frame", False):
            raise UnsupportedCapabilityError(
                f"{type(backend).__name__} does not declare start-frame conditioning "
                "(supports_start_frame); a start_frame is never downgraded to a reference image"
            )
        run_id = run_id or new_run_id()
        check_file_name_safe("run_id", run_id)
        output_dir = self.outputs_dir / benchmark_id / run_id
        if self.logger.path_for(benchmark_id, run_id).exists() or output_dir.exists():
            raise FileExistsError(f"run {benchmark_id}/{run_id} already exists")

        # Backends always receive absolute paths, independent of the cwd.
        resolved = request.with_resolved_paths(lambda p: str(resolve_path(p, self.root)))
        run = _Run(benchmark_id, run_id, request, self._metadata(backend), estimated_cost_usd, notes, purpose)

        stage = "validation"
        run_start = stage_start = time.perf_counter()

        def enter(next_stage: str) -> None:
            nonlocal stage, stage_start
            now = time.perf_counter()
            run.timings[stage] = now - stage_start
            stage, stage_start = next_stage, now

        try:
            missing = [p for p in resolved.input_paths if not Path(p).is_file()]
            if missing:
                raise FileNotFoundError(f"input files not found: {missing}")
            backend.validate_inputs(resolved)
            run.checksums = {p: sha256_file(p) for p in resolved.input_paths}

            enter("prepare")
            backend.prepare()

            enter("generation")
            output_dir.mkdir(parents=True, exist_ok=False)  # never reuse an output dir
            run.result = backend.generate(resolved, output_dir)

            enter("verification")
            self._verify(run.result, output_dir, resolved, run.checksums, self.root)
            run.media = self._measure_output(run.result, resolved, run.warnings)
            enter("done")
        except BackendNotInstalledError:
            raise
        except Exception as exc:
            run.timings[stage] = time.perf_counter() - stage_start
            run.timings["total"] = time.perf_counter() - run_start
            record = self._save_failure(run, exc, stage)
            raise BenchmarkRunError(stage, record, exc) from exc

        run.timings["total"] = time.perf_counter() - run_start
        try:
            record = self._build_record(run)
            self.logger.log(record)
        except FileExistsError:
            raise
        except Exception as exc:
            failed = self._save_failure(run, exc, "logging")
            raise BenchmarkRunError("logging", failed, exc) from exc
        # Verified and recorded: from now on the run's video and derived start
        # frame must not change. Read-only guards them against accidents.
        protect_run_files(record, self.root)
        return record

    # --- helpers ---

    @staticmethod
    def _benchmark_id(request: TaskRequest, benchmark_id: str | None) -> str:
        carried = getattr(request, "benchmark_id", None)
        if carried is not None and benchmark_id is not None and carried != benchmark_id:
            raise ValueError(f"benchmark_id {benchmark_id!r} conflicts with request's {carried!r}")
        chosen = benchmark_id or carried
        if chosen is None:
            raise ValueError(f"benchmark_id is required for {type(request).__name__}")
        check_file_name_safe("benchmark_id", chosen)
        return chosen

    @staticmethod
    def _metadata(backend: VideoBackend) -> BackendMetadata:
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
        request: TaskRequest,
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
        conditioning = result.start_frame_conditioning
        if getattr(request, "start_frame", None):
            if conditioning is None:
                raise RuntimeError(
                    "request has a start_frame but the backend did not report start-frame "
                    "conditioning; refusing a silent downgrade to a reference image"
                )
            if conditioning.derived_file is not None:
                derived = resolve_path(conditioning.derived_file, root)
                if not derived.is_relative_to(output_dir.resolve()) or not derived.is_file():
                    raise RuntimeError(f"derived start frame must be a file inside {output_dir}: {derived}")
        elif conditioning is not None:
            raise RuntimeError("backend reported start-frame conditioning for a request without start_frame")

    def _measure_output(
        self, result: GenerationResult, request: TaskRequest, warnings: list[str],
    ) -> VideoInfo | None:
        """Measure the output video and check it against the backend's claims
        and the request's pinned values. Appends non-fatal findings to `warnings`."""
        requested_fps = getattr(request, "fps", None)
        duration = getattr(request, "duration_seconds", None)
        tolerance = getattr(request, "duration_tolerance_seconds", None)
        try:
            info = self.media_probe(resolve_path(result.output_video, self.root))
        except ProbeUnavailableError as exc:
            if requested_fps is not None or tolerance is not None:
                raise ProbeUnavailableError(
                    f"cannot verify the requested fps/duration without a media probe: {exc}"
                ) from exc
            warnings.append(f"output video not measured ({exc}); fps/frame_count are backend-reported")
            return None
        if not isinstance(info, VideoInfo):
            raise TypeError(f"media probe returned {type(info).__name__}, expected VideoInfo")

        if result.fps is not None and abs(result.fps - info.fps) > FPS_TOLERANCE:
            raise RuntimeError(f"backend reported fps {result.fps:g}, measured {info.fps:g}")
        if result.frame_count is not None and result.frame_count != info.frame_count:
            raise RuntimeError(
                f"backend reported {result.frame_count} frames, measured {info.frame_count}"
            )
        if requested_fps is not None and abs(info.fps - requested_fps) > FPS_TOLERANCE:
            raise RuntimeError(f"requested fps {requested_fps:g}, measured {info.fps:g}")
        if duration is not None and tolerance is not None:
            # Tiny epsilon so a tolerance that is hit exactly is not lost to float rounding.
            if abs(info.duration_seconds - duration) > tolerance + 1e-9:
                raise RuntimeError(
                    f"measured duration {info.duration_seconds:.3f}s ({info.frame_count} frames "
                    f"at {info.fps:g} fps) is not within {tolerance:g}s of the requested {duration:g}s"
                )
        target = request.output_resolution
        if (info.width, info.height) != (target.width, target.height):
            warnings.append(
                f"measured resolution {info.width}x{info.height} differs from requested "
                f"{target.width}x{target.height}"
            )
        return info

    def _collect_environment(self) -> dict[str, Any]:
        try:
            return dict(self.environment_collector())
        except Exception as exc:
            return {"collection_error": f"{_error_type(exc)}: {exc}"}

    def _build_record(self, run: _Run, failure: dict[str, Any] | None = None) -> BenchmarkResult:
        request, meta = run.request, run.meta
        valid_result = run.result if isinstance(run.result, GenerationResult) else None
        effective = valid_result.parameters if valid_result else {}
        model_parameters = {**request.backend_parameters, **effective}

        note_lines = [run.notes]
        if valid_result:
            note_lines += [f"warning: {w}" for w in valid_result.warnings]
            note_lines += [f"warning: {w}" for w in run.warnings]
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

        source_video = getattr(request, "source_video", None)  # generation has none
        return BenchmarkResult(
            benchmark_id=run.benchmark_id,
            run_id=run.run_id,
            task_type=request.capability.value,
            task_parameters={**request.task_parameters(), "run_purpose": run.purpose},
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
            source_video=to_project_relative(source_video, self.root) if source_video else None,
            output_video=(
                to_project_relative(valid_result.output_video, self.root) if valid_result else None
            ),
            reference_assets=[to_project_relative(p, self.root) for p in request.reference_images],
            input_checksums={to_project_relative(p, self.root): h for p, h in run.checksums.items()},
            start_frame=self._start_frame_record(run, valid_result),
            resolution=request.output_resolution,
            # Measured when available (only on success), else backend-reported.
            output_media=run.media.to_record() if run.media and not failure else None,
            fps=run.media.fps if run.media else (valid_result.fps if valid_result else None),
            frame_count=(
                run.media.frame_count if run.media
                else (valid_result.frame_count if valid_result else None)
            ),
            gpu=environment.get("gpu_model"),
            vram_gb=environment.get("vram_gb"),
            validation_time_seconds=run.timings.get("validation"),
            prepare_time_seconds=run.timings.get("prepare"),
            generation_time_seconds=run.timings.get("generation"),
            verification_time_seconds=run.timings.get("verification"),
            total_time_seconds=run.timings.get("total"),
            estimated_cost_usd=run.estimated_cost_usd,
            environment=environment,
            notes="\n".join(line for line in note_lines if line),
        )

    def _start_frame_record(self, run: _Run, result: GenerationResult | None) -> dict[str, Any] | None:
        start_frame = getattr(run.request, "start_frame", None)
        if not start_frame:
            return None
        absolute = resolve_path(start_frame, self.root)
        try:
            original = image_dimensions(absolute)
        except OSError:
            original = None
        conditioning = result.start_frame_conditioning if result else None
        return {
            "path": to_project_relative(absolute, self.root),
            # Checksum taken before generation; None if validation failed first.
            "sha256": run.checksums.get(str(absolute)),
            "original_width": original.width if original else None,
            "original_height": original.height if original else None,
            "preprocessing": list(conditioning.preprocessing) if conditioning else None,
            "effective_width": conditioning.effective_width if conditioning else None,
            "effective_height": conditioning.effective_height if conditioning else None,
            "derived_file": (
                to_project_relative(conditioning.derived_file, self.root)
                if conditioning and conditioning.derived_file else None
            ),
        }

    def _save_failure(self, run: _Run, exc: Exception, stage: str) -> BenchmarkResult | None:
        """Write a status="failed" record. Returns None if that is not possible."""
        failure = {
            "status": "failed",
            "failure_stage": stage,
            "error_type": _error_type(exc),
            "error_message": str(exc)[:MAX_ERROR_MESSAGE_CHARS],
        }
        try:
            record = self._build_record(run, failure)
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
