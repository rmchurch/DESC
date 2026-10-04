"""Evaluate native T3D AI_GX transport on explicitly configured DESC equilibria.

Like ``desc.external.gx.gx``, the public callable accepts an equilibrium sequence.
It returns native ion-temperature profiles, without choosing an objective,
gradient, optimizer or equilibrium feedback. A small worker isolates the external
Python runtime; input rendering, caching, parsing and verification live here.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

__all__ = [
    "T3DAdapter",
    "T3DConfig",
    "IonTemperatureProfile",
    "T3DEvaluationError",
    "evaluate_t3d",
    "read_ion_temperature",
    "t3d",
]


def load_toml(text):
    """Parse explicit transport inputs with an optional Python 3.10 dependency."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise ImportError("T3D inputs require tomli on Python 3.10") from exc
    return tomllib.loads(text)


def digest(value):
    """Hash JSON-compatible execution identity content."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, default=_json_default
        ).encode()
    ).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def file_hash(path):
    """Hash a file without loading its complete content into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_weight_paths(root):
    """Match the installed v2 loader, excluding AppleDouble metadata sidecars."""
    return sorted(
        p
        for p in Path(root).glob("*.pth")
        if p.is_file() and not p.name.startswith("._")
    )


def atomic_json(path, value):
    """Write a run record through an atomic replacement."""
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False, default=_json_default
        )
        + "\n"
    )
    os.replace(tmp, path)


def equilibrium_identity(eq):
    """Hash physical params plus resolution, avoiding HDF5 serialization metadata."""
    params = {}
    for key, value in sorted(eq.params_dict.items()):
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Nonfinite equilibrium parameter: {key}")
        params[key] = {"shape": array.shape, "values": array.tolist()}
    return digest(
        {
            "params": params,
            "resolution": {
                key: getattr(eq, key)
                for key in ("L", "M", "N", "L_grid", "M_grid", "N_grid", "NFP", "sym")
            },
        }
    )


@dataclasses.dataclass(frozen=True)
class T3DConfig:
    """Explicit template and runtime settings for isolated AI_GX evaluation.

    Parameters
    ----------
    template : str
        Required user-supplied T3D input file. DESC supplies no physics template,
        trained weights, model directory or transport stopping horizon.
    python : str
        Interpreter with the external T3D/AI_GX runtime installed.
    source_root : str or None
        Optional directory containing the external T3D Python package.
    timeout_seconds, probe_timeout_seconds : float
        Bounds for execution and the CPU-only version/source probe.
    mode : {"evolve", "initialize"}
        Execute the supplied transport horizon, or initialize without advancing.
    retry_token : str
        Explicit new identity for a deliberate retry; no implicit retries occur.
    environment : dict
        Explicit subprocess environment overrides, including device visibility.
    require_gpu : bool
        Require one active native CUDA device and reject CPU fallback.
    """

    template: str
    python: str = sys.executable
    source_root: str | None = None
    timeout_seconds: float = 1800
    probe_timeout_seconds: float = 180
    mode: str = "evolve"  # initialize constructs native engine without advancing time
    retry_token: str = ""  # explicit new identity, never an automatic retry
    environment: dict = dataclasses.field(default_factory=dict)
    require_gpu: bool = (
        False  # opt-in native validation contract, no silent CPU fallback
    )

    def __post_init__(self):
        if self.timeout_seconds <= 0 or not np.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be finite and positive")
        if self.probe_timeout_seconds <= 0 or not np.isfinite(
            self.probe_timeout_seconds
        ):
            raise ValueError("probe_timeout_seconds must be finite and positive")
        if self.mode not in {"evolve", "initialize"}:
            raise ValueError("mode must be evolve or initialize")


def _set_key(text, section, key, value):
    """Replace only a specified TOML key, retaining comments and all physics inputs."""
    lines = text.splitlines(keepends=True)
    active = False
    replaced = 0
    for i, line in enumerate(lines):
        header = re.match(r"^\s*(\[\[?[^\]]+\]\]?)", line)
        if header:
            active = header.group(1) == section
        if active and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            lines[i] = f"  {key} = {json.dumps(value)}\n"
            replaced += 1
    if replaced != 1:
        raise ValueError(f"Template must contain one {section}.{key}")
    return "".join(lines)


class T3DAdapter:
    """Run configured native T3D once per equilibrium/configuration identity.

    Parameters
    ----------
    config : T3DConfig
        Explicit template, runtime, device and resource configuration.
    output_dir : path-like
        Dedicated directory for atomic run claims and immutable cached artifacts.
    """

    def __init__(self, config: T3DConfig, output_dir):
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.template = Path(config.template).resolve()
        self.text = self.template.read_text()
        self.inputs = load_toml(self.text)
        _verified_ion(self.inputs)
        if not self.inputs.get("log", {}).get("output_netcdf", False):
            raise ValueError("NetCDF output with explicit units is required")
        if "f_save" in self.inputs.get("log", {}):
            raise ValueError(
                "Remove f_save: output must stay in the isolated run directory"
            )
        model = self.inputs["model"][0]

        def resolve(value):
            path = Path(value)
            return (
                path if path.is_absolute() else self.template.parent / path
            ).resolve()

        self.model_dir = resolve(model["model_dir"])
        self.csv_path = resolve(model["csv_path"])
        self.gx_template = resolve(model["gx_template"])
        self.weights = model_weight_paths(self.model_dir)
        if not self.weights or not self.gx_template.is_file():
            raise FileNotFoundError(
                "Existing model weights and GX template are required; no downloads"
            )
        if "v2" not in str(self.model_dir):
            raise ValueError(
                "This adapter verifies the existing v2 ensemble; "
                "use a v2 model directory"
            )
        self.env = os.environ.copy()
        self.env.update({str(k): str(v) for k, v in config.environment.items()})
        roots = [str(Path(__file__).resolve().parents[2])]
        if config.source_root:
            roots.append(str(Path(config.source_root).resolve()))
        roots.append(self.env.get("PYTHONPATH", ""))
        self.env["PYTHONPATH"] = os.pathsep.join(roots)
        executable = shutil.which(config.python)
        if executable is None:
            raise FileNotFoundError(config.python)
        self.python = str(Path(executable).resolve())
        # Identity reads versions/sources only. It must also work during post-job
        # verification on a login node with no GPU. The actual worker keeps self.env.
        probe_env = self.env.copy()
        probe_env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        probe = subprocess.run(
            [self.python, "-m", "desc.external._t3d_worker", "--probe"],
            env=probe_env,
            text=True,
            capture_output=True,
            timeout=config.probe_timeout_seconds,
            check=True,
        )
        self.runtime = json.loads(probe.stdout)
        self.assets = self._asset_hashes()
        self.identity = digest(
            {
                "assets": self.assets,
                "runtime": self.runtime,
                "mode": config.mode,
                "retry_token": config.retry_token,
                "require_gpu": config.require_gpu,
                "environment": config.environment,
            }
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _asset_hashes(self):
        """Capture the template and every checkpoint used by the native loader."""
        return {
            "template": file_hash(self.template),
            "gx_template": file_hash(self.gx_template),
            # The v2 loader reads checkpoint hyperparameters, not the legacy CSV.
            "csv": file_hash(self.csv_path) if self.csv_path.is_file() else None,
            "weights": {
                p.name: file_hash(p) for p in model_weight_paths(self.model_dir)
            },
        }

    def _verify_assets(self):
        if self._asset_hashes() != self.assets:
            raise RuntimeError(
                "T3D inputs/model changed during optimization; construct a new adapter"
            )

    def validate_cached(self, run_id):
        """Verify durable successful output without launching a worker."""
        run_dir = self.output_dir / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"Incomplete run claim requires explicit recovery: {run_dir}"
            )
        previous = json.loads(manifest_path.read_text())
        if previous["status"] not in {"evolved", "initialized"}:
            raise RuntimeError(
                f"No automatic retry for {previous['status']} run: {run_dir}"
            )
        if previous["config_identity"] != self.identity:
            raise RuntimeError("Cached manifest belongs to a different configuration")
        for name, expected in previous["artifact_hashes"].items():
            if file_hash(run_dir / name) != expected:
                raise RuntimeError(f"Cached artifact changed: {run_dir / name}")
        self._validate_result(previous["result"], run_dir)
        return previous

    def _validate_result(self, result, run_dir):
        if result.get("schema_version") != 2 or result["runtime"] != self.runtime:
            raise RuntimeError(
                "T3D/DESC runtime or result schema changed since preflight"
            )
        if (
            result.get("mode") != self.config.mode
            or result.get("process_completed") is not True
        ):
            raise RuntimeError("Worker mode or process completion claim is invalid")
        if self.config.require_gpu:
            validate_gpu_preflight(result["gpu_preflight"])
            execution = result["execution"]
            if (
                execution["ai_gx_device"] != "cuda"
                or execution["jax_backend"] not in {"gpu", "cuda"}
                or execution["torch_visible_gpu_count"] != 1
                or execution["model_parameter_devices"] != ["cuda:0"]
                or result["ensemble_count"] != len(self.weights)
            ):
                raise RuntimeError("Required single-GPU native backend was not active")
        if self.config.mode == "initialize":
            if (
                result["status"] != "initialized"
                or result.get("requested_evolution_completed") is not False
            ):
                raise RuntimeError(
                    "Initialization is not transport evolution validation"
                )
        else:
            transport = parse_output(run_dir / "transport.nc")
            validate_evolution_result(result, transport, self.inputs)

    def evaluate(self, eq):
        """Run once per equilibrium/model/configuration, without implicit retries."""
        self._verify_assets()
        eq_id = equilibrium_identity(eq)
        run_id = digest({"equilibrium": eq_id, "config": self.identity})
        run_dir = self.output_dir / run_id
        manifest_path = run_dir / "manifest.json"
        try:
            run_dir.mkdir()  # atomic claim across processes
        except FileExistsError:
            return self.validate_cached(run_id)
        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "status": "running",
            "mode": self.config.mode,
            "process_completed": False,
            "requested_evolution_completed": False,
            "equilibrium_identity": eq_id,
            "config_identity": self.identity,
            "assets": self.assets,
            "runtime": self.runtime,
            "template_source": str(self.template),
            "model_dir": str(self.model_dir),
            "csv_path": str(self.csv_path),
            "started_unix": time.time(),
        }
        atomic_json(manifest_path, manifest)
        try:
            eq.save(str(run_dir / "equilibrium.h5"))
            text = _set_key(
                self.text, "[geometry]", "geo_file", str(run_dir / "equilibrium.h5")
            )
            for key, value in (
                ("model_dir", str(self.model_dir)),
                ("csv_path", str(self.csv_path)),
                ("gx_template", "gx_template.in"),
                ("gx_outputs", "gx-flux-tubes/"),
            ):
                text = _set_key(text, "[[model]]", key, value)
            rendered = load_toml(text)
            # An explicit audit that only handoff/resource paths changed.
            expected = json.loads(json.dumps(self.inputs))
            expected["geometry"]["geo_file"] = rendered["geometry"]["geo_file"]
            for key in ("model_dir", "csv_path", "gx_template", "gx_outputs"):
                expected["model"][0][key] = rendered["model"][0][key]
            if rendered != expected:
                raise RuntimeError("Rendered input changed physics")
            (run_dir / "transport.in").write_text(text)
            shutil.copyfile(self.gx_template, run_dir / "gx_template.in")
            command = [
                self.python,
                "-m",
                "desc.external._t3d_worker",
                "--mode",
                self.config.mode,
                "transport.in",
            ]
            if self.config.require_gpu:
                command.append("--require-gpu")
            manifest["command"] = command
            atomic_json(manifest_path, manifest)
            with (run_dir / "process.log").open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=run_dir,
                    env=self.env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.config.timeout_seconds,
                )
            manifest["returncode"] = completed.returncode
            manifest["process_completed"] = completed.returncode == 0
            if completed.returncode:
                raise RuntimeError(
                    f"T3D exited {completed.returncode}; see {run_dir / 'process.log'}"
                )
            result = json.loads((run_dir / "result.json").read_text())
            self._validate_result(result, run_dir)
            manifest.update(status=result["status"], result=result)
            manifest["requested_evolution_completed"] = result[
                "requested_evolution_completed"
            ]
            names = [
                "equilibrium.h5",
                "transport.in",
                "gx_template.in",
                "result.json",
                "process.log",
            ]
            if (run_dir / "backend.json").is_file():
                names.append("backend.json")
            if (run_dir / "gpu_preflight.json").is_file():
                names.append("gpu_preflight.json")
            if self.config.mode == "evolve":
                names.append("transport.nc")
            manifest["artifact_hashes"] = {
                name: file_hash(run_dir / name) for name in names
            }
            if result["status"] == "stopped_early":
                manifest["failure_kind"] = "premature_stop"
                raise RuntimeError(
                    "T3D stopped before requested evolution: "
                    f"{result['transport_stop_reason']}; "
                    f"achieved step={result['transport_step']}, "
                    f"time={result['transport_time']}; "
                    f"see {manifest_path}"
                )
        except Exception as exc:
            manifest.update(
                status=(
                    manifest["status"]
                    if manifest["status"] == "stopped_early"
                    else "failed"
                ),
                error=f"{type(exc).__name__}: {exc}",
            )
            manifest.setdefault(
                "failure_kind",
                (
                    "timeout"
                    if isinstance(exc, subprocess.TimeoutExpired)
                    else "invalid_or_failed_output"
                ),
            )
            manifest["requested_evolution_completed"] = False
            exc.run_id = run_id
            exc.failure_kind = manifest["failure_kind"]
            raise
        finally:
            manifest["finished_unix"] = time.time()
            atomic_json(manifest_path, manifest)
        return manifest


def validate_gpu_preflight(report):
    """Validate native arithmetic and the CPU metadata backend."""
    if (
        report["jax_backend"] not in {"gpu", "cuda"}
        or report["jax_array_platform"] not in {"gpu", "cuda"}
        or report["torch_visible_gpu_count"] != 1
        or report["torch_array_device"] != "cuda:0"
        or report["cpu_backend_available"] is not True
        or not np.isfinite(report["jax_sum"])
        or report["jax_sum"] != 3.0
        or not np.isfinite(report["torch_sum"])
        or report["torch_sum"] != 3.0
    ):
        raise RuntimeError(
            "GPU preflight requires one working CUDA device "
            "and DESC's CPU metadata backend"
        )


def evolution_outcome(step, time, requested_steps, requested_time, transport):
    """Classify the native OR time/step stopping limit; do not change that limit."""
    if (
        type(step) is not int
        or step < 0
        or type(requested_steps) is not int
        or requested_steps < 1
    ):
        raise ValueError("Invalid achieved/requested transport step count")
    if (
        not np.isfinite(time)
        or time < 0
        or not np.isfinite(requested_time)
        or requested_time <= 0
    ):
        raise ValueError("Invalid achieved/requested transport time")
    if transport["recorded_transport_step"] != step or not np.isclose(
        transport["final_profile_time"], time, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Engine progress disagrees with recorded transport output")
    progressed = step >= 1 and time > 0
    time_reached = time >= requested_time or np.isclose(
        time, requested_time, rtol=1e-10, atol=1e-12
    )
    steps_reached = step >= requested_steps
    horizon_reached = bool(time_reached or steps_reached)
    success = bool(progressed and horizon_reached)
    reason = (
        "no_transport_progress"
        if not progressed
        else (
            "requested_step_limit"
            if steps_reached
            else "requested_time_limit" if time_reached else "before_requested_horizon"
        )
    )
    return {
        "status": "evolved" if success else "stopped_early",
        "process_completed": True,
        "transport_step": step,
        "transport_time": float(time),
        "requested_steps": requested_steps,
        "requested_time": float(requested_time),
        "transport_output_valid": True,
        "transport_progressed": bool(progressed),
        "transport_horizon_reached": horizon_reached,
        "requested_evolution_completed": success,
        "premature_stop": not success,
        "transport_stop_reason": reason,
        "note": (
            "Requested evolution completion does not establish "
            "scientific convergence"
        ),
    }


def validate_evolution_result(result, transport, inputs):
    """Recompute worker claims using independently parsed output and input limits."""
    grid = inputs["grid"]
    if (
        len(transport["rho"]["values"]) != grid["N_radial"]
        or transport["flux_label"] != grid["flux_label"]
        or not np.isclose(
            transport["rho"]["values"][-1], grid["rho_edge"], rtol=1e-10, atol=1e-12
        )
    ):
        raise ValueError("Transport grid disagrees with the immutable input")
    requested = inputs.get("time", {})
    steps = requested.get("N_steps", 1000)  # verified native Time.py default
    end_time = requested.get("t_max", 1000)
    if requested.get("use_SI", False):
        end_time /= transport["normalizations"]["t_ref"]["values"]
    if result["requested_steps"] != steps or not np.isclose(
        result["requested_time"], end_time, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Worker requested limits disagree with the immutable input")
    expected = evolution_outcome(
        result["transport_step"], result["transport_time"], steps, end_time, transport
    )
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"Worker transport claim disagrees with recorded output: {key}"
            )
    if result["transport"] != transport:
        raise ValueError(
            "Worker parsed transport differs from independently read output"
        )
    return expected


def runtime_identity():
    """Record actual external source and runtime library versions."""
    from importlib import metadata

    from t3d import trinity_lib
    from t3d.ai import inference_engine
    from t3d.flux_models import AI_GX

    import desc

    # Include dirty working source, not just git HEAD/version.
    roots = {
        "desc": Path(desc.__file__).parent,
        "t3d": Path(trinity_lib.__file__).parent,
    }
    return {
        "desc_version": desc.__version__,
        "t3d_version": metadata.version("t3d"),
        "source_roots": {k: str(v) for k, v in roots.items()},
        "sources": {
            k: digest(
                {
                    str(p.relative_to(root)): file_hash(p)
                    for p in sorted(root.rglob("*.py"))
                }
            )
            for k, root in roots.items()
        },
        "ai_gx": file_hash(AI_GX.__file__),
        "inference_engine": file_hash(inference_engine.__file__),
        "adapter_sources": digest(
            {
                name: file_hash(Path(__file__).with_name(name))
                for name in ("t3d.py", "_t3d_worker.py")
            }
        ),
        "libraries": {
            name: metadata.version(name)
            for name in ("numpy", "scipy", "torch", "jax", "jaxlib", "netCDF4")
        },
    }


def gpu_preflight():
    """Tiny native JAX/Torch kernels before constructing transport geometry/models."""
    import jax
    import jax.numpy as jnp
    import torch

    if jax.default_backend() not in {"gpu", "cuda"} or not torch.cuda.is_available():
        raise RuntimeError("GPU preflight found no active native CUDA backend")
    jax_array = jnp.asarray([1.0, 2.0])
    torch_array = torch.tensor([1.0, 2.0], device="cuda:0")
    report = {
        "jax_backend": jax.default_backend(),
        "jax_array_platform": jax_array.device.platform,
        "jax_array_device": str(jax_array.device),
        "cpu_backend_available": bool(jax.devices("cpu")),
        "cpu_devices": [str(device) for device in jax.devices("cpu")],
        "torch_visible_gpu_count": torch.cuda.device_count(),
        "torch_array_device": str(torch_array.device),
        "gpu_name": torch.cuda.get_device_name(0),
        "jax_sum": float(jax_array.sum().block_until_ready()),
        "torch_sum": float(torch_array.sum().item()),
        "note": "Backend arithmetic only; no transport timestep or model inference",
    }
    validate_gpu_preflight(report)
    return report


def parse_output(path):
    """Parse native profiles and separate preceding flux snapshots."""
    return _read_native_output(path)[0]


def _read_native_output(path, inputs=None):
    """Read native transport and optional Ti metadata in one file context."""
    from netCDF4 import Dataset

    def read(group, name, index=None, shape=None, units=None):
        variable = group.variables[name]
        values = variable[:]
        if np.ma.is_masked(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid/masked output: {name}")
        if shape is not None and values.shape != shape:
            raise ValueError(
                f"Invalid shape for {name}: {values.shape}, expected {shape}"
            )
        actual_units = getattr(variable, "units", None)
        if units is not None and actual_units != units:
            raise ValueError(f"Invalid units for {name}: {actual_units}")
        values = values if index is None else values[index]
        return {"values": np.asarray(values).tolist(), "units": actual_units}

    with Dataset(path) as dataset:
        species = dataset.groups["species"]
        tags = [str(tag) for tag in species.variables["species_tags"][:]]
        if sorted(tags) != ["H", "e"]:
            raise ValueError(
                "Output species disagree with verified hydrogen/electron mapping"
            )
        t = read(dataset.groups["time"], "t", units="[t_ref]")
        count = len(t["values"])
        if count < 2:
            raise ValueError("No transport iteration was recorded")
        times = np.asarray(t["values"])
        steps = read(dataset.groups["time"], "t_step_idx", shape=(count,))["values"]
        if (
            times.shape != (count,)
            or np.any(times < 0)
            or np.any(np.diff(times) < 0)
            or np.any(np.asarray(steps) < 0)
            or np.any(np.diff(steps) < 0)
            or any(int(s) != s for s in steps)
        ):
            raise ValueError("Invalid/nonmonotonic recorded transport progress")
        rho = read(dataset.groups["grid"], "rho", units="-")
        flux_label = str(dataset.groups["grid"].variables["flux_label"][()])
        radial_count = len(rho["values"])
        midpoints = read(
            dataset.groups["grid"], "midpoints", shape=(radial_count - 1,), units="-"
        )
        if (
            radial_count < 2
            or np.any(np.diff(rho["values"]) <= 0)
            or np.any(np.asarray(rho["values"]) < 0)
            or np.any(np.asarray(rho["values"]) > 1)
            or np.any(np.asarray(midpoints["values"]) <= np.asarray(rho["values"][:-1]))
            or np.any(np.asarray(midpoints["values"]) >= np.asarray(rho["values"][1:]))
        ):
            raise ValueError("Invalid radial grid")
        norms = {
            name: read(dataset.groups["norms"], name, shape=())
            for name in dataset.groups["norms"].variables
        }
        if norms["t_ref"]["units"] != "s" or norms["t_ref"]["values"] <= 0:
            raise ValueError("Invalid time normalization")
        profile_units = {"T": "keV", "n": "10^20 m^-3", "p": "10^20 m^-3 keV"}
        flux_units = {"qflux": "[GB]", "Q_MW": "MW", "aLT": "-", "aLn": "-"}
        # Native T3D explicitly documents the final appended row has stale fluxes.
        # Keep final temperatures and the preceding flux snapshot with separate times.
        transport = {
            "time": t,
            "step_indices": steps,
            "recorded_transport_step": int(steps[-1]),
            "rho": rho,
            "midpoints": midpoints,
            "flux_label": flux_label,
            "normalizations": norms,
            "final_profile_time": t["values"][-1],
            "flux_snapshot_time": t["values"][-2],
            "final_row_flux_consistent": False,
            "profiles": {
                tag: {
                    key: read(species, f"{key}_{tag}", -1, (count, radial_count), units)
                    for key, units in profile_units.items()
                }
                for tag in tags
            },
            "flux_snapshot": {
                tag: {
                    key: read(
                        species, f"{key}_{tag}", -2, (count, radial_count - 1), units
                    )
                    for key, units in flux_units.items()
                }
                for tag in tags
            },
        }

        metadata = (
            _native_metadata(dataset, inputs, transport) if inputs is not None else None
        )
        return transport, metadata


@dataclasses.dataclass(frozen=True)
class IonTemperatureProfile:
    """One equilibrium's final native Ti samples; arrays are owned/read-only.

    ``temperature[0]`` is the first recorded radial point, not an axis/core
    temperature. Evidence and provenance are independent JSON-compatible copies.
    ``evolved`` means the requested native time/step horizon completed, not that
    scientific steady-state convergence was assessed.
    """

    temperature: np.ndarray
    rho: np.ndarray
    temperature_units: str
    rho_units: str
    flux_label: str
    coordinate_definition: str
    includes_magnetic_axis: bool
    species_type: str
    species_tag: str
    bulk_ion_tag: str
    time: float
    time_units: str
    t_ref_seconds: float
    time_seconds: float
    transport_step: int
    status: str
    evidence: dict
    provenance: dict

    def to_dict(self):
        """Return an independent, JSON-compatible snapshot for consumers."""
        result = dataclasses.asdict(self)
        result["temperature"] = self.temperature.tolist()
        result["rho"] = self.rho.tolist()
        return result


class T3DEvaluationError(RuntimeError):
    """No usable Ti profile; retains run status/evidence without retrying."""

    def __init__(self, message, *, run_dir=None, manifest=None):
        super().__init__(message)
        manifest = manifest if isinstance(manifest, dict) else {}
        self.run_dir = str(run_dir) if run_dir is not None else None
        self.run_id = manifest.get("run_id")
        self.status = manifest.get("status", "unavailable")
        self.failure_kind = manifest.get("failure_kind")
        self.evidence = _outcome_evidence(manifest)
        self.evidence["usable_ion_temperature_profile"] = False


def _outcome_evidence(manifest):
    result = manifest.get("result", {})
    result = result if isinstance(result, dict) else {}
    names = (
        "process_completed",
        "requested_evolution_completed",
        "transport_step",
        "transport_time",
        "requested_steps",
        "requested_time",
        "transport_output_valid",
        "transport_progressed",
        "transport_horizon_reached",
        "premature_stop",
        "transport_stop_reason",
        "note",
    )
    evidence = {name: result.get(name, manifest.get(name)) for name in names}
    evidence.update(scientific_convergence_assessed=False, scientific_convergence=None)
    return copy.deepcopy(evidence)


def _verified_ion(inputs):
    """Reject ambiguous species before any worker can be launched."""
    species = inputs.get("species", [])
    if not isinstance(species, list) or any(
        not isinstance(s, dict) or "type" not in s for s in species
    ):
        raise ValueError("Explicit species identities are required")
    ions = [s for s in species if s["type"] != "electron"]
    if not ions:
        raise ValueError("No ion species: cannot return ion temperature")
    if len(ions) != 1:
        raise ValueError("Multiple ion species: no implicit ion selection or averaging")
    ion = ions[0]
    electrons = [s for s in species if s["type"] == "electron"]
    if (
        ion["type"] != "hydrogen"
        or (ion.get("tag") or "H") != "H"
        or len(electrons) != 1
        or (electrons[0].get("tag") or "e") != "e"
    ):
        raise ValueError("Verified Ti interface requires hydrogen/H and electron/e")
    if any(s.get("density", {}).get("evolve", False) for s in species):
        raise ValueError("Verified AI_GX mapping does not evolve density")
    if electrons[0].get("temperature", {}).get("evolve", False):
        raise ValueError("Verified AI_GX mapping does not evolve electron temperature")
    if (
        inputs.get("grid", {}).get("flux_label") != "torflux"
        or inputs.get("geometry", {}).get("geo_option") != "desc"
        or inputs.get("physics", {}).get("update_equilibrium", False)
    ):
        raise ValueError(
            "Verified Ti interface requires torflux, DESC, and a fixed equilibrium"
        )
    models = inputs.get("model", [])
    if len(models) != 1 or models[0].get("model") != "AI_GX":
        raise ValueError("Exactly one AI_GX transport model is required")
    return ion


def _native_metadata(dataset, inputs, transport):
    """Read identity and raw solver evidence, retaining the native row times."""
    species = dataset.groups["species"]
    tags = [str(x) for x in species.variables["species_tags"][:]]
    types = [str(x) for x in species.variables["species_types"][:]]
    if len(tags) != len(types) or len(tags) != len(set(tags)):
        raise ValueError("Missing/ambiguous native species identity")
    ions = [(kind, tag) for kind, tag in zip(types, tags) if kind != "electron"]
    if not ions:
        raise ValueError("No native ion species: cannot label an electron profile Ti")
    if len(ions) != 1:
        raise ValueError("Multiple native ion species: no implicit ion selection")
    if dict(zip(tags, types)) != {"H": "hydrogen", "e": "electron"}:
        raise ValueError("Native species identities disagree with immutable input")
    bulk_tag = str(species.variables["bulk_ion_tag"][()])
    if bulk_tag != "H":
        raise ValueError("Native bulk ion disagrees with selected hydrogen/H")
    times = transport["time"]["values"]
    count = len(times)
    tg = dataset.groups["time"]
    trace = {}
    for name in ("t_rms", "t_iter_idx"):
        variable = tg.variables[name]
        values = variable[:]
        if (
            values.shape != (count,)
            or np.ma.is_masked(values)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError(f"Invalid native solver evidence: {name}")
        if name == "t_iter_idx" and np.any(values != np.floor(values)):
            raise ValueError("Noninteger native Newton iteration indices")
        trace[name] = {
            "values": np.asarray(values).tolist(),
            "units": getattr(variable, "units", None),
            "description": getattr(variable, "description", None),
        }
    trace.update(
        time=copy.deepcopy(transport["time"]),
        step_indices=copy.deepcopy(transport["step_indices"]),
        time_input=copy.deepcopy(inputs.get("time", {})),
        note="Raw native rows, including the appended final profile row; "
        "no new RMS threshold or scientific convergence decision",
    )
    return bulk_tag, trace


def read_ion_temperature(run_dir) -> IonTemperatureProfile:
    """Read an archived successful run without execution or a runtime/GPU probe.

    Verify stored artifact hashes and reparse native output against its immutable
    input/result. Historical model/source hashes are reported as recorded; the
    current installation need not equal the archived runtime. This is distinct
    from T3DAdapter's stricter current-configuration cache lookup.
    """
    run_dir = Path(run_dir).resolve()
    manifest = {}
    try:
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema_version") != 2
            or manifest.get("mode") != "evolve"
            or manifest.get("status") != "evolved"
            or manifest.get("returncode") != 0
            or manifest.get("process_completed") is not True
            or manifest.get("requested_evolution_completed") is not True
        ):
            raise ValueError(
                f"No usable Ti profile for run status {manifest.get('status')!r}"
            )
        expected_id = digest(
            {
                "equilibrium": manifest["equilibrium_identity"],
                "config": manifest["config_identity"],
            }
        )
        if manifest["run_id"] != expected_id:
            raise ValueError(
                "Run identity disagrees with archived equilibrium/configuration"
            )
        hashes = manifest["artifact_hashes"]
        required = {
            "equilibrium.h5",
            "transport.in",
            "gx_template.in",
            "result.json",
            "transport.nc",
            "process.log",
        }
        if not required.issubset(hashes):
            raise ValueError("Missing required artifact hashes for native Ti")
        for name, expected in hashes.items():
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid archived artifact name")
            if file_hash(run_dir / name) != expected:
                raise ValueError(f"Archived artifact changed: {name}")
        result = json.loads((run_dir / "result.json").read_text())
        if (
            result != manifest["result"]
            or result.get("schema_version") != 2
            or result.get("runtime") != manifest["runtime"]
            or result.get("mode") != "evolve"
        ):
            raise ValueError("Archived worker result/runtime disagrees with manifest")
        inputs = load_toml((run_dir / "transport.in").read_text())
        _verified_ion(inputs)
        if (
            result.get("model_class") != "AI_GX_FluxModel"
            or not manifest["assets"]["weights"]
            or result.get("ensemble_count") != len(manifest["assets"]["weights"])
            or manifest["assets"]["gx_template"] != hashes["gx_template.in"]
        ):
            raise ValueError("Archived AI_GX model evidence is incomplete/inconsistent")
        if result.get("gpu_preflight") is not None:
            validate_gpu_preflight(result["gpu_preflight"])
        transport, metadata = _read_native_output(run_dir / "transport.nc", inputs)
        validate_evolution_result(result, transport, inputs)
        bulk_tag, trace = metadata
        temperature = np.array(
            transport["profiles"]["H"]["T"]["values"], dtype=float, copy=True
        )
        rho = np.array(transport["rho"]["values"], dtype=float, copy=True)
        if temperature.shape != rho.shape or temperature.ndim != 1:
            raise ValueError("Ion temperature samples disagree with radial coordinates")
        temperature.setflags(write=False)
        rho.setflags(write=False)
        t_ref = float(transport["normalizations"]["t_ref"]["values"])
        final_time = float(transport["final_profile_time"])
        evidence = _outcome_evidence(manifest)
        evidence["usable_ion_temperature_profile"] = True
        evidence["native_solver_trace"] = trace
        evidence["final_row_flux_consistent"] = transport["final_row_flux_consistent"]
        provenance = {
            name: copy.deepcopy(manifest[name])
            for name in (
                "run_id",
                "equilibrium_identity",
                "config_identity",
                "runtime",
                "assets",
                "model_dir",
                "template_source",
                "artifact_hashes",
            )
        }
        provenance.update(
            run_dir=str(run_dir),
            manifest_path=str(manifest_path),
            manifest_sha256=file_hash(manifest_path),
            native_output=str(run_dir / "transport.nc"),
            temperature_variable="species/T_H",
            native_profile_row=-1,
            species_input=copy.deepcopy(inputs["species"]),
            execution=copy.deepcopy(result.get("execution")),
            ensemble_count=result["ensemble_count"],
            note="Model/source hashes are archived execution provenance; "
            "reading this profile does not revalidate current model files",
        )
        return IonTemperatureProfile(
            temperature=temperature,
            rho=rho,
            temperature_units="keV",
            rho_units="-",
            flux_label=transport["flux_label"],
            coordinate_definition="rho = sqrt(toroidal_flux / toroidal_flux_LCFS)",
            includes_magnetic_axis=bool(rho[0] == 0),
            species_type="hydrogen",
            species_tag="H",
            bulk_ion_tag=bulk_tag,
            time=final_time,
            time_units="[t_ref]",
            t_ref_seconds=t_ref,
            time_seconds=final_time * t_ref,
            transport_step=transport["recorded_transport_step"],
            status=manifest["status"],
            evidence=evidence,
            provenance=provenance,
        )
    except T3DEvaluationError:
        raise
    except Exception as exc:
        raise T3DEvaluationError(
            f"Cannot return ion temperature: {exc}", run_dir=run_dir, manifest=manifest
        ) from exc


def _validate_adapter(adapter):
    if not isinstance(adapter, T3DAdapter):
        raise TypeError("Pass an existing T3DAdapter explicitly")
    if adapter.config.mode != "evolve":
        raise ValueError(
            "Ti evaluation requires mode='evolve'; initialization has no evolved Ti"
        )
    _verified_ion(adapter.inputs)


def evaluate_t3d(eq, *, adapter: T3DAdapter) -> IonTemperatureProfile:
    """Evaluate one current DESC equilibrium via the existing handoff/cache."""
    _validate_adapter(adapter)
    try:
        manifest = adapter.evaluate(eq)
    except Exception as exc:
        # Existing failures/cache refusals retain their original identity/status.
        manifest, run_dir = {}, None
        try:
            run_id = digest(
                {"equilibrium": equilibrium_identity(eq), "config": adapter.identity}
            )
            run_dir = adapter.output_dir / run_id
            path = run_dir / "manifest.json"
            if path.is_file():
                manifest = json.loads(path.read_text())
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        raise T3DEvaluationError(
            f"T3D evaluation produced no usable Ti: {exc}",
            run_dir=run_dir,
            manifest=manifest,
        ) from exc
    return read_ion_temperature(adapter.output_dir / manifest["run_id"])


def t3d(equilibria, *, adapter: T3DAdapter) -> list[IonTemperatureProfile]:
    """Like gx(eq_list), evaluate in input order; return one radial profile per eq.

    Profiles are not reduced or stacked: their native grids/times stay explicit.
    Execution is serial and cached; any failure raises, with no partial batch
    return or automatic retry. Successful preceding runs remain in the cache.
    """
    _validate_adapter(adapter)
    if hasattr(equilibria, "params_dict"):
        raise TypeError(
            "t3d expects an equilibrium sequence; use evaluate_t3d for one eq"
        )
    equilibria = list(equilibria)
    for eq in equilibria:
        equilibrium_identity(eq)  # validate the whole batch before costly execution
    return [evaluate_t3d(eq, adapter=adapter) for eq in equilibria]
