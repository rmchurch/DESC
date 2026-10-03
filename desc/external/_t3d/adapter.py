"""Isolated equilibrium handoff, immutable physics template and durable run cache.

Like desc.external.gx, this writes inputs and launches a separate process. It is
deliberately not an ExternalObjective: transport profiles do not define the
existing neural heat-flux residual or its derivatives.
"""

from __future__ import annotations

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

from .outcomes import validate_evolution_result, validate_gpu_preflight


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
        models = self.inputs.get("model", [])
        if len(models) != 1 or models[0].get("model") != "AI_GX":
            raise ValueError(
                "Exactly one AI_GX model is required; external GX is not enabled"
            )
        if self.inputs["geometry"].get("geo_option") != "desc":
            raise ValueError("Native DESC geometry is required")
        if self.inputs.get("physics", {}).get("update_equilibrium", False):
            raise ValueError("T3D must not evolve the optimization equilibrium")
        species = self.inputs.get("species", [])
        if sorted(s["type"] for s in species) != ["electron", "hydrogen"]:
            raise ValueError("Verified AI_GX mapping covers hydrogen and electron only")
        if any(s["density"].get("evolve", False) for s in species):
            raise ValueError("AI_GX does not support density transport")
        if any(
            s["temperature"].get("evolve", False)
            for s in species
            if s["type"] == "electron"
        ):
            raise ValueError(
                "Verified AI_GX mapping does not evolve electron temperature"
            )
        if not self.inputs.get("log", {}).get("output_netcdf", False):
            raise ValueError("NetCDF output with explicit units is required")
        if "f_save" in self.inputs.get("log", {}):
            raise ValueError(
                "Remove f_save: output must stay in the isolated run directory"
            )
        model = models[0]

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
        roots = [str(Path(__file__).resolve().parents[3])]
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
            [self.python, "-m", "desc.external._t3d.worker", "--probe"],
            env=probe_env,
            text=True,
            capture_output=True,
            timeout=config.probe_timeout_seconds,
            check=True,
        )
        self.runtime = json.loads(probe.stdout)
        self.assets = {
            "template": file_hash(self.template),
            "gx_template": file_hash(self.gx_template),
            # The native v2 loader reads checkpoint hyperparameters and does not
            # access this legacy CSV. The original template points at a missing CSV.
            "csv": file_hash(self.csv_path) if self.csv_path.is_file() else None,
            "weights": {p.name: file_hash(p) for p in self.weights},
        }
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

    def _verify_assets(self):
        current = {
            "template": file_hash(self.template),
            "gx_template": file_hash(self.gx_template),
            "csv": file_hash(self.csv_path) if self.csv_path.is_file() else None,
            "weights": {
                p.name: file_hash(p) for p in model_weight_paths(self.model_dir)
            },
        }
        if current != self.assets:
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
            from .worker import parse_output

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
                "desc.external._t3d.worker",
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
