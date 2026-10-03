"""GX-like equilibrium evaluation returning a verified ion temperature profile.

Execution belongs to T3DAdapter. This module selects no scalar objective, performs
no extrapolation, and supplies no derivative or equilibrium pressure feedback.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .adapter import T3DAdapter, digest, equilibrium_identity, file_hash, load_toml
from .outcomes import validate_evolution_result, validate_gpu_preflight


@dataclass(frozen=True)
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
        result = asdict(self)
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


def _native_metadata(path, inputs, transport):
    """Read identity and raw solver evidence, retaining the native row times."""
    from netCDF4 import Dataset

    with Dataset(path) as dataset:
        species = dataset.groups["species"]
        tags = [str(x) for x in species.variables["species_tags"][:]]
        types = [str(x) for x in species.variables["species_types"][:]]
        if len(tags) != len(types) or len(tags) != len(set(tags)):
            raise ValueError("Missing/ambiguous native species identity")
        ions = [(kind, tag) for kind, tag in zip(types, tags) if kind != "electron"]
        if not ions:
            raise ValueError(
                "No native ion species: cannot label an electron profile Ti"
            )
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
        from .worker import parse_output

        transport = parse_output(run_dir / "transport.nc")
        validate_evolution_result(result, transport, inputs)
        bulk_tag, trace = _native_metadata(run_dir / "transport.nc", inputs, transport)
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
