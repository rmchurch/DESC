"""Native T3D subprocess entry point and unit-preserving NetCDF parser."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path

import numpy as np

from .adapter import atomic_json, digest, file_hash
from .outcomes import evolution_outcome, validate_gpu_preflight


def runtime_identity():
    """Record actual external source and runtime library versions."""
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
            {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        "libraries": {
            name: metadata.version(name)
            for name in ("numpy", "scipy", "torch", "jax", "jaxlib", "netCDF4")
        },
    }


def parse_output(path):
    """Parse native profiles and separate preceding flux snapshots."""
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
        return {
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


def main():
    """Run the explicitly configured native worker command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--mode", choices=("initialize", "evolve"), default="initialize"
    )
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    # The installed DESC backend defaults to set_device('cpu'), which also clears
    # CUDA_VISIBLE_DEVICES. Select GPU before runtime_identity imports that backend.
    if args.require_gpu:
        import desc

        desc.set_device("gpu")
    runtime = runtime_identity()
    if args.probe:
        print(json.dumps(runtime, sort_keys=True))
        return
    preflight = gpu_preflight() if args.require_gpu else None
    if preflight is not None:
        atomic_json("gpu_preflight.json", preflight)
    from t3d.Logbook import log
    from t3d.trinity_lib import TrinityEngine

    log.set_handlers(term_stream=True, file_stream=True, file_handler="transport.out")
    engine = TrinityEngine(args.input)
    model = next(iter(engine.flux_models.get_models_list()))
    if model.__class__.__name__ != "AI_GX_FluxModel" or not model.ai_engine.models:
        raise RuntimeError("Native AI_GX ensemble was not loaded")
    import jax
    import torch

    execution = {
        "ai_gx_device": model.ai_engine.device,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "torch_visible_gpu_count": torch.cuda.device_count(),
        "model_parameter_devices": sorted(
            {str(next(net.parameters()).device) for net in model.ai_engine.models}
        ),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    atomic_json("backend.json", execution)
    if args.require_gpu and (
        execution["ai_gx_device"] != "cuda"
        or execution["jax_backend"] not in {"gpu", "cuda"}
        or execution["torch_visible_gpu_count"] != 1
        or execution["model_parameter_devices"] != ["cuda:0"]
    ):
        raise RuntimeError(
            "Required single-GPU backend was not active; no CPU fallback"
        )
    result = {
        "schema_version": 2,
        "status": "initialized",
        "runtime": runtime,
        "process_completed": True,
        "requested_evolution_completed": False,
        "ensemble_count": len(model.ai_engine.models),
        "execution": execution,
        "gpu_preflight": preflight,
        "model_class": model.__class__.__name__,
        "mode": args.mode,
    }
    if args.mode == "evolve":
        engine.evolve_profiles()
        transport = parse_output("transport.nc")
        result.update(transport=transport)
        result.update(
            evolution_outcome(
                int(engine.time.step_idx),
                float(engine.time.time),
                int(engine.time.N_steps),
                float(engine.time.t_max),
                transport,
            )
        )
    else:
        result.update(
            density_evolve=list(engine.species.n_evolve_keys),
            temperature_evolve=list(engine.species.T_evolve_keys),
            N_radial=int(engine.grid.N_radial),
            rho_edge=float(engine.grid.rho[-1]),
            equilibrium_file=engine.geometry.geo_file,
            note=(
                "Native engine/models/geometry initialized; "
                "no flux inference or transport timestep"
            ),
        )
    atomic_json("result.json", result)
    log.finalize()


if __name__ == "__main__":
    main()
