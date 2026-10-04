"""Small subprocess entry point for the caller's native T3D Python environment."""

from __future__ import annotations

import argparse
import json

from .t3d import (
    atomic_json,
    evolution_outcome,
    gpu_preflight,
    parse_output,
    runtime_identity,
)


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
