"""Bounded CPU initialization/inference and archived-profile check, never evolve.

Use an external T3D-enabled interpreter and explicit template/assets. The caller
must set CPU visibility/thread limits and enforce an overall wall-time bound.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    """Check native execution of the relocated wrapper without transport steps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--equilibrium", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--t3d-source", required=True)
    parser.add_argument("--archived-run", required=True)
    args = parser.parse_args()
    from desc import set_device

    set_device("cpu")
    from desc.external._t3d.adapter import atomic_json
    from desc.external.t3d import T3DAdapter, T3DConfig, read_ion_temperature
    from desc.io import load

    eq = load(args.equilibrium)
    if hasattr(eq, "__len__"):
        eq = eq[-1]
    config = T3DConfig(
        template=args.template,
        source_root=args.t3d_source,
        mode="initialize",
        timeout_seconds=120,
        probe_timeout_seconds=60,
    )
    adapter = T3DAdapter(config, args.output)
    first = adapter.evaluate(eq)
    second = adapter.evaluate(eq)
    assert first["run_id"] == second["run_id"]
    folder = adapter.output_dir / first["run_id"]
    restored = load(str(folder / "equilibrium.h5"))
    for key in eq.params_dict:
        np.testing.assert_array_equal(eq.params_dict[key], restored.params_dict[key])
    from t3d.ai.inference_engine import AI_GX_InferenceEngine, create_gx_geometry

    engine = AI_GX_InferenceEngine(str(adapter.model_dir), str(adapter.csv_path))
    rho = 0.535
    tensor = create_gx_geometry(str(folder / "equilibrium.h5"), rho**2, 0.0)
    mean, std = engine.inference(feature_tensor=tensor, tprims=3.0, fprims=0.9)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
    profile = read_ion_temperature(args.archived_run)
    from netCDF4 import Dataset

    with Dataset(Path(args.archived_run) / "transport.nc") as dataset:
        np.testing.assert_array_equal(
            profile.temperature, dataset.groups["species"].variables["T_H"][-1]
        )
        np.testing.assert_array_equal(
            profile.rho, dataset.groups["grid"].variables["rho"][:]
        )
    assert profile.temperature_units == "keV"
    assert profile.species_type == "hydrogen" and profile.species_tag == "H"
    assert profile.evidence["scientific_convergence_assessed"] is False
    report = {
        "initialization_run_id": first["run_id"],
        "cached_second_call": True,
        "equilibrium_roundtrip_exact": True,
        "ensemble_count": len(engine.models),
        "feature_tensor_shape": list(tensor.shape),
        "rho": rho,
        "s": rho**2,
        "mean_native_ensemble_output": np.asarray(mean).tolist(),
        "std_native_ensemble_output": np.asarray(std).tolist(),
        "transport_timesteps_executed": 0,
        "runtime": adapter.runtime,
        "archived_profile_exact_match": True,
        "archived_profile": profile.to_dict(),
    }
    atomic_json(Path(args.output) / "native_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
