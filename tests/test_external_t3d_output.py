"""Progress/partial-output/timeout contracts and bounded scheduler smoke checks."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from desc.external._t3d.adapter import atomic_json
from desc.external._t3d.outcomes import (
    evolution_outcome,
    validate_evolution_result,
    validate_gpu_preflight,
)
from desc.external._t3d.worker import parse_output
from desc.external.t3d import T3DAdapter

from .test_external_t3d import Eq
from .test_external_t3d import setup_adapter as _setup_adapter

pytestmark = pytest.mark.unit
setup_adapter = _setup_adapter


def write_output(path, step=1, time=0.1, defect=None):
    """Write output."""
    with Dataset(path, "w") as d:
        for name, size in (("nt", 3), ("nr", 9), ("nm", 8), ("ns", 2)):
            d.createDimension(name, size)

        def var(group, name, values, dims, units=None, dtype="f8"):
            v = group.createVariable(name, dtype, dims)
            v[:] = np.asarray(values)
            if units is not None:
                v.units = units
            return v

        tg = d.createGroup("time")
        var(tg, "t", [0, 0, time], ("nt",), "[t_ref]")
        var(tg, "t_step_idx", [0, 0, step], ("nt",), dtype="i4")
        grid = d.createGroup("grid")
        rho = np.linspace(0.7 / 17, 0.7, 9)
        var(grid, "rho", rho, ("nr",), "-")
        var(grid, "midpoints", (rho[:-1] + rho[1:]) / 2, ("nm",), "-")
        var(grid, "flux_label", "torflux", (), dtype=str)
        norms = d.createGroup("norms")
        var(norms, "t_ref", 0.3, (), "s")
        species = d.createGroup("species")
        var(species, "species_tags", ["H", "e"], ("ns",), dtype=str)
        for tag in ("H", "e"):
            for key, units in {
                "T": "keV",
                "n": "10^20 m^-3",
                "p": "10^20 m^-3 keV",
                "qflux": "[GB]",
                "Q_MW": "MW",
                "aLT": "-",
                "aLn": "-",
            }.items():
                if defect == "missing" and key == "Q_MW" and tag == "e":
                    continue
                dims = ("nt", "nr") if key in ("T", "n", "p") else ("nt", "nm")
                if defect == "shape" and key == "T" and tag == "H":
                    dims = ("nt", "nm")
                v = var(
                    species,
                    f"{key}_{tag}",
                    np.ones((3, 9 if dims[1] == "nr" else 8)),
                    dims,
                    units,
                )
                if key == "T" and tag == "H":
                    if defect == "nan":
                        v[0, 0] = np.nan  # check even nonfinal rows
                    elif defect == "masked":
                        v[1, 0] = np.ma.masked
                    elif defect == "units":
                        v.units = "eV"


@pytest.mark.parametrize(
    "step,time,steps,end,success,reason",
    [
        (0, 0.0, 1, 0.1, False, "no_transport_progress"),
        (1, 0.1, 1000, 10.0, False, "before_requested_horizon"),
        (1, 0.05, 1, 0.1, True, "requested_step_limit"),
        (3, 0.1, 1000, 0.1, True, "requested_time_limit"),
    ],
)
def test_normal_exit_is_not_requested_evolution(
    step, time, steps, end, success, reason
):
    """Normal exit is not requested evolution."""
    result = evolution_outcome(
        step,
        time,
        steps,
        end,
        {"recorded_transport_step": step, "final_profile_time": time},
    )
    assert result["process_completed"]
    assert result["requested_evolution_completed"] is success
    assert result["premature_stop"] is not success
    assert result["status"] == ("evolved" if success else "stopped_early")
    assert result["transport_stop_reason"] == reason


def test_progress_requires_matching_recorded_output():
    """Progress requires matching recorded output."""
    for step, time in ((1, np.nan), (1, -1.0), (2, 0.1)):
        with pytest.raises(ValueError):
            evolution_outcome(
                step,
                time,
                1,
                0.1,
                {"recorded_transport_step": 1, "final_profile_time": 0.1},
            )


@pytest.mark.parametrize("defect", ["missing", "shape", "nan", "masked", "units"])
def test_partial_or_invalid_netcdf_rejected(tmp_path, defect):
    """Partial or invalid netcdf rejected."""
    path = tmp_path / "transport.nc"
    write_output(path, defect=defect)
    with pytest.raises((ValueError, KeyError)):
        parse_output(path)


def native_result(folder, step=1, time=0.1, steps=1000, end=10.0, defect=None):
    """Native result."""
    write_output(folder / "transport.nc", step, time, defect)
    if defect is not None:
        # A normal process exit with an incomplete worker report must still fail.
        atomic_json(
            folder / "result.json",
            {
                "schema_version": 2,
                "runtime": {},
                "mode": "evolve",
                "process_completed": True,
                "status": "evolved",
            },
        )
        return None
    transport = parse_output(folder / "transport.nc")
    result = {
        "schema_version": 2,
        "runtime": {},
        "mode": "evolve",
        "transport": transport,
    }
    result.update(evolution_outcome(step, time, steps, end, transport))
    atomic_json(folder / "result.json", result)
    return result


@pytest.mark.parametrize("step,time", [(0, 0.0), (1, 0.1)])
def test_early_stop_recorded_and_never_retried(
    setup_adapter, tmp_path, monkeypatch, step, time
):
    """Early stop recorded and never retried."""
    config, _ = setup_adapter
    adapter = T3DAdapter(replace(config, mode="evolve"), tmp_path / "runs")
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        native_result(Path(kwargs["cwd"]), step=step, time=time)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", worker)
    with pytest.raises(RuntimeError, match="stopped before requested evolution"):
        adapter.evaluate(Eq())
    manifest = json.loads(next(adapter.output_dir.glob("*/manifest.json")).read_text())
    assert manifest["status"] == "stopped_early"
    assert (
        manifest["process_completed"] and not manifest["requested_evolution_completed"]
    )
    assert manifest["result"]["transport_step"] == step
    assert manifest["failure_kind"] == "premature_stop"
    assert "transport.nc" in manifest["artifact_hashes"]
    with pytest.raises(RuntimeError, match="No automatic retry"):
        adapter.evaluate(Eq())
    assert len(calls) == 1


def test_timeout_rejects_even_a_partial_success_report(
    setup_adapter, tmp_path, monkeypatch
):
    """Timeout rejects even a partial success report."""
    config, _ = setup_adapter
    adapter = T3DAdapter(replace(config, mode="evolve"), tmp_path / "runs")

    def worker(command, **kwargs):
        native_result(Path(kwargs["cwd"]), step=1, time=10.0)
        raise subprocess.TimeoutExpired(command, 0.1)

    monkeypatch.setattr(subprocess, "run", worker)
    with pytest.raises(subprocess.TimeoutExpired):
        adapter.evaluate(Eq())
    manifest = json.loads(next(adapter.output_dir.glob("*/manifest.json")).read_text())
    assert manifest["status"] == "failed" and manifest["failure_kind"] == "timeout"
    assert (
        not manifest["process_completed"]
        and not manifest["requested_evolution_completed"]
    )
    assert "result" not in manifest
    with pytest.raises(RuntimeError, match="No automatic retry"):
        adapter.evaluate(Eq())


def test_zero_exit_partial_output_is_not_transport_validation(
    setup_adapter, tmp_path, monkeypatch
):
    """Zero exit partial output is not transport validation."""
    config, _ = setup_adapter
    adapter = T3DAdapter(replace(config, mode="evolve"), tmp_path / "runs")

    def worker(command, **kwargs):
        native_result(Path(kwargs["cwd"]), defect="missing")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", worker)
    with pytest.raises(KeyError):
        adapter.evaluate(Eq())
    manifest = json.loads(next(adapter.output_dir.glob("*/manifest.json")).read_text())
    assert manifest["process_completed"] and manifest["status"] == "failed"
    assert not manifest["requested_evolution_completed"]


def test_successful_evolution_validated_and_cached(
    setup_adapter, tmp_path, monkeypatch
):
    """Successful evolution validated and cached."""
    config, _ = setup_adapter
    adapter = T3DAdapter(replace(config, mode="evolve"), tmp_path / "runs")
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        native_result(Path(kwargs["cwd"]), time=10.0)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", worker)
    manifest = adapter.evaluate(Eq())
    assert manifest["status"] == "evolved" and manifest["requested_evolution_completed"]
    assert adapter.evaluate(Eq())["run_id"] == manifest["run_id"]
    assert len(calls) == 1


def test_worker_claims_checked_against_template_and_netcdf(tmp_path):
    """Worker claims checked against template and netcdf."""
    result = native_result(tmp_path, time=10.0)
    transport = parse_output(tmp_path / "transport.nc")
    inputs = {
        "time": {"t_max": 10.0},
        "grid": {"N_radial": 9, "rho_edge": 0.7, "flux_label": "torflux"},
    }
    validate_evolution_result(result, transport, inputs)
    for key, value in (
        ("requested_time", 1.0),
        ("requested_evolution_completed", False),
        ("transport_step", 0),
        ("status", "completed"),
    ):
        with pytest.raises(ValueError):
            validate_evolution_result(dict(result, **{key: value}), transport, inputs)


GPU_PREFLIGHT = {
    "jax_backend": "cuda",
    "jax_array_platform": "cuda",
    "torch_visible_gpu_count": 1,
    "torch_array_device": "cuda:0",
    "cpu_backend_available": True,
    "jax_sum": 3.0,
    "torch_sum": 3.0,
}


@pytest.mark.parametrize(
    "key,value",
    [
        ("jax_backend", "cpu"),
        ("jax_array_platform", "cpu"),
        ("torch_visible_gpu_count", 4),
        ("torch_array_device", "cpu"),
        ("cpu_backend_available", False),
        ("jax_sum", 2.0),
        ("torch_sum", np.nan),
    ],
)
def test_gpu_preflight_requires_real_device_arithmetic_and_cpu_metadata(key, value):
    """Gpu preflight requires real device arithmetic and cpu metadata."""
    validate_gpu_preflight(GPU_PREFLIGHT)
    with pytest.raises(RuntimeError, match="GPU preflight"):
        validate_gpu_preflight(dict(GPU_PREFLIGHT, **{key: value}))


def test_worker_selects_desc_gpu_before_runtime_backend_import(monkeypatch):
    """Worker selects desc gpu before runtime backend import."""
    import sys
    from types import SimpleNamespace

    from desc.external._t3d import worker

    selected = []
    monkeypatch.setitem(
        sys.modules, "desc", SimpleNamespace(set_device=selected.append)
    )
    monkeypatch.setattr(sys, "argv", ["worker", "--require-gpu", "transport.in"])

    def at_runtime_import():
        assert selected == ["gpu"]
        raise RuntimeError("checked backend import order")

    monkeypatch.setattr(worker, "runtime_identity", at_runtime_import)
    with pytest.raises(RuntimeError, match="checked backend import order"):
        worker.main()
