"""Public Ti consumer, identity, incomplete-result and cache contracts."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from desc.external.t3d import (
    T3DAdapter,
    T3DEvaluationError,
    atomic_json,
    evaluate_t3d,
    file_hash,
    parse_output,
    read_ion_temperature,
    t3d,
)

from .test_external_t3d import Eq
from .test_external_t3d import setup_adapter as _setup_adapter
from .test_external_t3d_output import native_result

pytestmark = pytest.mark.unit
setup_adapter = _setup_adapter


@pytest.fixture
def evaluator(setup_adapter, tmp_path, monkeypatch):
    """Evaluator."""
    config, _ = setup_adapter
    adapter = T3DAdapter(replace(config, mode="evolve"), tmp_path / "runs")
    calls, state = [], {"time": 10.0, "step": 1, "omit": None}

    def worker(command, **kwargs):
        calls.append(command)
        folder = Path(kwargs["cwd"])
        result = native_result(folder, time=state["time"], step=state["step"])
        with Dataset(folder / "transport.nc", "a") as d:
            species, tg = d.groups["species"], d.groups["time"]
            if state["omit"] != "species_types":
                v = species.createVariable("species_types", str, ("ns",))
                v[:] = np.asarray(["hydrogen", "electron"])
            if state["omit"] != "bulk_ion_tag":
                species.createVariable("bulk_ion_tag", str, ())[()] = "H"
            for name, values in (
                ("t_rms", [0.2, 0.01, 0.01]),
                ("t_iter_idx", [0, 1, 0]),
            ):
                if state["omit"] != name:
                    v = tg.createVariable(name, "f8", ("nt",))
                    v[:] = values
                    v.description = "Native recorded evidence"
            species.variables["T_H"][-1, :] = np.linspace(2, 3, 9)
            species.variables["T_e"][
                -1, :
            ] = 99  # catches accidental Te/first-species selection
        result.update(
            transport=parse_output(folder / "transport.nc"),
            ensemble_count=1,
            model_class="AI_GX_FluxModel",
            execution={"ai_gx_device": "cpu"},
        )
        atomic_json(folder / "result.json", result)
        if state.get("timeout") or state.get("timeout_on_call") == len(calls):
            raise subprocess.TimeoutExpired(command, 0.1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", worker)
    return adapter, calls, state


def refresh_artifacts(folder):
    """Rehash edited fixtures to test semantic checks beyond hash mismatch."""
    manifest = json.loads((folder / "manifest.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    result["transport"] = parse_output(folder / "transport.nc")
    atomic_json(folder / "result.json", result)
    manifest["result"] = result
    manifest["artifact_hashes"] = {
        name: file_hash(folder / name) for name in manifest["artifact_hashes"]
    }
    atomic_json(folder / "manifest.json", manifest)


def test_profile_values_coordinates_time_species_and_evidence(evaluator):
    """Profile values coordinates time species and evidence."""
    adapter, calls, _ = evaluator
    profile = evaluate_t3d(Eq(), adapter=adapter)
    np.testing.assert_array_equal(profile.temperature, np.linspace(2, 3, 9))
    np.testing.assert_allclose(profile.rho, np.linspace(0.7 / 17, 0.7, 9))
    assert profile.temperature_units == "keV" and profile.rho_units == "-"
    assert profile.flux_label == "torflux" and "sqrt" in profile.coordinate_definition
    assert not profile.includes_magnetic_axis and profile.rho[0] > 0
    assert (profile.species_type, profile.species_tag, profile.bulk_ion_tag) == (
        "hydrogen",
        "H",
        "H",
    )
    assert (
        profile.time,
        profile.time_units,
        profile.t_ref_seconds,
        profile.time_seconds,
    ) == (10.0, "[t_ref]", 0.3, 3.0)
    assert profile.status == "evolved" and profile.transport_step == 1
    assert profile.evidence["scientific_convergence_assessed"] is False
    assert profile.evidence["scientific_convergence"] is None
    assert profile.evidence["usable_ion_temperature_profile"] is True
    trace = profile.evidence["native_solver_trace"]
    assert (
        trace["t_rms"]["values"] == [0.2, 0.01, 0.01]
        and trace["t_rms"]["units"] is None
    )
    assert trace["t_iter_idx"]["values"] == [0, 1, 0]
    assert trace["time"]["values"] == [0, 0, 10]
    assert trace["time_input"]["newton_tolerance"] == 0.02
    assert profile.provenance["ensemble_count"] == 1
    assert profile.provenance["temperature_variable"] == "species/T_H"
    assert len(calls) == 1
    with pytest.raises(ValueError):
        profile.temperature[0] = 123
    with pytest.raises(ValueError):
        profile.rho[0] = 0
    serialized = profile.to_dict()
    json.dumps(serialized, allow_nan=False)
    serialized["temperature"][0] = -1
    serialized["evidence"]["native_solver_trace"]["t_rms"]["values"][0] = -1
    assert profile.temperature[0] == 2 and trace["t_rms"]["values"][0] == 0.2


def test_gx_like_batch_preserves_order_and_reuses_existing_cache(evaluator):
    """Gx like batch preserves order and reuses existing cache."""
    adapter, calls, _ = evaluator
    eqs = [Eq(2), Eq(1), Eq(2)]
    profiles = t3d(eqs, adapter=adapter)
    assert len(profiles) == 3 and len(calls) == 2
    assert profiles[0].provenance["run_id"] == profiles[2].provenance["run_id"]
    assert profiles[0].provenance["run_id"] != profiles[1].provenance["run_id"]
    assert t3d([], adapter=adapter) == []
    with pytest.raises(TypeError, match="sequence"):
        t3d(Eq(), adapter=adapter)
    with pytest.raises((AttributeError, TypeError)):
        t3d([Eq(), None], adapter=adapter)
    assert len(calls) == 2


def test_archive_reader_never_probes_or_executes_current_runtime(
    evaluator, monkeypatch
):
    """Archive reader never probes or executes current runtime."""
    adapter, _, _ = evaluator
    profile = evaluate_t3d(Eq(), adapter=adapter)
    adapter.runtime = {"new_installation": True}

    def forbidden(*args, **kwargs):
        raise AssertionError("Archive consumption must not launch/probe")

    monkeypatch.setattr(subprocess, "run", forbidden)
    reread = read_ion_temperature(profile.provenance["run_dir"])
    np.testing.assert_array_equal(reread.temperature, profile.temperature)
    assert reread.provenance["runtime"] == {}  # original archived identity


def test_batch_failure_keeps_prior_success_without_retry_or_partial_return(evaluator):
    """Batch failure keeps prior success without retry or partial return."""
    adapter, calls, state = evaluator
    state["timeout_on_call"] = 2
    for _ in range(2):
        with pytest.raises(T3DEvaluationError) as caught:
            t3d([Eq(1), Eq(2), Eq(3)], adapter=adapter)
        assert caught.value.status == "failed"
        assert caught.value.evidence["usable_ion_temperature_profile"] is False
    assert len(calls) == 2
    assert len(t3d([Eq(1)], adapter=adapter)) == 1
    assert len(calls) == 2
    assert len(list(adapter.output_dir.glob("*/manifest.json"))) == 2


@pytest.mark.parametrize(
    "species,message",
    [
        ([{"type": "electron"}], "No ion species"),
        (
            [{"type": "hydrogen"}, {"type": "deuterium"}, {"type": "electron"}],
            "Multiple ion species",
        ),
        ([{"type": "deuterium"}, {"type": "electron"}], "requires hydrogen"),
        ([{"type": "hydrogen", "tag": "i"}, {"type": "electron"}], "requires hydrogen"),
    ],
)
def test_ambiguous_input_species_rejected_before_execution(evaluator, species, message):
    """Ambiguous input species rejected before execution."""
    adapter, calls, _ = evaluator
    adapter.inputs["species"] = species
    with pytest.raises(ValueError, match=message):
        evaluate_t3d(Eq(), adapter=adapter)
    assert calls == []


@pytest.mark.parametrize(
    "native_types,message",
    [
        (["electron", "electron"], "No native ion species"),
        (["hydrogen", "deuterium"], "Multiple native ion species"),
        (["electron", "hydrogen"], "identities disagree"),
    ],
)
def test_native_species_metadata_cannot_silently_label_wrong_ti(
    evaluator, native_types, message
):
    """Native species metadata cannot silently label wrong ti."""
    adapter, _, _ = evaluator
    profile = evaluate_t3d(Eq(), adapter=adapter)
    folder = Path(profile.provenance["run_dir"])
    with Dataset(folder / "transport.nc", "a") as d:
        d.groups["species"].variables["species_types"][:] = np.asarray(native_types)
    refresh_artifacts(folder)
    with pytest.raises(T3DEvaluationError, match=message):
        read_ion_temperature(folder)


def test_wrong_bulk_ion_rejected(evaluator):
    """Wrong bulk ion rejected."""
    adapter, _, _ = evaluator
    folder = Path(evaluate_t3d(Eq(), adapter=adapter).provenance["run_dir"])
    with Dataset(folder / "transport.nc", "a") as d:
        d.groups["species"].variables["bulk_ion_tag"][()] = "e"
    refresh_artifacts(folder)
    with pytest.raises(T3DEvaluationError, match="bulk ion"):
        read_ion_temperature(folder)


@pytest.mark.parametrize(
    "omit", ["species_types", "bulk_ion_tag", "t_rms", "t_iter_idx"]
)
def test_missing_native_identity_or_solver_evidence_not_returned(evaluator, omit):
    """Missing native identity or solver evidence not returned."""
    adapter, calls, state = evaluator
    state["omit"] = omit
    with pytest.raises(T3DEvaluationError, match=omit):
        evaluate_t3d(Eq(), adapter=adapter)
    assert len(calls) == 1
    with pytest.raises(T3DEvaluationError):
        evaluate_t3d(Eq(), adapter=adapter)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "state_name", ["failed", "stopped_early", "running", "initialized"]
)
def test_non_evolved_archive_has_status_evidence_and_no_profile(evaluator, state_name):
    """Non evolved archive has status evidence and no profile."""
    adapter, _, _ = evaluator
    folder = Path(evaluate_t3d(Eq(), adapter=adapter).provenance["run_dir"])
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["status"] = state_name
    manifest["requested_evolution_completed"] = False
    manifest["result"]["requested_evolution_completed"] = False
    atomic_json(folder / "manifest.json", manifest)
    with pytest.raises(T3DEvaluationError, match="No usable Ti") as caught:
        read_ion_temperature(folder)
    assert caught.value.status == state_name
    assert caught.value.evidence["requested_evolution_completed"] is False


def test_early_stop_error_retains_native_progress_and_never_retries(evaluator):
    """Early stop error retains native progress and never retries."""
    adapter, calls, state = evaluator
    state["time"] = 0.1
    for _ in range(2):
        with pytest.raises(T3DEvaluationError) as caught:
            evaluate_t3d(Eq(), adapter=adapter)
        assert caught.value.status == "stopped_early"
        assert caught.value.failure_kind == "premature_stop"
        assert caught.value.evidence["transport_step"] == 1
        assert caught.value.evidence["transport_time"] == 0.1
        assert caught.value.evidence["requested_evolution_completed"] is False
    assert len(calls) == 1


def test_native_timeout_cannot_return_partial_profile(evaluator):
    """Native timeout cannot return partial profile."""
    adapter, calls, state = evaluator
    state["timeout"] = True
    for _ in range(2):
        with pytest.raises(T3DEvaluationError) as caught:
            evaluate_t3d(Eq(), adapter=adapter)
        assert (
            caught.value.status == "failed" and caught.value.failure_kind == "timeout"
        )
    assert len(calls) == 1


@pytest.mark.parametrize("defect", ["missing", "changed", "unhashed"])
def test_missing_or_tampered_output_cannot_return_profile(evaluator, defect):
    """Missing or tampered output cannot return profile."""
    adapter, _, _ = evaluator
    folder = Path(evaluate_t3d(Eq(), adapter=adapter).provenance["run_dir"])
    if defect == "missing":
        (folder / "transport.nc").unlink()
    elif defect == "changed":
        with Dataset(folder / "transport.nc", "a") as d:
            d.groups["species"].variables["T_H"][-1, 0] = 987
    else:
        manifest = json.loads((folder / "manifest.json").read_text())
        del manifest["artifact_hashes"]["transport.nc"]
        atomic_json(folder / "manifest.json", manifest)
    with pytest.raises(T3DEvaluationError):
        read_ion_temperature(folder)


def test_initialization_and_absent_run_not_mislabeled_as_ti(evaluator, tmp_path):
    """Initialization and absent run not mislabeled as ti."""
    adapter, calls, _ = evaluator
    adapter.config = replace(adapter.config, mode="initialize")
    with pytest.raises(ValueError, match="initialization"):
        evaluate_t3d(Eq(), adapter=adapter)
    assert calls == []
    with pytest.raises(T3DEvaluationError) as caught:
        read_ion_temperature(tmp_path / "absent")
    assert caught.value.status == "unavailable"


@pytest.mark.parametrize("manifest", [[], None, {"status": "failed", "result": None}])
def test_malformed_manifest_still_raises_structured_consumer_error(tmp_path, manifest):
    """Malformed manifest still raises structured consumer error."""
    atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(T3DEvaluationError) as caught:
        read_ion_temperature(tmp_path)
    assert caught.value.evidence["usable_ion_temperature_profile"] is False


@pytest.mark.parametrize(
    "name,value",
    [("t_rms", np.nan), ("t_rms", -1), ("t_iter_idx", 0.5), ("t_iter_idx", -1)],
)
def test_invalid_solver_evidence_rejected(evaluator, name, value):
    """Invalid solver evidence rejected."""
    adapter, _, _ = evaluator
    folder = Path(evaluate_t3d(Eq(), adapter=adapter).provenance["run_dir"])
    with Dataset(folder / "transport.nc", "a") as d:
        d.groups["time"].variables[name][0] = value
    refresh_artifacts(folder)
    with pytest.raises(T3DEvaluationError):
        read_ion_temperature(folder)
