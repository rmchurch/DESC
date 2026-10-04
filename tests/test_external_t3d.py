"""Fast persistence, failure and input-contract checks (no native transport)."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from desc.external.t3d import (
    T3DAdapter,
    T3DConfig,
    _set_key,
    equilibrium_identity,
    load_toml,
)

pytestmark = pytest.mark.unit


class Eq:
    """Eq."""

    L = M = N = L_grid = M_grid = N_grid = NFP = 1
    sym = True

    def __init__(self, value=1):
        self.params_dict = {"R_lmn": np.array([float(value)]), "Psi": np.array(1.0)}

    def save(self, path):
        """Write a stand-in equilibrium handoff for persistence tests."""
        Path(path).write_text(str(self.params_dict))


@pytest.fixture
def setup_adapter(tmp_path, monkeypatch):
    """Setup adapter."""
    base = Path(__file__).parent / "inputs" / "t3d_ai_gx"
    text = (base / "transport.in").read_text()
    model = tmp_path / "models.v2"
    model.mkdir()
    (model / "model.pth").write_bytes(b"existing-weight")
    (tmp_path / "results.csv").write_text("existing-csv")
    (tmp_path / "gx_template.in").write_bytes((base / "gx_template.in").read_bytes())
    text = _set_key(text, "[[model]]", "model_dir", str(model))
    text = _set_key(text, "[[model]]", "csv_path", str(tmp_path / "results.csv"))
    template = tmp_path / "template.in"
    template.write_text(text)
    calls = []

    def run(command, **kwargs):
        if "--probe" in command:
            return subprocess.CompletedProcess(command, 0, "{}", "")
        calls.append(command)
        Path(kwargs["cwd"], "result.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "status": "initialized",
                    "runtime": {},
                    "mode": "initialize",
                    "process_completed": True,
                    "requested_evolution_completed": False,
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    config = T3DConfig(template=str(template), mode="initialize")
    return config, calls


def test_exact_physics_handoff_and_content_cache(setup_adapter, tmp_path):
    """Exact physics handoff and content cache."""
    config, calls = setup_adapter
    adapter = T3DAdapter(config, tmp_path / "runs")
    result = adapter.evaluate(Eq())
    run_dir = adapter.output_dir / result["run_id"]
    rendered = load_toml((run_dir / "transport.in").read_text())
    for section in ("species", "grid", "time", "physics", "log"):
        assert rendered[section] == adapter.inputs[section]
    assert rendered["geometry"]["geo_file"] == str(run_dir / "equilibrium.h5")
    assert adapter.evaluate(Eq())["run_id"] == result["run_id"]
    assert len(calls) == 1
    assert (
        T3DAdapter(config, adapter.output_dir).evaluate(Eq())["run_id"]
        == result["run_id"]
    )
    assert len(calls) == 1
    adapter.evaluate(Eq(2))
    assert len(calls) == 2


def test_missing_legacy_csv_v2_allowed(setup_adapter, tmp_path):
    """Missing legacy csv v2 allowed."""
    config, _ = setup_adapter
    (tmp_path / "results.csv").unlink()
    adapter = T3DAdapter(config, tmp_path / "runs")
    assert adapter.assets["csv"] is None


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing-result"])
def test_failures_never_retry(setup_adapter, tmp_path, monkeypatch, failure):
    """Failures never retry."""
    config, _ = setup_adapter
    adapter = T3DAdapter(config, tmp_path / "runs")
    attempts = []

    def run(command, **kwargs):
        attempts.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 3 if failure == "exit" else 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired, FileNotFoundError)):
        adapter.evaluate(Eq())
    manifest = next(adapter.output_dir.glob("*/manifest.json"))
    assert json.loads(manifest.read_text())["status"] == "failed"
    with pytest.raises(RuntimeError, match="No automatic retry"):
        adapter.evaluate(Eq())
    assert len(attempts) == 1


def test_interrupted_claim_and_cache_integrity(setup_adapter, tmp_path):
    """Interrupted claim and cache integrity."""
    config, calls = setup_adapter
    adapter = T3DAdapter(config, tmp_path / "runs")
    result = adapter.evaluate(Eq())
    folder = adapter.output_dir / result["run_id"]
    (folder / "equilibrium.h5").write_text("changed")
    with pytest.raises(RuntimeError, match="Cached artifact changed"):
        adapter.evaluate(Eq())
    result["status"] = "running"
    (folder / "manifest.json").write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="No automatic retry"):
        adapter.evaluate(Eq())
    assert len(calls) == 1


def test_model_changes_refused_and_explicit_retry_identity(setup_adapter, tmp_path):
    """Model changes refused and explicit retry identity."""
    config, _ = setup_adapter
    adapter = T3DAdapter(config, tmp_path / "runs")
    retried = T3DAdapter(
        replace(config, retry_token="manual-recovery-1"), adapter.output_dir
    )
    assert adapter.identity != retried.identity
    (adapter.model_dir / "model.pth").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed during"):
        adapter.evaluate(Eq())
    new_adapter = T3DAdapter(config, adapter.output_dir)
    assert adapter.identity != new_adapter.identity


def test_physics_rejections(setup_adapter, tmp_path):
    """Physics rejections."""
    config, _ = setup_adapter
    text = Path(config.template).read_text().replace('model = "AI_GX"', 'model = "GX"')
    Path(config.template).write_text(text)
    with pytest.raises(ValueError, match="Exactly one AI_GX"):
        T3DAdapter(config, tmp_path / "runs")


def test_equilibrium_identity_includes_resolution_and_fields():
    """Equilibrium identity includes resolution and fields."""
    eq = Eq()
    original = equilibrium_identity(eq)
    eq.N = 2
    assert equilibrium_identity(eq) != original
    eq.sym = np.bool_(True)
    eq.NFP = np.int64(1)
    assert equilibrium_identity(eq)
    eq.N = 1
    eq.params_dict["Psi"] = np.array(2.0)
    assert equilibrium_identity(eq) != original
