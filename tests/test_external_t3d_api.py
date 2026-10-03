"""Configuration and dependency boundaries of the public transport API."""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from desc.external.t3d import T3DConfig

pytestmark = pytest.mark.unit


def test_physics_template_is_explicit_and_hook_policy_stays_outside_desc():
    """Require a caller's template without importing application hook policy."""
    with pytest.raises(TypeError, match="template"):
        T3DConfig()
    config = T3DConfig(template="user-supplied.in")
    assert config.template == "user-supplied.in"
    names = {field.name for field in dataclasses.fields(config)}
    assert "every_n" not in names and "failure_policy" not in names


def test_public_import_is_independent_of_optional_runtime_and_device_selection():
    """Import the reusable API without external packages, hooks or backend setup."""
    script = """
import importlib.abc
import os
import sys

class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"t3d", "torch", "netCDF4", "tomli", "jax"}:
            raise AssertionError("Optional runtime imported: " + fullname)

before = {key: os.environ.get(key) for key in
          ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS")}
sys.meta_path.insert(0, RejectOptional())
from desc.external.t3d import T3DConfig, t3d
assert callable(t3d)
assert T3DConfig(template="explicit.in").template == "explicit.in"
assert "desc.backend" not in sys.modules
assert "desc.optimize" not in sys.modules
assert "t3d_checkpoints" not in sys.modules
assert before == {key: os.environ.get(key) for key in before}
"""
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
