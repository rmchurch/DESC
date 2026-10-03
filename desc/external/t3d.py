"""Evaluate native T3D AI_GX transport on explicitly configured DESC equilibria.

The public callable mirrors ``desc.external.gx.gx`` equilibrium-list input, but
returns one ion-temperature radial profile per equilibrium. No scalar objective,
derivative, optimizer, scientific convergence policy or pressure feedback is
selected. T3D, its template and trained model assets are supplied by the caller.
Optional external dependencies are loaded only when configured or evaluated.
"""

from ._t3d.adapter import T3DAdapter, T3DConfig
from ._t3d.evaluation import (
    IonTemperatureProfile,
    T3DEvaluationError,
    evaluate_t3d,
    read_ion_temperature,
    t3d,
)

__all__ = [
    "T3DAdapter",
    "T3DConfig",
    "IonTemperatureProfile",
    "T3DEvaluationError",
    "evaluate_t3d",
    "read_ion_temperature",
    "t3d",
]
