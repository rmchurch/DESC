"""Validate recorded transport progress independently of a zero process exit."""

import numpy as np


def validate_gpu_preflight(report):
    """Validate native arithmetic and the CPU metadata backend."""
    if (
        report["jax_backend"] not in {"gpu", "cuda"}
        or report["jax_array_platform"] not in {"gpu", "cuda"}
        or report["torch_visible_gpu_count"] != 1
        or report["torch_array_device"] != "cuda:0"
        or report["cpu_backend_available"] is not True
        or not np.isfinite(report["jax_sum"])
        or report["jax_sum"] != 3.0
        or not np.isfinite(report["torch_sum"])
        or report["torch_sum"] != 3.0
    ):
        raise RuntimeError(
            "GPU preflight requires one working CUDA device "
            "and DESC's CPU metadata backend"
        )


def evolution_outcome(step, time, requested_steps, requested_time, transport):
    """Classify the native OR time/step stopping limit; do not change that limit."""
    if (
        type(step) is not int
        or step < 0
        or type(requested_steps) is not int
        or requested_steps < 1
    ):
        raise ValueError("Invalid achieved/requested transport step count")
    if (
        not np.isfinite(time)
        or time < 0
        or not np.isfinite(requested_time)
        or requested_time <= 0
    ):
        raise ValueError("Invalid achieved/requested transport time")
    if transport["recorded_transport_step"] != step or not np.isclose(
        transport["final_profile_time"], time, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Engine progress disagrees with recorded transport output")
    progressed = step >= 1 and time > 0
    time_reached = time >= requested_time or np.isclose(
        time, requested_time, rtol=1e-10, atol=1e-12
    )
    steps_reached = step >= requested_steps
    horizon_reached = bool(time_reached or steps_reached)
    success = bool(progressed and horizon_reached)
    reason = (
        "no_transport_progress"
        if not progressed
        else (
            "requested_step_limit"
            if steps_reached
            else "requested_time_limit" if time_reached else "before_requested_horizon"
        )
    )
    return {
        "status": "evolved" if success else "stopped_early",
        "process_completed": True,
        "transport_step": step,
        "transport_time": float(time),
        "requested_steps": requested_steps,
        "requested_time": float(requested_time),
        "transport_output_valid": True,
        "transport_progressed": bool(progressed),
        "transport_horizon_reached": horizon_reached,
        "requested_evolution_completed": success,
        "premature_stop": not success,
        "transport_stop_reason": reason,
        "note": (
            "Requested evolution completion does not establish "
            "scientific convergence"
        ),
    }


def validate_evolution_result(result, transport, inputs):
    """Recompute worker claims using independently parsed output and input limits."""
    grid = inputs["grid"]
    if (
        len(transport["rho"]["values"]) != grid["N_radial"]
        or transport["flux_label"] != grid["flux_label"]
        or not np.isclose(
            transport["rho"]["values"][-1], grid["rho_edge"], rtol=1e-10, atol=1e-12
        )
    ):
        raise ValueError("Transport grid disagrees with the immutable input")
    requested = inputs.get("time", {})
    steps = requested.get("N_steps", 1000)  # verified native Time.py default
    end_time = requested.get("t_max", 1000)
    if requested.get("use_SI", False):
        end_time /= transport["normalizations"]["t_ref"]["values"]
    if result["requested_steps"] != steps or not np.isclose(
        result["requested_time"], end_time, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Worker requested limits disagree with the immutable input")
    expected = evolution_outcome(
        result["transport_step"], result["transport_time"], steps, end_time, transport
    )
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"Worker transport claim disagrees with recorded output: {key}"
            )
    if result["transport"] != transport:
        raise ValueError(
            "Worker parsed transport differs from independently read output"
        )
    return expected
