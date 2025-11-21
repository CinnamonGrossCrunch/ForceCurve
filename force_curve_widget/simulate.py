"""Simulation utilities for configurable cable-tension force curves."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Simple-mode configuration -------------------------------------------------

_ENV_SIMPLE = os.getenv("FORCE_CURVE_SIMPLE_MODE")
if _ENV_SIMPLE is None:
    SIMPLE_MODE = True
else:
    SIMPLE_MODE = _ENV_SIMPLE.lower() not in {"0", "false", "no"}

DEFAULT_FS = 500
DEFAULT_WEIGHT_LBS = 80.0
DEFAULT_CONCENTRIC_SPLIT = (0.4, 0.4, 0.2)
DEFAULT_ECCENTRIC_SPLIT = (0.4, 0.6)
DEFAULT_MULTIPLIERS = {
    "k_spike": 1.5,
    "k_plateau": 1.05,
    "k_low_ecc": 0.25,
    "k_brake": 1.25,
}


def _weight_to_kN(weight_value: float, unit: str) -> float:
    unit = unit.lower()
    if unit.startswith("kg"):
        mass_kg = weight_value
    elif unit.startswith("lb"):
        mass_kg = weight_value * 0.45359237
    else:
        raise ValueError("weight unit must be 'lbs' or 'kg'")
    force_newtons = mass_kg * 9.80665
    return force_newtons / 1000.0


def _split_durations(total: float, ratios: Sequence[float]) -> Tuple[float, ...]:
    total = max(total, 0.0)
    ratios = list(ratios)
    baseline = sum(ratios)
    if baseline <= 0:
        raise ValueError("ratios must sum to a positive value")
    normalized = [r / baseline for r in ratios]
    return tuple(total * r for r in normalized)


def _velocity_increments(count: int, distance: float, a: float, b: float) -> np.ndarray:
    """Return distance increments shaped by a beta-like velocity profile."""
    if count <= 0 or distance <= 0:
        return np.zeros(count)
    centers = (np.arange(count, dtype=float) + 0.5) / count
    centers = np.clip(centers, 1e-3, 1 - 1e-3)
    profile = np.power(centers, a - 1.0) * np.power(1.0 - centers, b - 1.0)
    profile_sum = profile.sum()
    if profile_sum <= 0:
        return np.full(count, distance / count)
    return profile / profile_sum * distance


def _cumulative_positions(increments: np.ndarray, *, start: float, ascending: bool) -> np.ndarray:
    if increments.size == 0:
        return np.zeros(0, dtype=float)
    cumulative = np.cumsum(increments)
    shifted = np.concatenate(([0.0], cumulative[:-1]))
    if ascending:
        positions = start + shifted
    else:
        positions = start - shifted
    return positions


@dataclass
class SimulationParams:
    """Parameters controlling the concentric/eccentric profile of a single rep."""

    W_kN: float = 0.36
    t_c1: float = 0.30
    t_c2: float = 0.40
    t_c3: float = 0.20
    t_top: float = 0.20
    t_e1: float = 0.40
    t_e2: float = 0.60
    k_spike: float = 1.5
    k_plateau: float = 1.05
    k_low_ecc: float = 0.25
    k_brake: float = 1.25
    fs: int = 500
    stroke_m: float = 0.5

    def total_time(self) -> float:
        return self.t_c1 + self.t_c2 + self.t_c3 + self.t_top + self.t_e1 + self.t_e2


@dataclass
class NoiseSettings:
    """Sensor effects to optionally apply to the simulated force trace."""

    noise_enabled: bool = False
    noise_std: float = 0.0  # kN
    drift_per_second: float = 0.0  # kN / s
    quantization_enabled: bool = False
    quantization_bits: int = 16
    quantization_range_kN: float = 2.0  # full scale range in kN

    def has_noise(self) -> bool:
        return self.noise_enabled and self.noise_std > 0

    def has_drift(self) -> bool:
        return abs(self.drift_per_second) > 0

    def has_quantization(self) -> bool:
        return self.quantization_enabled and self.quantization_bits > 0


def _mask(t: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    return (t >= t_start) & (t < t_end)


def simulate_trace(params: SimulationParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure mathematical model with no sensor effects."""

    dt = 1.0 / params.fs
    t = np.arange(0.0, params.total_time(), dt)
    F = np.zeros_like(t)
    pos = np.zeros_like(t)

    F_spike = params.k_spike * params.W_kN
    F_plateau = params.k_plateau * params.W_kN
    F_hold = params.W_kN
    F_low = params.k_low_ecc * params.W_kN
    F_brake = params.k_brake * params.W_kN

    t0 = 0.0
    t1 = t0 + params.t_c1
    m = _mask(t, t0, t1)
    F[m] = np.linspace(0.0, F_spike, m.sum(), endpoint=False)

    t0 = t1
    t1 = t0 + params.t_c2
    m = _mask(t, t0, t1)
    F[m] = np.linspace(F_spike, F_plateau, m.sum(), endpoint=False)

    t0 = t1
    t1 = t0 + params.t_c3
    m = _mask(t, t0, t1)
    F[m] = np.linspace(F_plateau, F_hold, m.sum(), endpoint=False)

    t0 = t1
    t1 = t0 + params.t_top
    m = _mask(t, t0, t1)
    F[m] = F_hold

    t0 = t1
    t1 = t0 + params.t_e1
    m = _mask(t, t0, t1)
    F[m] = np.linspace(F_hold, F_low, m.sum(), endpoint=False)

    t0 = t1
    t1 = t0 + params.t_e2
    m = _mask(t, t0, t1)
    n = m.sum()
    half = max(1, n // 2)
    ecc = np.concatenate(
        [
            np.linspace(F_low, F_brake, half, endpoint=False),
            np.linspace(F_brake, 0.0, n - half, endpoint=True),
        ]
    )
    F[m] = ecc[: n]

    stroke = max(params.stroke_m, 0.0)
    con_total = params.t_c1 + params.t_c2 + params.t_c3
    ecc_total = params.t_e1 + params.t_e2

    if stroke <= 0 or (con_total <= 0 and ecc_total <= 0):
        return t, np.clip(F, 0.0, None), pos

    if con_total > 0 and stroke > 0:
        mask_con = _mask(t, 0.0, con_total)
        n_con = mask_con.sum()
        increments = _velocity_increments(n_con, stroke, a=1.2, b=2.2)
        pos_con = _cumulative_positions(increments, start=0.0, ascending=True)
        if pos_con.size:
            pos_con[-1] = min(stroke, pos_con[-1])
            pos[mask_con] = pos_con

    mask_pause = _mask(t, con_total, con_total + params.t_top)
    if mask_pause.any():
        pos[mask_pause] = stroke

    if ecc_total > 0 and stroke > 0:
        mask_ecc = _mask(t, con_total + params.t_top, params.total_time())
        n_ecc = mask_ecc.sum()
        increments = _velocity_increments(n_ecc, stroke, a=2.4, b=1.2)
        pos_ecc = _cumulative_positions(increments, start=stroke, ascending=False)
        if pos_ecc.size:
            pos_ecc[-1] = 0.0
            pos[mask_ecc] = np.clip(pos_ecc, 0.0, stroke)

    return t, np.clip(F, 0.0, None), np.clip(pos, 0.0, stroke)


def _apply_sensor_effects(
    force: np.ndarray,
    dt: float,
    noise: Optional[NoiseSettings],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not noise:
        return force, None

    adjusted = force.astype(float).copy()
    if noise.has_noise():
        adjusted += rng.normal(0.0, noise.noise_std, size=adjusted.size)

    if noise.has_drift():
        adjusted += noise.drift_per_second * dt * np.arange(adjusted.size)

    counts = None
    if noise.has_quantization():
        max_counts = (1 << noise.quantization_bits) - 1
        if max_counts <= 0:
            max_counts = 1
        kN_per_count = noise.quantization_range_kN / max_counts
        # Clip to sensor range [0, range]
        clipped = np.clip(adjusted, 0.0, noise.quantization_range_kN)
        counts = np.rint(clipped / kN_per_count).astype(int)
        adjusted = counts * kN_per_count

    return np.clip(adjusted, 0.0, None), counts


def simulate_rep(
    params: SimulationParams,
    *,
    noise: Optional[NoiseSettings] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate a repetition and return a time-series dataframe.

    Parameters
    ----------
    params:
        Fully specified tempo, multiplier, and sampling parameters.
    noise:
        Optional :class:`NoiseSettings` describing Gaussian noise, drift, and
        quantization to apply after the analytic trace is generated.
    seed:
        Random seed forwarded to NumPy's default RNG for reproducible noise.

    Returns
    -------
    pandas.DataFrame
        Columns include ``time_s``, ``force_kN``, ``position_m``, ``velocity_mps``
        plus ``adc_counts`` when quantization is active.
    """

    t, F, pos = simulate_trace(params)
    rng = np.random.default_rng(seed)
    F_adj, counts = _apply_sensor_effects(F, dt=1.0 / params.fs, noise=noise, rng=rng)
    
    # Calculate velocity using backward difference so velocity at time t
    # represents how we arrived at position[t] from position[t-1]
    dt = 1.0 / params.fs
    velocity = np.zeros_like(pos)
    if pos.size > 1:
        velocity[1:] = np.diff(pos) / dt
        velocity[0] = 0.0  # First point always starts at zero velocity
        velocity[-1] = 0.0  # Last point always ends at zero velocity

    data = {
        "time_s": t,
        "force_kN": F_adj,
        "position_m": pos,
        "velocity_mps": velocity,
    }
    if counts is not None:
        data["adc_counts"] = counts
    return pd.DataFrame(data)


def simulate_rep_simple(
    weight_value: float = DEFAULT_WEIGHT_LBS,
    *,
    weight_unit: str = "lbs",
    t_concentric: float = 1.0,
    t_pause_top: float = 0.2,
    t_eccentric: float = 2.0,
    fs: int = DEFAULT_FS,
    stroke_m: float = 0.5,
    multipliers: Optional[dict[str, float]] = None,
    noise: Optional[NoiseSettings] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate a rep using high-level, user-friendly tempo inputs.

    Parameters
    ----------
    weight_value:
        Stack weight value entered by the user.
    weight_unit:
        Either ``"lbs"`` or ``"kg"``; determines how ``weight_value`` is
        converted into cable tension (kN).
    t_concentric / t_pause_top / t_eccentric:
        Total durations for the upward, pause, and downward phases. Internally
        split according to ``DEFAULT_CONCENTRIC_SPLIT`` and
        ``DEFAULT_ECCENTRIC_SPLIT``.
    fs:
        Sample rate passed to :class:`SimulationParams`.
    multipliers:
        Optional overrides for ``k_spike``, ``k_plateau``, ``k_low_ecc``, and
        ``k_brake``. Missing keys fall back to ``DEFAULT_MULTIPLIERS``.
    stroke_m:
        Estimated distance the stack travels during the concentric phase.

    noise / seed:
        Passed through to :func:`simulate_rep` so the same sensor effects and
        reproducibility knobs remain available.

    Returns
    -------
    pandas.DataFrame
        Same schema as :func:`simulate_rep`.
    """

    W_kN = _weight_to_kN(weight_value, weight_unit)
    m = {**DEFAULT_MULTIPLIERS, **(multipliers or {})}
    t_c1, t_c2, t_c3 = _split_durations(t_concentric, DEFAULT_CONCENTRIC_SPLIT)
    t_e1, t_e2 = _split_durations(t_eccentric, DEFAULT_ECCENTRIC_SPLIT)

    params = SimulationParams(
        W_kN=W_kN,
        t_c1=t_c1,
        t_c2=t_c2,
        t_c3=t_c3,
        t_top=t_pause_top,
        t_e1=t_e1,
        t_e2=t_e2,
        k_spike=m["k_spike"],
        k_plateau=m["k_plateau"],
        k_low_ecc=m["k_low_ecc"],
        k_brake=m["k_brake"],
        fs=fs,
        stroke_m=stroke_m,
    )
    return simulate_rep(params, noise=noise, seed=seed)


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a numeric value.")


def _prompt_int(prompt: str, default: int) -> int:
    return int(round(_prompt_float(prompt, float(default))))


def _prompt_choice(prompt: str, default: str, choices: Sequence[str]) -> str:
    choices_lower = {c.lower(): c for c in choices}
    default_lower = default.lower()
    while True:
        raw = input(f"{prompt} {choices} [{default}]: ").strip().lower()
        if not raw:
            return choices_lower[default_lower]
        if raw in choices_lower:
            return choices_lower[raw]
        print(f"Please choose one of {choices}.")


def _collect_simple_inputs() -> dict[str, object]:
    print("\nSimple mode (high-level inputs)")
    weight = _prompt_float("Weight", DEFAULT_WEIGHT_LBS)
    unit = _prompt_choice("Units", "lbs", ("lbs", "kg"))
    t_concentric = _prompt_float("Total concentric time (s)", 1.0)
    t_pause = _prompt_float("Pause at top (s)", 0.2)
    t_eccentric = _prompt_float("Total eccentric time (s)", 2.0)
    fs = _prompt_int("Sample rate (Hz)", DEFAULT_FS)
    stroke_m = _prompt_float("Stack travel / ROM (m)", 0.5)
    return {
        "weight_value": weight,
        "weight_unit": unit,
        "t_concentric": t_concentric,
        "t_pause_top": t_pause,
        "t_eccentric": t_eccentric,
        "fs": fs,
        "stroke_m": stroke_m,
    }


def _plot_dataframe(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["time_s"], df["force_kN"], label="Force (kN)", linewidth=2.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cable tension (kN)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Simulated Force Curve")

    if "adc_counts" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["time_s"], df["adc_counts"], color="tab:orange", linewidth=1.2, alpha=0.7)
        ax2.set_ylabel("ADC counts")

    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def _run_cli() -> None:
    print("Force Curve Simulator\n========================")
    if SIMPLE_MODE:
        kwargs = _collect_simple_inputs()
        df = simulate_rep_simple(**kwargs)
    else:
        print("Advanced mode active. Edit SimulationParams() below to tweak values.\n")
        params = SimulationParams()
        df = simulate_rep(params)

    summary = df[["force_kN"]].agg(["min", "max", "mean"]).T
    print("\nForce summary (kN):")
    print(summary.round(3))
    _plot_dataframe(df)


if __name__ == "__main__":
    _run_cli()
