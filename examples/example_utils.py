"""Shared helpers for cellml2py example scripts.

Import this module from any script in the examples/ directory.  Each script
inserts its own directory into sys.path so the import resolves without
requiring an installed package.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Forcing helpers
# ---------------------------------------------------------------------------

def build_injection_current(
    forcing_mode: str,
    constant_current: float,
    amp: float,
    pulse_start: float,
    pulse_width: float,
    pulse_interval: float,
    n_pulses: int,
    ramp_start: float = 0.0,
    ramp_end: float = 1.0,
    ramp_i_start: float = 0.0,
    ramp_i_end: float = 1.0,
) -> Callable[[float], float]:
    """Return a scalar injection-current callable ``i(t) -> float``.

    Parameters
    ----------
    forcing_mode:
        ``"constant"`` for a DC current, ``"pulse"`` for a train of
        rectangular pulses, or ``"ramp"`` for a linearly-ramped current.
    constant_current:
        Value returned in constant mode.
    amp:
        Pulse amplitude in pulse mode.
    pulse_start:
        Time of the leading edge of the first pulse.
    pulse_width:
        Duration of each pulse.
    pulse_interval:
        Time between the leading edges of successive pulses.
    n_pulses:
        Number of pulses.
    ramp_start:
        Time at which the ramp begins (returns *ramp_i_start* before this).
    ramp_end:
        Time at which the ramp ends (returns *ramp_i_end* after this).
    ramp_i_start:
        Current value at *ramp_start*.
    ramp_i_end:
        Current value at *ramp_end*.
    """
    if forcing_mode == "constant":
        def _constant(_t: float) -> float:
            return constant_current
        return _constant

    if forcing_mode == "ramp":
        _span = ramp_end - ramp_start
        _di = ramp_i_end - ramp_i_start

        def _ramp(t):
            # Pure-arithmetic clip: no Python if-branches on t, so JAX can
            # trace this function.  Works for plain floats and JAX tracers.
            t_lo = t + (ramp_start - t) * (t < ramp_start)       # max(t, ramp_start)
            t_clamped = t_lo - (t_lo - ramp_end) * (t_lo > ramp_end)  # min(above, ramp_end)
            return ramp_i_start + _di * (t_clamped - ramp_start) / _span

        return _ramp

    starts = [pulse_start + pulse_interval * i for i in range(n_pulses)]

    def _pulse(t):
        # Avoid Python if-branches on t so the function is JAX-traceable.
        result = sum(
            amp * ((t >= start) * (t <= start + pulse_width))
            for start in starts
        )
        return np.minimum(result, amp)

    return _pulse


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------

def add_forcing_args(
    parser: argparse.ArgumentParser,
    *,
    default_constant: float = 0.0,
    default_amp: float = 20.0,
    default_pulse_start: float = 10.0,
    default_pulse_width: float = 2.0,
    default_pulse_interval: float = 15.0,
    default_n_pulses: int = 5,
) -> None:
    """Add standard injection-current forcing arguments to *parser*."""
    parser.add_argument(
        "--forcing-mode",
        choices=("pulse", "constant", "ramp"),
        default="constant",
        help="How to generate the injection current.  (default: constant)",
    )
    parser.add_argument(
        "--constant-current",
        type=float,
        default=default_constant,
        metavar="AMPS",
        help="DC current value used when --forcing-mode constant.  "
             f"(default: {default_constant})",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=default_amp,
        help=f"Pulse amplitude in pulse mode.  (default: {default_amp})",
    )
    parser.add_argument(
        "--pulse-start",
        type=float,
        default=default_pulse_start,
        help=f"Start time of first pulse.  (default: {default_pulse_start})",
    )
    parser.add_argument(
        "--pulse-width",
        type=float,
        default=default_pulse_width,
        help=f"Duration of each pulse.  (default: {default_pulse_width})",
    )
    parser.add_argument(
        "--pulse-interval",
        type=float,
        default=default_pulse_interval,
        help=f"Time between pulse leading edges.  (default: {default_pulse_interval})",
    )
    parser.add_argument(
        "--n-pulses",
        type=int,
        default=default_n_pulses,
        help=f"Number of injected pulses.  (default: {default_n_pulses})",
    )
    parser.add_argument(
        "--ramp-start",
        type=float,
        default=0.0,
        metavar="T",
        help="Time at which the ramp begins (ramp mode only).  (default: 0.0)",
    )
    parser.add_argument(
        "--ramp-end",
        type=float,
        default=1.0,
        metavar="T",
        help="Time at which the ramp ends (ramp mode only).  (default: 1.0)",
    )
    parser.add_argument(
        "--ramp-i-start",
        type=float,
        default=0.0,
        metavar="AMPS",
        help="Current at ramp start (ramp mode only).  (default: 0.0)",
    )
    parser.add_argument(
        "--ramp-i-end",
        type=float,
        default=1.0,
        metavar="AMPS",
        help="Current at ramp end (ramp mode only).  (default: 1.0)",
    )


def build_injection_current_from_args(args: argparse.Namespace) -> Callable[[float], float]:
    """Build a :func:`build_injection_current` callable from parsed *args*."""
    return build_injection_current(
        forcing_mode=args.forcing_mode,
        constant_current=args.constant_current,
        amp=args.amp,
        pulse_start=args.pulse_start,
        pulse_width=args.pulse_width,
        pulse_interval=args.pulse_interval,
        n_pulses=args.n_pulses,
        ramp_start=getattr(args, "ramp_start", 0.0),
        ramp_end=getattr(args, "ramp_end", 1.0),
        ramp_i_start=getattr(args, "ramp_i_start", 0.0),
        ramp_i_end=getattr(args, "ramp_i_end", 1.0),
    )


def add_solver_args(
    parser: argparse.ArgumentParser,
    *,
    default_method: str = "LSODA",
    default_max_step: float = float("inf"),
    default_rtol: float = 1e-4,
    default_atol: float = 1e-6,
) -> None:
    """Add standard ODE solver arguments to *parser*."""
    parser.add_argument(
        "--method",
        default=default_method,
        choices=("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"),
        help=f"ODE solver method.  Stiff-capable: LSODA, Radau, BDF.  "
             f"(default: {default_method})",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=default_rtol,
        help=f"Relative solver tolerance.  (default: {default_rtol})",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=default_atol,
        help=f"Absolute solver tolerance.  (default: {default_atol})",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=default_max_step,
        metavar="DT",
        help=f"Maximum solver step size.  (default: {default_max_step})",
    )


def add_output_args(
    parser: argparse.ArgumentParser,
    *,
    default_plot_out: str = "output.png",
) -> None:
    """Add --plot-out and --show arguments to *parser*."""
    parser.add_argument(
        "--plot-out",
        default=default_plot_out,
        help=f"Path to save the output plot.  (default: {default_plot_out})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plot window interactively after saving.",
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_voltage_and_current(
    t: np.ndarray,
    voltage: np.ndarray,
    current_trace: np.ndarray,
    *,
    title: str = "Membrane Voltage with Forcing Current",
    voltage_label: str = "V (mV)",
    current_label: str = "i_inj",
    time_label: str = "time",
    plot_path: str | Path = "output.png",
    show: bool = False,
) -> Path:
    """Save a two-panel voltage / current plot and return the output path."""
    fig, (ax_v, ax_i) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )

    ax_v.plot(t, voltage, color="tab:blue", linewidth=1.8)
    ax_v.set_ylabel(voltage_label)
    ax_v.set_title(title)
    ax_v.grid(alpha=0.3)

    ax_i.step(t, current_trace, where="post", color="tab:red", linewidth=1.5)
    ax_i.set_xlabel(time_label)
    ax_i.set_ylabel(current_label)
    ax_i.grid(alpha=0.3)

    out = Path(plot_path)
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def plot_trajectory(
    t: np.ndarray,
    y: np.ndarray,
    state_names: tuple[str, ...] | list[str],
    *,
    title: str = "State Trajectory",
    plot_path: str | Path = "trajectory.png",
    show: bool = False,
    max_legend: int = 8,
) -> Path:
    """Save a single-panel all-states trajectory plot and return the output path."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for idx, name in enumerate(state_names):
        ax.plot(t, y[idx], label=name)
    ax.set_xlabel("time")
    ax.set_ylabel("state value")
    ax.set_title(title)
    if len(state_names) <= max_legend:
        ax.legend(loc="best")
    ax.grid(alpha=0.3)

    out = Path(plot_path)
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out
