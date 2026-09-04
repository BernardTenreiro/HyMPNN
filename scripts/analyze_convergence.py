#!/usr/bin/env python3
"""Fit convergence models to the validation MAE curves in ``logs/``.

The primary purpose of this script is to test whether one constant exponential
decay rate is an adequate summary of a training run.  It also fits an
asymptotic power law with the same number of parameters as a diagnostic:

    exponential: MAE(e) = c + a exp(-k e)
    power law:   MAE(e) = c + a (e + 1)^(-p)

For a fixed rate, the optimal non-negative floor and amplitude are found by
least squares.  The remaining one-dimensional rate search is dependency-free
and deterministic; SciPy is not required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

RUN_ORDER = (
    "has_128_64",
    "has_64_64",
    "he_128_64",
    "he_64_64",
    "hs_128_64",
    "hs_64_64",
    "hj_128_64",
    "hj_64_64",
    "s5_128",
    "s5_64",
    "s7_128",
    "s7_64",
)
WINDOW_STARTS = (0, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class Curve:
    name: str
    epochs: np.ndarray
    values: np.ndarray
    completed: bool


@dataclass(frozen=True)
class Fit:
    model: str
    rate: float
    floor: float
    amplitude: float
    sum_squared_error: float
    r_squared: float
    root_mean_squared_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/analysis/convergence"),
    )
    return parser.parse_args()


def load_curves(log_directory: Path) -> list[Curve]:
    paths = {path.parent.name: path for path in log_directory.glob("*/metrics.json")}
    missing = set(RUN_ORDER) - paths.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise FileNotFoundError(f"missing metrics.json for: {names}")

    curves = []
    for name in RUN_ORDER:
        with paths[name].open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)

        epochs = np.asarray(metrics["epochs"], dtype=float)
        values = np.asarray(metrics["val_mae"], dtype=float)
        if epochs.ndim != 1 or values.ndim != 1 or len(epochs) != len(values):
            raise ValueError(f"invalid epoch or val_mae arrays in {paths[name]}")
        if len(epochs) < 3 or not np.all(np.diff(epochs) > 0):
            raise ValueError(
                f"epochs must contain at least three increasing values in {paths[name]}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"val_mae contains non-finite values in {paths[name]}")

        curves.append(Curve(name, epochs, values, bool(metrics.get("completed", False))))
    return curves


def solve_nonnegative_linear_fit(basis: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Return the least-squares floor and amplitude constrained to be non-negative."""
    design = np.column_stack((np.ones_like(basis), basis))
    unconstrained, *_ = np.linalg.lstsq(design, values, rcond=None)

    candidates: list[tuple[float, float]] = []
    if np.all(unconstrained >= 0.0):
        candidates.append((float(unconstrained[0]), float(unconstrained[1])))

    candidates.append((max(float(values.mean()), 0.0), 0.0))
    basis_norm = float(np.dot(basis, basis))
    if basis_norm > np.finfo(float).tiny:
        amplitude = max(float(np.dot(basis, values) / basis_norm), 0.0)
        candidates.append((0.0, amplitude))

    def squared_error(parameters: tuple[float, float]) -> float:
        floor, candidate_amplitude = parameters
        residual = values - floor - candidate_amplitude * basis
        return float(np.dot(residual, residual))

    return min(candidates, key=squared_error)


def fit_rate_model(
    epochs: np.ndarray,
    values: np.ndarray,
    model: str,
) -> Fit:
    if model == "exponential":
        make_basis: Callable[[float], np.ndarray] = lambda rate: np.exp(-rate * epochs)
    elif model == "power_law":
        make_basis = lambda rate: np.power(epochs + 1.0, -rate)
    else:
        raise ValueError(f"unknown model: {model}")

    def evaluate(log_rate: float) -> tuple[float, float, float, float]:
        rate = math.exp(log_rate)
        basis = make_basis(rate)
        floor, amplitude = solve_nonnegative_linear_fit(basis, values)
        residual = values - floor - amplitude * basis
        return float(np.dot(residual, residual)), rate, floor, amplitude

    log_rates = np.linspace(-12.0, 2.0, 2801)
    evaluations = [evaluate(log_rate) for log_rate in log_rates]
    best_index = min(range(len(evaluations)), key=lambda index: evaluations[index][0])
    lower = float(log_rates[max(0, best_index - 1)])
    upper = float(log_rates[min(len(log_rates) - 1, best_index + 1)])

    # Golden-section refinement of the best interval in log-rate space.
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = evaluate(left)
    right_value = evaluate(right)
    for _ in range(80):
        if left_value[0] < right_value[0]:
            upper, right, right_value = right, left, left_value
            left = upper - ratio * (upper - lower)
            left_value = evaluate(left)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + ratio * (upper - lower)
            right_value = evaluate(right)

    sum_squared_error, rate, floor, amplitude = min(
        left_value,
        right_value,
        key=lambda result: result[0],
    )
    centered = values - values.mean()
    total_sum_squares = float(np.dot(centered, centered))
    r_squared = 1.0 - sum_squared_error / total_sum_squares
    rmse = math.sqrt(sum_squared_error / len(values))
    return Fit(model, rate, floor, amplitude, sum_squared_error, r_squared, rmse)


def predict(fit: Fit, epochs: np.ndarray) -> np.ndarray:
    if fit.model == "exponential":
        basis = np.exp(-fit.rate * epochs)
    elif fit.model == "power_law":
        basis = np.power(epochs + 1.0, -fit.rate)
    else:
        raise ValueError(f"unknown model: {fit.model}")
    return fit.floor + fit.amplitude * basis


def akaike_information_criterion(fit: Fit, sample_count: int) -> float:
    # Both models have three fitted parameters, including the floor.
    return sample_count * math.log(fit.sum_squared_error / sample_count) + 2 * 3


def write_summary(
    curves: list[Curve],
    fits: dict[str, dict[str, Fit]],
    output_path: Path,
) -> None:
    field_names = (
        "run",
        "completed",
        "epochs",
        "exp_k",
        "exp_half_life_epochs",
        "exp_floor",
        "exp_r_squared",
        "exp_rmse",
        "power_p",
        "power_floor",
        "power_r_squared",
        "power_rmse",
        "aic_exp_minus_power",
    )
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=field_names)
        writer.writeheader()
        for curve in curves:
            exponential = fits[curve.name]["exponential"]
            power_law = fits[curve.name]["power_law"]
            writer.writerow(
                {
                    "run": curve.name,
                    "completed": curve.completed,
                    "epochs": len(curve.epochs),
                    "exp_k": exponential.rate,
                    "exp_half_life_epochs": math.log(2.0) / exponential.rate,
                    "exp_floor": exponential.floor,
                    "exp_r_squared": exponential.r_squared,
                    "exp_rmse": exponential.root_mean_squared_error,
                    "power_p": power_law.rate,
                    "power_floor": power_law.floor,
                    "power_r_squared": power_law.r_squared,
                    "power_rmse": power_law.root_mean_squared_error,
                    "aic_exp_minus_power": akaike_information_criterion(
                        exponential, len(curve.epochs)
                    )
                    - akaike_information_criterion(power_law, len(curve.epochs)),
                }
            )


def write_window_sensitivity(curves: list[Curve], output_directory: Path) -> None:
    field_names = ("run", "start_epoch", "samples", "exp_k", "power_p")
    with (output_directory / "window_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=field_names)
        writer.writeheader()
        for curve in curves:
            for start_epoch in WINDOW_STARTS:
                selected = curve.epochs >= start_epoch
                if int(selected.sum()) < 3:
                    continue
                exponential = fit_rate_model(
                    curve.epochs[selected], curve.values[selected], "exponential"
                )
                power_law = fit_rate_model(
                    curve.epochs[selected], curve.values[selected], "power_law"
                )
                writer.writerow(
                    {
                        "run": curve.name,
                        "start_epoch": start_epoch,
                        "samples": int(selected.sum()),
                        "exp_k": exponential.rate,
                        "power_p": power_law.rate,
                    }
                )


def write_markdown_report(
    curves: list[Curve],
    fits: dict[str, dict[str, Fit]],
    common_horizon: int,
    output_directory: Path,
) -> None:
    lines = [
        "# Validation-MAE convergence fits",
        "",
        "Fits use `val_mae`; test MAE is intentionally not used for model comparison.",
        f"The comparison below uses the common epoch range 0–{common_horizon} because one run ",
        "is incomplete. The exponential is `c + a exp(-k e)` and the power law is",
        "`c + a (e + 1)^(-p)`. Larger `k` or `p` denotes faster fitted decay.",
        "",
        "| Run | Status | Exp k | Exp half-life | Exp R² | Power p | Power R² | ΔAIC (exp−power) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for curve in curves:
        exponential = fits[curve.name]["exponential"]
        power_law = fits[curve.name]["power_law"]
        delta_aic = akaike_information_criterion(
            exponential, len(curve.epochs)
        ) - akaike_information_criterion(power_law, len(curve.epochs))
        status = "complete" if curve.completed else f"incomplete ({len(curve.epochs)})"
        lines.append(
            f"| {curve.name} | {status} | {exponential.rate:.5f} | "
            f"{math.log(2.0) / exponential.rate:.1f} | {exponential.r_squared:.4f} | "
            f"{power_law.rate:.4f} | {power_law.r_squared:.4f} | {delta_aic:.1f} |"
        )
    lines.extend(
        [
            "",
            "A positive ΔAIC favors the power law. The two candidate models have the same",
            "number of fitted parameters, so this comparison does not involve a complexity penalty",
            "difference. Full-curve fits are in `summary.csv`; common-horizon fits are in",
            "`common_horizon_summary.csv`; and start-window sensitivity is in",
            "`window_sensitivity.csv`.",
            "",
        ]
    )
    (output_directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_fits(
    curves: list[Curve],
    fits: dict[str, dict[str, Fit]],
    output_directory: Path,
) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(14, 15), sharex=True)
    for axis, curve in zip(axes.flat, curves):
        plot_epochs = curve.epochs + 1.0
        exponential = fits[curve.name]["exponential"]
        power_law = fits[curve.name]["power_law"]
        axis.plot(plot_epochs, curve.values, color="0.35", linewidth=1.0, label="validation")
        axis.plot(
            plot_epochs,
            predict(exponential, curve.epochs),
            color="#d95f02",
            linewidth=1.8,
            linestyle="--",
            label=f"exponential (R²={exponential.r_squared:.3f})",
        )
        axis.plot(
            plot_epochs,
            predict(power_law, curve.epochs),
            color="#1b9e77",
            linewidth=1.8,
            label=f"power law (R²={power_law.r_squared:.3f})",
        )
        suffix = "" if curve.completed else " — INCOMPLETE"
        axis.set_title(f"{curve.name}{suffix}")
        axis.set_xscale("log")
        axis.grid(linestyle=":", linewidth=0.6, alpha=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8)

    for axis in axes[-1, :]:
        axis.set_xlabel("Epoch")
    for axis in axes[:, 0]:
        axis.set_ylabel("Validation MAE")
    figure.suptitle("Asymptotic convergence fits", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_directory / "fits.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_rate_comparison(
    curves: list[Curve],
    fits: dict[str, dict[str, Fit]],
    common_horizon: int,
    output_directory: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 7))
    panels = (
        ("exponential", "k", "Single exponential (diagnostic)", "#d95f02"),
        ("power_law", "p", "Power law (preferred)", "#1b9e77"),
    )
    completed = {curve.name: curve.completed for curve in curves}

    for axis, (model, symbol, title, color) in zip(axes, panels):
        ordered = sorted(curves, key=lambda curve: fits[curve.name][model].rate)
        names = [curve.name for curve in ordered]
        rates = [fits[name][model].rate for name in names]
        bars = axis.barh(names, rates, color=color, alpha=0.85)
        for bar, name, rate in zip(bars, names, rates):
            if not completed[name]:
                bar.set_hatch("///")
                bar.set_edgecolor("black")
            axis.text(
                rate,
                bar.get_y() + bar.get_height() / 2.0,
                f" {rate:.4f}",
                va="center",
                fontsize=8,
            )
        axis.set_xlabel(f"Fitted {symbol} (larger = faster decay)")
        axis.set_title(title)
        axis.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(right=max(rates) * 1.17)

    figure.suptitle(f"Convergence-rate comparison, epochs 0–{common_horizon}")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        output_directory / "rate_comparison.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    curves = load_curves(args.logs)
    full_fits = {
        curve.name: {
            model: fit_rate_model(curve.epochs, curve.values, model)
            for model in ("exponential", "power_law")
        }
        for curve in curves
    }
    common_horizon = int(min(curve.epochs[-1] for curve in curves))
    common_curves = []
    for curve in curves:
        selected = curve.epochs <= common_horizon
        common_curves.append(
            Curve(
                curve.name,
                curve.epochs[selected],
                curve.values[selected],
                curve.completed,
            )
        )
    common_fits = {
        curve.name: {
            model: fit_rate_model(curve.epochs, curve.values, model)
            for model in ("exponential", "power_law")
        }
        for curve in common_curves
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(curves, full_fits, args.output_dir / "summary.csv")
    write_summary(
        common_curves,
        common_fits,
        args.output_dir / "common_horizon_summary.csv",
    )
    write_window_sensitivity(curves, args.output_dir)
    write_markdown_report(common_curves, common_fits, common_horizon, args.output_dir)
    plot_fits(curves, full_fits, args.output_dir)
    plot_rate_comparison(common_curves, common_fits, common_horizon, args.output_dir)
    print(f"Wrote convergence analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
