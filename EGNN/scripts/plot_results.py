#!/usr/bin/env python3
"""Plot validation checkpoints from one or more HyMPNN result files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunResults:
    label: str
    epochs: np.ndarray
    mean_absolute_error: np.ndarray
    mean_squared_error: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MAE and MSE curves from HyMPNN losess.json files."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Legend label and a losess.json file (or its experiment directory).",
    )
    parser.add_argument("--output", type=Path, default=Path("results.png"))
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def load_run(specification: str) -> RunResults:
    try:
        label, raw_path = specification.split("=", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"expected LABEL=PATH, received {specification!r}") from error

    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "losess.json"
    with path.open(encoding="utf-8") as result_file:
        results = json.load(result_file)

    epochs = np.asarray(results["epochs"], dtype=int) + 1
    mean_absolute_error = np.asarray(results["test_mae"], dtype=float)
    mean_squared_error = np.asarray(results["test_mse"], dtype=float)
    if not (len(epochs) == len(mean_absolute_error) == len(mean_squared_error)):
        raise ValueError(f"checkpoint arrays have different lengths in {path}")
    if not len(epochs):
        raise ValueError(f"no validation checkpoints found in {path}")

    return RunResults(label, epochs, mean_absolute_error, mean_squared_error)


def plot_results(runs: list[RunResults], output: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    metrics = (
        ("mean_squared_error", "MSE", "Test MSE"),
        ("mean_absolute_error", "MAE", "Test MAE"),
    )

    for axis, (attribute, y_label, title) in zip(axes, metrics):
        for run in runs:
            values = getattr(run, attribute)
            axis.semilogy(run.epochs, values, linewidth=1.8, label=run.label)
            if attribute == "mean_squared_error":
                best_index = int(values.argmin())
                axis.scatter(run.epochs[best_index], values[best_index], s=24, zorder=3)

        axis.set_xlabel("Epoch")
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.grid(which="major", linestyle=":", linewidth=0.6, alpha=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(runs), 4))
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")


def main() -> None:
    args = parse_args()
    plot_results([load_run(specification) for specification in args.run], args.output, args.dpi)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
