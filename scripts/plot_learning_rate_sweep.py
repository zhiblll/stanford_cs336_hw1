from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path


DEFAULT_RUNS = [
    ("max_lr=3e-4", Path("artifacts/runs/lr_3e-4_5k/metrics.jsonl")),
    ("max_lr=1e-3", Path("artifacts/runs/tinystories_5k/metrics.jsonl")),
    ("max_lr=3e-3", Path("artifacts/runs/lr_3e-3_5k/metrics.jsonl")),
    ("max_lr=1e-2", Path("artifacts/runs/lr_1e-2_5k/metrics.jsonl")),
]

COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]


@dataclass
class Run:
    label: str
    path: Path
    rows: list[dict[str, float | int | None]]

    @property
    def validation_rows(self) -> list[dict[str, float | int | None]]:
        return [row for row in self.rows if row["validation_loss"] is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning-rate sweep metrics without third-party dependencies."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=METRICS_PATH",
        help="Run to include. Repeat for multiple runs. Defaults to the four TinyStories LR sweep runs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/plots/lr_sweep"),
        help="Directory for generated SVG plots and summary CSV.",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=15,
        help="Moving-average window for training loss.",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="TinyStories Learning-Rate Sweep",
        help="Prefix used in plot titles.",
    )
    return parser.parse_args()


def load_run(label: str, path: Path) -> Run:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"step", "train_loss", "validation_loss", "learning_rate", "elapsed_seconds"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing metrics: {sorted(missing)}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")
    if not any(row["validation_loss"] is not None for row in rows):
        raise ValueError(f"No validation loss records found in: {path}")
    return Run(label=label, path=path, rows=rows)


def parse_runs(run_args: list[str]) -> list[Run]:
    specs = DEFAULT_RUNS if not run_args else []
    if run_args:
        for spec in run_args:
            if "=" not in spec:
                raise ValueError(f"--run must use LABEL=METRICS_PATH format: {spec}")
            label, path = spec.split("=", 1)
            if not label or not path:
                raise ValueError(f"--run must use LABEL=METRICS_PATH format: {spec}")
            specs.append((label, Path(path)))
    return [load_run(label, path) for label, path in specs]


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("smoothing_window must be positive.")
    smoothed = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        current_window = min(index + 1, window)
        smoothed.append(running_sum / current_window)
    return smoothed


def padded_bounds(values: list[float], start_at_zero: bool = False) -> tuple[float, float]:
    low = 0.0 if start_at_zero else min(values)
    high = max(values)
    if math.isclose(low, high):
        padding = max(abs(high) * 0.05, 1e-6)
        return low - padding, high + padding
    padding = (high - low) * 0.08
    return max(0.0, low - padding) if start_at_zero else low - padding, high + padding


def format_tick(value: float, scientific: bool = False) -> str:
    if scientific:
        return f"{value:.1e}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def plot_svg(
    runs: list[Run],
    series: list[tuple[list[float], list[float]]],
    title: str,
    y_label: str,
    output_path: Path,
    scientific_y: bool = False,
    start_y_at_zero: bool = False,
    markers: bool = False,
) -> None:
    width, height = 1040, 620
    left, right, top, bottom = 94, 260, 70, 76
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [x for xs, _ in series for x in xs]
    y_values = [y for _, ys in series for y in ys]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = padded_bounds(y_values, start_at_zero=start_y_at_zero)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #222; letter-spacing: 0; }",
        ".title { font-size: 22px; font-weight: 600; }",
        ".label { font-size: 14px; }",
        ".tick { font-size: 12px; fill: #555; }",
        ".legend { font-size: 13px; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="35">{escape(title)}</text>',
    ]

    for tick in range(6):
        fraction = tick / 5
        y_value = y_max - fraction * (y_max - y_min)
        y = top + fraction * plot_height
        svg.append(
            f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.2f}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">'
            f"{escape(format_tick(y_value, scientific_y))}</text>"
        )

    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = left + fraction * plot_width
        svg.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_height}" '
            'stroke="#f3f4f6" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle">'
            f"{escape(format_tick(x_value))}</text>"
        )

    svg.extend(
        [
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_height}" stroke="#333"/>',
            f'<line x1="{left}" x2="{left + plot_width}" y1="{top + plot_height}" '
            f'y2="{top + plot_height}" stroke="#333"/>',
            f'<text class="label" x="{left + plot_width / 2:.2f}" y="{height - 25}" text-anchor="middle">'
            "Training step</text>",
            f'<text class="label" x="27" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
            f'transform="rotate(-90 27 {top + plot_height / 2:.2f})">{escape(y_label)}</text>',
        ]
    )

    for index, ((xs, ys), run) in enumerate(zip(series, runs)):
        color = COLORS[index % len(COLORS)]
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if markers:
            for x, y in zip(xs, ys):
                svg.append(
                    f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.6" fill="{color}"/>'
                )
        legend_y = top + 18 + index * 30
        legend_x = left + plot_width + 34
        svg.append(
            f'<line x1="{legend_x}" x2="{legend_x + 30}" y1="{legend_y}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text class="legend" x="{legend_x + 40}" y="{legend_y + 5}">{escape(run.label)}</text>'
        )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_summary(runs: list[Run], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "metrics_path",
                "final_step",
                "final_train_loss",
                "final_validation_loss",
                "best_validation_loss",
                "best_validation_step",
                "elapsed_minutes",
            ]
        )
        for run in runs:
            validation_rows = run.validation_rows
            best = min(validation_rows, key=lambda row: float(row["validation_loss"]))
            final = validation_rows[-1]
            writer.writerow(
                [
                    run.label,
                    run.path,
                    run.rows[-1]["step"],
                    f"{float(run.rows[-1]['train_loss']):.6f}",
                    f"{float(final['validation_loss']):.6f}",
                    f"{float(best['validation_loss']):.6f}",
                    best["step"],
                    f"{float(run.rows[-1]['elapsed_seconds']) / 60:.2f}",
                ]
            )


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    output_dir = args.output_dir

    validation_series = [
        (
            [float(row["step"]) for row in run.validation_rows],
            [float(row["validation_loss"]) for row in run.validation_rows],
        )
        for run in runs
    ]
    train_series = [
        (
            [float(row["step"]) for row in run.rows],
            moving_average([float(row["train_loss"]) for row in run.rows], args.smoothing_window),
        )
        for run in runs
    ]
    learning_rate_series = [
        (
            [float(row["step"]) for row in run.rows],
            [float(row["learning_rate"]) for row in run.rows],
        )
        for run in runs
    ]

    plot_svg(
        runs,
        validation_series,
        f"{args.title_prefix}: Validation Loss",
        "Validation cross-entropy",
        output_dir / "validation_loss.svg",
        markers=True,
    )
    plot_svg(
        runs,
        train_series,
        f"{args.title_prefix}: Training Loss (moving average, window={args.smoothing_window})",
        "Training cross-entropy",
        output_dir / "training_loss_smoothed.svg",
    )
    plot_svg(
        runs,
        learning_rate_series,
        f"{args.title_prefix}: Learning-Rate Schedule",
        "Learning rate",
        output_dir / "learning_rate_schedule.svg",
        scientific_y=True,
        start_y_at_zero=True,
    )
    write_summary(runs, output_dir / "summary.csv")

    print(f"Wrote {output_dir / 'validation_loss.svg'}")
    print(f"Wrote {output_dir / 'training_loss_smoothed.svg'}")
    print(f"Wrote {output_dir / 'learning_rate_schedule.svg'}")
    print(f"Wrote {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
