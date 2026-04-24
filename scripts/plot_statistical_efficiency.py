#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_summary_value(summary_path: Path, key: str) -> float | None:
    if not summary_path.exists():
        return None
    for line in summary_path.read_text().splitlines():
        if ":" not in line:
            continue
        lhs, rhs = line.split(":", 1)
        if lhs.strip() != key:
            continue
        value = rhs.strip()
        if value == "n/a":
            return None
        return float(value)
    return None


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    smoothed: list[float] = []
    radius = window // 2
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        chunk = values[start:end]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def read_loss_points(loss_points_csv: Path) -> list[dict]:
    with loss_points_csv.open() as f:
        return list(csv.DictReader(f))


def build_efficiency_rows(points: list[dict], throughput: float, smooth_window: int) -> list[dict]:
    rows: list[dict] = []
    for prev, curr in zip(points, points[1:]):
        prev_loss = float(prev["loss"])
        curr_loss = float(curr["loss"])
        prev_samples = float(prev["approx_samples_seen"])
        curr_samples = float(curr["approx_samples_seen"])
        delta_samples = curr_samples - prev_samples
        if delta_samples <= 0:
            continue
        loss_drop = prev_loss - curr_loss
        goodput = loss_drop / delta_samples
        stat_eff = throughput * goodput
        rows.append(
            {
                "step": int(curr["step"]),
                "epoch": float(curr["epoch"]) if curr["epoch"] else None,
                "approx_samples_seen": int(curr_samples),
                "prev_loss": prev_loss,
                "loss": curr_loss,
                "loss_drop_per_sample": goodput,
                "statistical_efficiency": stat_eff,
            }
        )

    smoothed = moving_average([row["statistical_efficiency"] for row in rows], smooth_window)
    for row, smooth_value in zip(rows, smoothed):
        row["smoothed_statistical_efficiency"] = smooth_value
    return rows


def write_rows(rows: list[dict], output_csv: Path) -> None:
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "epoch",
                "approx_samples_seen",
                "prev_loss",
                "loss",
                "loss_drop_per_sample",
                "statistical_efficiency",
                "smoothed_statistical_efficiency",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict], output_png: Path, smooth_window: int, throughput: float) -> None:
    xs = [row["approx_samples_seen"] for row in rows]
    raw = [row["statistical_efficiency"] for row in rows]
    smooth = [row["smoothed_statistical_efficiency"] for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(xs, raw, linewidth=1.0, alpha=0.35, label="Raw")
    plt.plot(xs, smooth, linewidth=2.0, label=f"Smoothed (window={smooth_window})")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel("Approx samples seen")
    plt.ylabel("Statistical efficiency")
    plt.title(f"Statistical efficiency curve (throughput={throughput:.6g} samples/s)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-points", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--smooth-window", type=int, default=7)
    args = parser.parse_args()

    throughput = parse_summary_value(args.summary, "samples_per_sec")
    if throughput is None:
        raise SystemExit(f"Could not read samples_per_sec from {args.summary}")

    points = read_loss_points(args.loss_points)
    if len(points) < 2:
        raise SystemExit("Need at least 2 loss points to compute statistical efficiency")

    rows = build_efficiency_rows(points, throughput, args.smooth_window)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    write_rows(rows, args.output_csv)
    plot_rows(rows, args.output_png, args.smooth_window, throughput)


if __name__ == "__main__":
    main()
