#!/usr/bin/env python3
import csv
import json
import os
import re
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_dataset_size(data_path: Path) -> int | None:
    if not data_path.exists():
        return None
    with open(data_path, "r") as f:
        data = json.load(f)
    return len(data)


def parse_metric_number(raw: str) -> float | None:
    raw = raw.strip()
    if raw.endswith("%"):
        raw = raw[:-1].strip()
    token = raw.split()[0]
    try:
        return float(token)
    except (ValueError, IndexError):
        return None


def format_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.8g}"


def parse_train_log_points(train_log: Path) -> list[dict]:
    if not train_log.exists():
        return []

    points = []
    loss_re = re.compile(r"[\"']loss[\"']\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    step_re = re.compile(r"[\"']step[\"']\s*:\s*([0-9]+)")
    epoch_re = re.compile(r"[\"']epoch[\"']\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    lr_re = re.compile(r"[\"']learning_rate[\"']\s*:\s*([0-9eE\+\-\.]+)")

    for line_no, line in enumerate(train_log.read_text(errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        payload = None
        if stripped.startswith("{") and "loss" in stripped:
            try:
                payload = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                payload = None

        if isinstance(payload, dict) and "loss" in payload:
            try:
                loss = float(payload["loss"])
            except (TypeError, ValueError):
                continue
            step = payload.get("step")
            try:
                step = int(step) if step is not None else None
            except (TypeError, ValueError):
                step = None
            epoch = payload.get("epoch")
            try:
                epoch = float(epoch) if epoch is not None else None
            except (TypeError, ValueError):
                epoch = None
            lr = payload.get("learning_rate")
            try:
                lr = float(lr) if lr is not None else None
            except (TypeError, ValueError):
                lr = None
            points.append(
                {
                    "line_no": line_no,
                    "step": step,
                    "epoch": epoch,
                    "loss": loss,
                    "learning_rate": lr,
                }
            )
            continue

        loss_match = loss_re.search(line)
        if not loss_match:
            continue
        step_match = step_re.search(line)
        epoch_match = epoch_re.search(line)
        lr_match = lr_re.search(line)
        points.append(
            {
                "line_no": line_no,
                "step": int(step_match.group(1)) if step_match else None,
                "epoch": float(epoch_match.group(1)) if epoch_match else None,
                "loss": float(loss_match.group(1)),
                "learning_rate": float(lr_match.group(1)) if lr_match else None,
            }
        )

    for idx, point in enumerate(points, start=1):
        if point["step"] is None:
            point["step"] = idx

    return points


def write_loss_points_csv(points: list[dict], csv_path: Path, effective_batch: int) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["line_no", "step", "epoch", "loss", "learning_rate", "approx_samples_seen"],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "line_no": point["line_no"],
                    "step": point["step"],
                    "epoch": point["epoch"],
                    "loss": point["loss"],
                    "learning_rate": point["learning_rate"],
                    "approx_samples_seen": point["step"] * effective_batch,
                }
            )


def plot_loss_curve(points: list[dict], output_path: Path) -> bool:
    if len(points) < 2:
        return False

    steps = [point["step"] for point in points]
    losses = [point["loss"] for point in points]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Logged step")
    plt.ylabel("Training loss")
    plt.title("Loss curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def parse_gpu_samples(gpu_log: Path) -> dict[str, dict]:
    per_gpu = {}
    if not gpu_log.exists():
        return per_gpu

    rows = gpu_log.read_text(errors="ignore").splitlines()
    for row in rows[1:]:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 7:
            continue
        gpu_index = parts[1]
        gpu_name = parts[2]
        util = parse_metric_number(parts[3])
        mem = parse_metric_number(parts[5])
        gpu_entry = per_gpu.setdefault(
            gpu_index,
            {
                "name": gpu_name,
                "util_vals": [],
                "mem_vals": [],
            },
        )
        if util is not None:
            gpu_entry["util_vals"].append(util)
        if mem is not None:
            gpu_entry["mem_vals"].append(mem)
    return per_gpu


def plot_gpu_series(per_gpu: dict[str, dict], key: str, ylabel: str, title: str, output_path: Path) -> bool:
    if not per_gpu:
        return False

    plt.figure(figsize=(8, 5))
    plotted = False
    for gpu_index in sorted(per_gpu, key=lambda x: int(x)):
        values = per_gpu[gpu_index][key]
        if not values:
            continue
        xs = list(range(1, len(values) + 1))
        plt.plot(xs, values, linewidth=1.5, label=f"GPU {gpu_index}")
        plotted = True

    if not plotted:
        plt.close()
        return False

    plt.xlabel("Monitor sample")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def main():
    method_name = os.environ["METHOD_NAME"]
    run_label = os.environ.get("RUN_LABEL", method_name)
    project_dir = Path(os.environ["PROJECT_DIR"])
    run_user = os.environ.get("RUN_USER", "unknown_user")
    run_id = os.environ.get("RUN_ID", "unknown_run")
    run_dir = Path(os.environ["RUN_DIR"])
    data_path = Path(os.environ["DATA_PATH"])
    train_log = Path(os.environ["TRAIN_LOG"])
    gpu_log = Path(os.environ["GPU_LOG"])
    summary_txt = Path(os.environ["SUMMARY_TXT"])
    summary_json = Path(os.environ["SUMMARY_JSON"])
    summary_csv = Path(os.environ["SUMMARY_CSV"])
    plots_dir = Path(os.environ.get("PLOTS_DIR", run_dir / "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    start_ts = int(os.environ["START_TS"])
    end_ts = int(os.environ["END_TS"])
    runtime_s = max(end_ts - start_ts, 1)
    exit_code = int(os.environ["TRAIN_EXIT"])

    per_device_batch = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = per_device_batch * grad_accum * world_size

    sample_count = read_dataset_size(data_path)

    progress_re = re.compile(r"\[(\d+)/(\d+)\]")
    points = parse_train_log_points(train_log)
    losses = [point["loss"] for point in points]
    last_step = ""
    total_steps = ""
    trainer_metrics = {}
    if train_log.exists():
        for line in train_log.read_text(errors="ignore").splitlines():
            progress_match = progress_re.search(line)
            if progress_match:
                last_step = progress_match.group(1)
                total_steps = progress_match.group(2)
            stripped = line.strip()
            if stripped.startswith("{") and "train_runtime" in stripped:
                try:
                    trainer_metrics = ast.literal_eval(stripped)
                except (SyntaxError, ValueError):
                    pass

    if not last_step and points:
        last_step = str(points[-1]["step"])

    trainer_runtime_value = trainer_metrics.get("train_runtime")
    trainer_runtime_s_float = None
    try:
        trainer_runtime_s_float = float(trainer_runtime_value)
    except (TypeError, ValueError):
        trainer_runtime_s_float = None

    processed_samples = None
    if last_step:
        processed_samples = int(last_step) * effective_batch
    elif sample_count is not None:
        processed_samples = sample_count

    throughput_samples_per_sec = None
    if processed_samples is not None:
        throughput_samples_per_sec = processed_samples / runtime_s

    steps_per_sec_value = None
    if last_step:
        steps_per_sec_value = int(last_step) / runtime_s
    elif sample_count is not None and effective_batch > 0:
        steps_per_sec_value = (sample_count / effective_batch) / runtime_s

    initial_loss_value = losses[0] if losses else None
    final_loss_value = losses[-1] if losses else None
    avg_loss_value = (sum(losses) / len(losses)) if losses else None
    loss_delta_value = None
    goodput_value = None
    statistical_efficiency_value = None
    if initial_loss_value is not None and final_loss_value is not None:
        loss_delta_value = initial_loss_value - final_loss_value
        if processed_samples:
            goodput_value = loss_delta_value / processed_samples
        if throughput_samples_per_sec is not None and goodput_value is not None:
            statistical_efficiency_value = throughput_samples_per_sec * goodput_value

    loss_points_csv = plots_dir / "loss_points.csv"
    write_loss_points_csv(points, loss_points_csv, effective_batch)
    loss_curve_path = plots_dir / "loss_curve.png"
    loss_curve_written = plot_loss_curve(points, loss_curve_path)

    trainer_runtime_s = trainer_metrics.get("train_runtime", "n/a")
    trainer_samples_per_sec = trainer_metrics.get("train_samples_per_second", "n/a")
    trainer_steps_per_sec = trainer_metrics.get("train_steps_per_second", "n/a")
    trainer_train_loss = trainer_metrics.get("train_loss", "n/a")
    trainer_epoch = trainer_metrics.get("epoch", "n/a")

    max_gpu_mem_mb = "n/a"
    avg_gpu_util = "n/a"
    gpu_count_seen = "0"
    per_gpu_metrics_json = "{}"
    if gpu_log.exists():
        per_gpu = parse_gpu_samples(gpu_log)
        mem_vals = []
        util_vals = []
        for gpu_data in per_gpu.values():
            mem_vals.extend(gpu_data["mem_vals"])
            util_vals.extend(gpu_data["util_vals"])
        if mem_vals:
            max_gpu_mem_mb = f"{max(mem_vals):.0f}"
        if util_vals:
            avg_gpu_util = f"{sum(util_vals)/len(util_vals):.1f}"
        if per_gpu:
            gpu_count_seen = str(len(per_gpu))
            per_gpu_summary = {}
            for gpu_index in sorted(per_gpu, key=lambda x: int(x)):
                gpu_data = per_gpu[gpu_index]
                per_gpu_summary[gpu_index] = {
                    "name": gpu_data["name"],
                    "max_mem_mb": round(max(gpu_data["mem_vals"]), 1) if gpu_data["mem_vals"] else None,
                    "avg_util_pct": round(sum(gpu_data["util_vals"]) / len(gpu_data["util_vals"]), 1)
                    if gpu_data["util_vals"] else None,
                }
            per_gpu_metrics_json = json.dumps(per_gpu_summary, sort_keys=True)
        else:
            per_gpu_metrics_json = "{}"

    gpu_util_curve_path = plots_dir / "gpu_utilization_curve.png"
    gpu_mem_curve_path = plots_dir / "gpu_memory_curve.png"
    gpu_util_curve_written = plot_gpu_series(
        per_gpu if gpu_log.exists() else {},
        "util_vals",
        "GPU utilization (%)",
        "GPU utilization over time",
        gpu_util_curve_path,
    )
    gpu_mem_curve_written = plot_gpu_series(
        per_gpu if gpu_log.exists() else {},
        "mem_vals",
        "Memory used (MiB)",
        "GPU memory over time",
        gpu_mem_curve_path,
    )

    rows = [
        ("method", method_name),
        ("exit_code", str(exit_code)),
        ("runtime_s", str(runtime_s)),
        ("samples", str(sample_count) if sample_count is not None else "n/a"),
        ("processed_samples", format_metric(processed_samples)),
        ("effective_batch", str(effective_batch)),
        ("samples_per_sec", format_metric(throughput_samples_per_sec)),
        ("steps_per_sec", format_metric(steps_per_sec_value)),
        ("last_logged_step", last_step or "n/a"),
        ("total_steps", total_steps or "n/a"),
        ("initial_loss", format_metric(initial_loss_value)),
        ("final_loss", format_metric(final_loss_value)),
        ("loss_delta", format_metric(loss_delta_value)),
        ("avg_logged_loss", format_metric(avg_loss_value)),
        ("goodput_loss_drop_per_sample", format_metric(goodput_value)),
        ("statistical_efficiency", format_metric(statistical_efficiency_value)),
        ("trainer_runtime_s", str(trainer_runtime_s)),
        ("trainer_samples_per_sec", str(trainer_samples_per_sec)),
        ("trainer_steps_per_sec", str(trainer_steps_per_sec)),
        ("trainer_train_loss", str(trainer_train_loss)),
        ("trainer_epoch", str(trainer_epoch)),
        ("gpu_count_seen", gpu_count_seen),
        ("max_gpu_mem_mb", max_gpu_mem_mb),
        ("avg_gpu_util_pct", avg_gpu_util),
        ("per_gpu_metrics_json", per_gpu_metrics_json),
        ("run_user", run_user),
        ("run_id", run_id),
        ("run_dir", str(run_dir)),
        ("train_log", str(train_log)),
        ("gpu_log", str(gpu_log)),
        ("loss_points_csv", str(loss_points_csv)),
        ("loss_curve_png", str(loss_curve_path) if loss_curve_written else "n/a"),
        ("gpu_util_curve_png", str(gpu_util_curve_path) if gpu_util_curve_written else "n/a"),
        ("gpu_mem_curve_png", str(gpu_mem_curve_path) if gpu_mem_curve_written else "n/a"),
        ("project_dir", str(project_dir)),
    ]

    with open(summary_txt, "w") as f:
        key_width = max(len(k) for k, _ in rows)
        f.write(f"{run_label} Summary\n")
        f.write("=" * (len(run_label) + 8) + "\n")
        for key, value in rows:
            f.write(f"{key:<{key_width}} : {value}\n")

    with open(summary_json, "w") as f:
        json.dump({k: v for k, v in rows}, f, indent=2)

    csv_header = [k for k, _ in rows]
    csv_row = {k: v for k, v in rows}
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    print("")
    print(f"=== {run_label} Summary ===")
    key_width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key:<{key_width}} | {value}")
    print("")
    print(f"Saved summary text: {summary_txt}")
    print(f"Appended summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
