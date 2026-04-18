#!/usr/bin/env python3
import csv
import json
import os
import re
import ast
from pathlib import Path


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

    start_ts = int(os.environ["START_TS"])
    end_ts = int(os.environ["END_TS"])
    runtime_s = max(end_ts - start_ts, 1)
    exit_code = int(os.environ["TRAIN_EXIT"])

    per_device_batch = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = per_device_batch * grad_accum * world_size

    sample_count = read_dataset_size(data_path)
    samples_per_sec = "n/a"
    steps_per_sec = "n/a"
    if sample_count is not None:
        samples_per_sec = f"{sample_count / runtime_s:.3f}"
        steps_per_sec = f"{(sample_count / effective_batch) / runtime_s:.3f}"

    loss_re = re.compile(r"[\"']loss[\"']\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    step_re = re.compile(r"[\"']step[\"']\s*:\s*([0-9]+)")
    progress_re = re.compile(r"\[(\d+)/(\d+)\]")
    losses = []
    last_step = ""
    total_steps = ""
    trainer_metrics = {}
    if train_log.exists():
        for line in train_log.read_text(errors="ignore").splitlines():
            loss_match = loss_re.search(line)
            if loss_match:
                losses.append(float(loss_match.group(1)))
            step_match = step_re.search(line)
            if step_match:
                last_step = step_match.group(1)
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

    final_loss = f"{losses[-1]:.4f}" if losses else "n/a"
    avg_loss = f"{sum(losses)/len(losses):.4f}" if losses else "n/a"
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
        rows = gpu_log.read_text(errors="ignore").splitlines()
        mem_vals = []
        util_vals = []
        per_gpu = {}
        for row in rows[1:]:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 7:
                gpu_index = parts[1]
                gpu_name = parts[2]
                util = parse_metric_number(parts[3])
                mem = parse_metric_number(parts[5])
                if util is not None:
                    util_vals.append(util)
                if mem is not None:
                    mem_vals.append(mem)
                if gpu_index not in per_gpu:
                    per_gpu[gpu_index] = {
                        "name": gpu_name,
                        "mem_vals": [],
                        "util_vals": [],
                    }
                if mem is not None:
                    per_gpu[gpu_index]["mem_vals"].append(mem)
                if util is not None:
                    per_gpu[gpu_index]["util_vals"].append(util)
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

    rows = [
        ("method", method_name),
        ("exit_code", str(exit_code)),
        ("runtime_s", str(runtime_s)),
        ("samples", str(sample_count) if sample_count is not None else "n/a"),
        ("effective_batch", str(effective_batch)),
        ("samples_per_sec", samples_per_sec),
        ("steps_per_sec", steps_per_sec),
        ("last_logged_step", last_step or "n/a"),
        ("total_steps", total_steps or "n/a"),
        ("final_loss", final_loss),
        ("avg_logged_loss", avg_loss),
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
