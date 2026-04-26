"""
Cross-method comparison plots for the Stage 2 LoRA fine-tune workload.

Reads summary.json + train.log + plots/loss_points.csv from selected runs across
all team members and produces a single bundle of comparison figures + a summary
CSV under `plots/comparison/`.

Methods covered:
  - 1 GPU LoRA baseline           (rassul.magauin, single GPU)
  - DDP                           (youssef.ghallab, 2 nodes x 1 GPU)
  - DeepSpeed ZeRO-2 (best)       (youssef.ghallab, 2 nodes x 1 GPU)
  - DeepSpeed ZeRO-3              (youssef.ghallab, 2 nodes x 1 GPU, timed out)
  - PyTorch FSDP                  (rassul.magauin, 2 nodes x 1 GPU, stopped early)
  - Manual Pipeline (split=12)    (youssef.ghallab, 2 nodes x 1 GPU)

Usage:
    python scripts/plot_method_comparison.py
"""

import json
import re
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "logs" / "runs"
OUT = PROJECT / "plots" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)


# Each entry: short label, base run dir, color, completed-or-not (for bar shading)
METHODS = [
    ("1 GPU LoRA", "rassul.magauin/stage2_1gpu_lora_20260415_174805", "#9aa3b2", True),
    ("DDP",        "youssef.ghallab/stage2_ddp_lora_20260420_000617", "#1f77b4", True),
    ("ZeRO-2",     "youssef.ghallab/stage2_zero2_lora_20260424_134311", "#2ca02c", True),
    ("Pipeline (k=12)", "youssef.ghallab/stage2_pipe_lora_dist_20260424_153422_rank1", "#ff7f0e", True),
    ("ZeRO-3",     "youssef.ghallab/stage2_zero3_lora_20260420_041514", "#d62728", False),
    ("FSDP",       "rassul.magauin/stage2_fsdp_lora_20260426_143948", "#9467bd", False),
]


def load_summary(run_dir: Path):
    """Read summary.json if present, otherwise return a synthetic dict."""
    f = run_dir / "summary.json"
    if f.exists():
        with f.open() as fh:
            return json.load(fh)
    return {}


def parse_loss_steps(train_log: Path):
    """Yield (step_int, loss_float) from train.log, supporting both HF Trainer
    format and the manual pipeline trainer format.

    HF: `{'loss': 1.23, 'learning_rate': 1e-5, 'epoch': 0.0}` + tqdm progress.
    Pipeline: `INFO epoch 1 | step 312/625 | loss 1.05 | lr 5e-5 | 7s elapsed`.
    """
    text = train_log.read_text(errors="ignore")
    text = text.replace("\r", "\n")
    pipe_re = re.compile(r"step\s+(\d+)/\d+\s+\|\s+loss\s+([0-9.]+)")
    loss_re = re.compile(r"\{'loss': ([0-9.]+),")
    step_re = re.compile(r"(\d+)/(\d+) \[")

    losses = []
    last_step = 0
    for line in text.splitlines():
        m = pipe_re.search(line)
        if m:
            losses.append((int(m.group(1)), float(m.group(2))))
            continue
        m = step_re.search(line)
        if m:
            last_step = int(m.group(1))
        m = loss_re.search(line)
        if m:
            losses.append((last_step, float(m.group(1))))
    return losses


def gpu_csv_stats(run_dir: Path):
    """Compute (max_mem_mb, avg_util_pct, runtime_s) from a run's gpu.csv.

    Used as a fallback for runs without summary.json (e.g. ZeRO-3 timeout).
    Format: timestamp, index, name, util.gpu %, util.mem %, mem.used MiB, mem.total MiB
    """
    f = run_dir / "gpu.csv"
    if not f.exists():
        return float("nan"), float("nan"), float("nan")
    util_vals = []
    mem_vals = []
    timestamps = []
    with f.open() as fh:
        next(fh, None)  # header
        for raw in fh:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 7:
                continue
            timestamps.append(parts[0])
            try:
                util_vals.append(float(parts[3].replace("%", "").strip()))
                mem_vals.append(float(parts[5].replace("MiB", "").strip()))
            except ValueError:
                pass
    if not util_vals:
        return float("nan"), float("nan"), float("nan")
    # Crude runtime: count of 5s samples * 5
    runtime = (len(util_vals) - 1) * 5
    return max(mem_vals), sum(util_vals) / len(util_vals), runtime


def loss_curve(run_dir: Path):
    """Prefer plots/loss_points.csv; fall back to scraping train.log."""
    csv_path = run_dir / "plots" / "loss_points.csv"
    if csv_path.exists():
        with csv_path.open() as fh:
            r = csv.DictReader(fh)
            rows = list(r)
        if rows:
            keys = rows[0].keys()
            step_key = "step" if "step" in keys else next(iter(keys))
            loss_key = "loss" if "loss" in keys else [k for k in keys if k != step_key][0]
            return [(int(float(r[step_key])), float(r[loss_key])) for r in rows
                    if r[step_key] not in ("", None) and r[loss_key] not in ("", None)]
    log = run_dir / "train.log"
    if log.exists():
        return parse_loss_steps(log)
    return []


def safe_float(v, default=float("nan")):
    if v in (None, "n/a", "", "nan"):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    rows = []
    loss_curves = {}

    for label, rel, color, completed in METHODS:
        run_dir = RUNS / rel
        s = load_summary(run_dir)
        # Special handling for ZeRO-3 (no summary.json) — extract from train.log + gpu.csv
        if not s and (run_dir / "train.log").exists():
            losses = parse_loss_steps(run_dir / "train.log")
            mem, util, runtime = gpu_csv_stats(run_dir)
            s = {
                "samples_per_sec": "n/a",
                "runtime_s": runtime,
                "max_gpu_mem_mb": mem,
                "avg_gpu_util_pct": util,
                "effective_batch": 8,  # known from launcher
                "last_logged_step": losses[-1][0] if losses else "n/a",
                "final_loss": losses[-1][1] if losses else "n/a",
                "avg_logged_loss": (sum(l for _, l in losses) / len(losses)) if losses else "n/a",
            }
        # If summary lacks loss but train.log has it (e.g. pipeline rank1), backfill
        if (s.get("final_loss") in (None, "n/a", "") or s.get("avg_logged_loss") in (None, "n/a", "")) \
                and (run_dir / "train.log").exists():
            losses = parse_loss_steps(run_dir / "train.log")
            if losses:
                s.setdefault("final_loss", losses[-1][1])
                s["final_loss"] = losses[-1][1]
                s["avg_logged_loss"] = sum(l for _, l in losses) / len(losses)
                s.setdefault("last_logged_step", losses[-1][0])
        rows.append({
            "method": label,
            "completed": completed,
            "effective_batch": s.get("effective_batch", "n/a"),
            "runtime_s": safe_float(s.get("runtime_s")),
            "samples_per_sec": safe_float(s.get("samples_per_sec")),
            "max_gpu_mem_mb": safe_float(s.get("max_gpu_mem_mb")),
            "avg_gpu_util_pct": safe_float(s.get("avg_gpu_util_pct")),
            "last_logged_step": s.get("last_logged_step", "n/a"),
            "final_loss": safe_float(s.get("final_loss")),
            "avg_logged_loss": safe_float(s.get("avg_logged_loss")),
            "color": color,
        })
        loss_curves[label] = loss_curve(run_dir)

    # ZeRO-3 / FSDP throughput from the train.log if summary lacks it
    for r in rows:
        if r["method"] in ("ZeRO-3", "FSDP") and (np.isnan(r["samples_per_sec"]) or r["samples_per_sec"] == 0):
            run_dir = RUNS / next(rel for label, rel, *_ in METHODS if label == r["method"])
            text = (run_dir / "train.log").read_text(errors="ignore").replace("\r", "\n")
            # Extract step rate from last tqdm line
            m = re.findall(r"(\d+)/(\d+) \[(\d+):(\d+)<.*?,\s*([0-9.]+)s/it\]", text)
            if m:
                _, _, _, _, sec_per_it = m[-1]
                eff = r["effective_batch"] if isinstance(r["effective_batch"], (int, float)) else 8
                r["samples_per_sec"] = eff / float(sec_per_it)

    # ---------- Summary CSV ----------
    out_csv = OUT / "method_summary.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "completed", "effective_batch", "runtime_s",
                    "samples_per_sec", "speedup_vs_ddp", "max_gpu_mem_mb",
                    "avg_gpu_util_pct", "last_logged_step", "final_loss",
                    "avg_logged_loss"])
        ddp_sps = next(r["samples_per_sec"] for r in rows if r["method"] == "DDP")
        for r in rows:
            speedup = r["samples_per_sec"] / ddp_sps if ddp_sps else float("nan")
            w.writerow([
                r["method"], r["completed"], r["effective_batch"],
                f"{r['runtime_s']:.0f}" if not np.isnan(r["runtime_s"]) else "n/a",
                f"{r['samples_per_sec']:.3f}" if not np.isnan(r["samples_per_sec"]) else "n/a",
                f"{speedup:.2f}" if not np.isnan(speedup) else "n/a",
                f"{r['max_gpu_mem_mb']:.0f}" if not np.isnan(r["max_gpu_mem_mb"]) else "n/a",
                f"{r['avg_gpu_util_pct']:.1f}" if not np.isnan(r["avg_gpu_util_pct"]) else "n/a",
                r["last_logged_step"],
                f"{r['final_loss']:.3f}" if not np.isnan(r["final_loss"]) else "n/a",
                f"{r['avg_logged_loss']:.3f}" if not np.isnan(r["avg_logged_loss"]) else "n/a",
            ])
    print(f"Wrote {out_csv}")

    # ---------- Bar chart helper ----------
    labels = [r["method"] for r in rows]
    colors = [r["color"] for r in rows]
    completed = [r["completed"] for r in rows]
    edge = ["black" if c else "red" for c in completed]
    hatch = [None if c else "//" for c in completed]

    def bar(ax, values, ylabel, title, fmt="{:.2f}", annotate=True):
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, edgecolor=edge, linewidth=1.5)
        for b, h in zip(bars, hatch):
            if h:
                b.set_hatch(h)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if annotate:
            for xi, v in zip(x, values):
                if not np.isnan(v):
                    ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8)

    # ---------- 1) Throughput ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sps = [r["samples_per_sec"] for r in rows]
    bar(ax, sps, "Throughput (samples/sec)", "Stage 2 LoRA — Throughput by Method", fmt="{:.2f}")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT / "1_throughput.png", dpi=140)
    plt.close(fig)

    # ---------- 2) Wall-clock ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    runtime_min = [r["runtime_s"] / 60 if not np.isnan(r["runtime_s"]) else np.nan for r in rows]
    bar(ax, runtime_min, "Wall-clock (min)", "Stage 2 LoRA — Wall-clock Time (red hatch = stopped early / timed out)", fmt="{:.0f}")
    fig.tight_layout()
    fig.savefig(OUT / "2_wallclock.png", dpi=140)
    plt.close(fig)

    # ---------- 3) Speedup vs DDP ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ddp_sps = next(r["samples_per_sec"] for r in rows if r["method"] == "DDP")
    speedup = [r["samples_per_sec"] / ddp_sps if ddp_sps else np.nan for r in rows]
    bar(ax, speedup, "Throughput / DDP", "Stage 2 LoRA — Speedup vs DDP Baseline", fmt="{:.2f}x")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.6, label="DDP = 1.0x")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "3_speedup_vs_ddp.png", dpi=140)
    plt.close(fig)

    # ---------- 4) Peak GPU memory ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mem_gib = [r["max_gpu_mem_mb"] / 1024 if not np.isnan(r["max_gpu_mem_mb"]) else np.nan for r in rows]
    bar(ax, mem_gib, "Peak GPU memory (GiB)", "Stage 2 LoRA — Peak GPU Memory", fmt="{:.1f}")
    ax.axhline(32.0, color="red", linestyle=":", alpha=0.5, label="RTX 5000 Ada limit (32 GiB)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "4_gpu_memory.png", dpi=140)
    plt.close(fig)

    # ---------- 5) GPU utilization ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    util = [r["avg_gpu_util_pct"] for r in rows]
    bar(ax, util, "Avg GPU utilization (%)", "Stage 2 LoRA — Average GPU Utilization", fmt="{:.0f}%")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(OUT / "5_gpu_utilization.png", dpi=140)
    plt.close(fig)

    # ---------- 6) Loss curves overlay (loss vs step) ----------
    def smooth(xs, k):
        if len(xs) < k:
            return xs
        out = []
        cum = 0.0
        from collections import deque
        win = deque(maxlen=k)
        for v in xs:
            win.append(v)
            out.append(sum(win) / len(win))
        return out

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in rows:
        curve = loss_curves[r["method"]]
        if not curve:
            continue
        steps = [s for s, _ in curve]
        losses = [l for _, l in curve]
        # Window: 5% of curve length, min 5, max 100
        k = max(5, min(100, len(losses) // 20))
        sm = smooth(losses, k)
        ax.plot(steps, sm, label=f"{r['method']} (smoothed, k={k})",
                color=r["color"], alpha=0.95,
                linewidth=1.6 if r["completed"] else 2.0,
                linestyle="-" if r["completed"] else "--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (rolling mean)")
    ax.set_title("Stage 2 LoRA — Loss vs Step (dashed = stopped early / timed out)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0.5, 3.5)
    fig.tight_layout()
    fig.savefig(OUT / "6_loss_vs_step.png", dpi=140)
    plt.close(fig)

    # ---------- 7) Loss vs samples (statistical efficiency) ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in rows:
        curve = loss_curves[r["method"]]
        if not curve:
            continue
        eff = r["effective_batch"]
        if not isinstance(eff, (int, float)) or eff in ("n/a", 0):
            continue
        eff = float(eff)
        samples = [s * eff for s, _ in curve]
        losses = [l for _, l in curve]
        k = max(5, min(100, len(losses) // 20))
        sm = smooth(losses, k)
        ax.plot(samples, sm, label=f"{r['method']} (batch={int(eff)})",
                color=r["color"], alpha=0.95,
                linewidth=1.6 if r["completed"] else 2.0,
                linestyle="-" if r["completed"] else "--")
    ax.set_xlabel("Samples processed")
    ax.set_ylabel("Loss (rolling mean)")
    ax.set_title("Stage 2 LoRA — Loss vs Samples (statistical efficiency)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0.5, 3.5)
    fig.tight_layout()
    fig.savefig(OUT / "7_loss_vs_samples.png", dpi=140)
    plt.close(fig)

    # ---------- 8b) Goodput: loss-drop per second ----------
    # Goodput = (initial_loss - final_loss) / wall_time, in "loss units / s".
    # Higher = more useful learning per second of wall-clock.
    # We use the smoothed first/last 5% of the loss curve to avoid noise.
    goodput_per_sec = []
    goodput_per_sample = []
    for r in rows:
        curve = loss_curves[r["method"]]
        if not curve or np.isnan(r["runtime_s"]):
            goodput_per_sec.append(np.nan)
            goodput_per_sample.append(np.nan)
            continue
        losses = [l for _, l in curve]
        head = sum(losses[: max(1, len(losses) // 20)]) / max(1, len(losses) // 20)
        tail = sum(losses[-max(1, len(losses) // 20):]) / max(1, len(losses) // 20)
        delta = head - tail  # positive = loss decreased
        goodput_per_sec.append(delta / r["runtime_s"])
        eff = r["effective_batch"] if isinstance(r["effective_batch"], (int, float)) else 1
        samples = float(eff) * (r["last_logged_step"] if isinstance(r["last_logged_step"], int) else len(curve))
        goodput_per_sample.append(delta / samples if samples else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    bar(axes[0], goodput_per_sec, "Loss drop per second",
        "Goodput — Loss Reduction per Wall-Clock Second", fmt="{:.5f}")
    axes[0].axhline(0, color="black", linewidth=0.5)
    bar(axes[1], goodput_per_sample, "Loss drop per sample",
        "Goodput — Loss Reduction per Training Sample (statistical efficiency)", fmt="{:.4f}")
    axes[1].axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "9_goodput.png", dpi=140)
    plt.close(fig)

    # Append goodput columns to the summary CSV
    rows_csv = list(csv.reader(open(OUT / "method_summary.csv")))
    rows_csv[0] += ["goodput_loss_per_sec", "goodput_loss_per_sample"]
    for i, (gps, gpsam) in enumerate(zip(goodput_per_sec, goodput_per_sample), start=1):
        rows_csv[i] += [
            f"{gps:.6f}" if not np.isnan(gps) else "n/a",
            f"{gpsam:.6f}" if not np.isnan(gpsam) else "n/a",
        ]
    with (OUT / "method_summary.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows_csv)

    # ---------- 8) Memory–throughput tradeoff ----------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in rows:
        if np.isnan(r["max_gpu_mem_mb"]) or np.isnan(r["samples_per_sec"]):
            continue
        marker = "o" if r["completed"] else "X"
        ax.scatter(r["max_gpu_mem_mb"] / 1024, r["samples_per_sec"],
                   s=200, c=r["color"], edgecolor="black", marker=marker,
                   label=r["method"])
        ax.annotate(r["method"],
                    (r["max_gpu_mem_mb"] / 1024 + 0.2, r["samples_per_sec"] * 1.05),
                    fontsize=9)
    ax.set_xlabel("Peak GPU memory (GiB)")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_yscale("log")
    ax.set_title("Memory–Throughput Tradeoff (X = stopped early)")
    ax.grid(alpha=0.3)
    ax.axvline(32.0, color="red", linestyle=":", alpha=0.5, label="32 GiB limit")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "8_memory_throughput.png", dpi=140)
    plt.close(fig)

    print(f"Wrote 8 plots to {OUT}/")
    print()
    print("Summary table (method | runtime | sps | speedup vs DDP | peak GiB | util | last loss):")
    for r in rows:
        sp = r["samples_per_sec"] / ddp_sps if ddp_sps else float("nan")
        print(f"  {r['method']:>16s}  "
              f"{r['runtime_s']/60:6.1f} min  "
              f"{r['samples_per_sec']:6.3f} sps  "
              f"{sp:5.2f}x  "
              f"{r['max_gpu_mem_mb']/1024:5.1f} GiB  "
              f"{r['avg_gpu_util_pct']:5.1f}%  "
              f"{r['final_loss']:.3f}")


if __name__ == "__main__":
    main()
