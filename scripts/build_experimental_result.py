#!/usr/bin/env python3
"""Build the paper-facing experimental audit from completed artifacts only.

This script is deliberately CPU-only: it parses frozen JSON/CSV artifacts,
recomputes aggregate statistics, and renders tables/figures.  It never imports
model code, opens a checkpoint, or invokes training/inference/teacher stages.
"""

from __future__ import annotations

import csv
import html
import json
import math
import subprocess
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experimental_result.md"
FIG_DIR = ROOT / "reports" / "experimental_result_figures"


def load_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object: {path}")
    return payload


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else math.nan


def pm(values: Iterable[float], scale: float = 100.0, digits: int = 2) -> str:
    values = list(values)
    mean, std = mean_std(values)
    if len(values) == 1:
        return f"{mean * scale:.{digits}f} (n=1)"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f}"


def num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def md_table(headers: list[str], rows: list[list[Any]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["---"] * len(headers)
    rendered = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    rendered.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(rendered)


def save_svg(stem: str, width: int, height: int, body: list[str]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIG_DIR / f"{stem}.svg"
    png_path = FIG_DIR / f"{stem}.png"
    payload = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.axis{font-size:13px}.small{font-size:11px}.grid{stroke:#dbe3ed;stroke-width:1}.base{stroke:#475569;stroke-width:1.2}</style>',
        *body,
        "</svg>",
    ]
    svg_path.write_text("\n".join(payload) + "\n", encoding="utf-8")
    subprocess.run(
        ["convert", "-density", "300", str(svg_path), "-background", "white", "-alpha", "remove", str(png_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def text(x: float, y: float, value: Any, css: str = "axis", anchor: str = "start", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" text-anchor="{anchor}"{transform}>{html.escape(str(value))}</text>'


def line_chart(stem: str, title: str, x_values: list[float], series: list[tuple[str, list[float], str, str]], x_label: str, y_label: str, y_min: float | None = None, y_max: float | None = None) -> None:
    width, height = 1000, 600
    left, right, top, bottom = 95, 35, 70, 95
    plot_w, plot_h = width-left-right, height-top-bottom
    low = min(min(values) for _, values, _, _ in series) if y_min is None else y_min
    high = max(max(values) for _, values, _, _ in series) if y_max is None else y_max
    pad = max((high-low)*.08, .5); low -= 0 if y_min is not None else pad; high += 0 if y_max is not None else pad
    xmin, xmax = min(x_values), max(x_values)
    sx = lambda value: left + (value-xmin)/(xmax-xmin or 1)*plot_w
    sy = lambda value: top + (high-value)/(high-low or 1)*plot_h
    body = [text(width/2, 32, title, "title", "middle")]
    for tick in np.linspace(low, high, 6):
        y = sy(float(tick)); body += [f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" class="grid"/>', text(left-8,y+4,f"{tick:.1f}","small","end")]
    body += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="base"/>', f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="base"/>']
    for value in x_values:
        body.append(text(sx(value),height-bottom+22,f"{value:g}","small","middle"))
    for idx, (name, values, color, dash) in enumerate(series):
        dash = dash or "none"
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x,y in zip(x_values,values))
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
        for x,y in zip(x_values,values): body.append(f'<circle cx="{sx(x)}" cy="{sy(y)}" r="4" fill="{color}"/>')
        body += [f'<line x1="{width-260}" y1="{80+idx*22}" x2="{width-230}" y2="{80+idx*22}" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>', text(width-220,84+idx*22,name,"small")]
    body += [text(width/2,height-20,x_label,"axis","middle"), text(22,height/2,y_label,"axis","middle",-90)]
    save_svg(stem,width,height,body)


MAIN_SOURCES = [
    ("Zero-shot Qwen", "reference", "none", "outputs/stage2_runs/experiment_a/zero_shot/budget_128/seed_42/result.json"),
    ("Random synthetic", "synthetic", "CE", "outputs/stage2_runs/experiment_a/random_synthetic/budget_128/seed_42/result.json"),
    ("Random MNLI", "human MNLI", "CE", "outputs/stage2_runs/experiment_a/random_mnli/budget_128/seed_42/result.json"),
    ("Uncertainty MNLI", "human MNLI", "CE", "outputs/stage2_runs/experiment_a/uncertainty_mnli/budget_128/seed_42/result.json"),
    ("Failure-driven, no verification", "synthetic", "CE", "outputs/stage2_runs/experiment_a/failure_driven_no_verification/budget_128/seed_42/result.json"),
    ("Margin (MNLI)", "human MNLI", "CE", "outputs/stage2_runs_literature/experiment_b/margin_mnli/budget_128/seed_42/result.json"),
    ("Core-set (MNLI)", "human MNLI", "CE", "outputs/stage2_runs_literature/experiment_b/coreset_mnli/budget_128/seed_42/result.json"),
    ("BADGE (MNLI)", "human MNLI", "CE", "outputs/stage2_runs_literature/experiment_b/badge_mnli/budget_128/seed_42/result.json"),
    ("CAL (MNLI)", "human MNLI", "CE", "outputs/stage2_runs_literature/experiment_b/cal_mnli/budget_128/seed_42/result.json"),
    ("Margin (synthetic)", "static verified synthetic", "CE", "outputs/stage2_runs_literature/experiment_b/margin_synthetic/budget_128/seed_42/result.json"),
    ("Core-set (synthetic)", "static verified synthetic", "CE", "outputs/stage2_runs_literature/experiment_b/coreset_synthetic/budget_128/seed_42/result.json"),
    ("BADGE (synthetic)", "static verified synthetic", "CE", "outputs/stage2_runs_literature/experiment_b/badge_synthetic/budget_128/seed_42/result.json"),
    ("CAL (synthetic)", "static verified synthetic", "CE", "outputs/stage2_runs_literature/experiment_b/cal_synthetic/budget_128/seed_42/result.json"),
    ("Verified V7 continuation", "closed-loop synthetic", "CE", "outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/continuation/result.json"),
    ("Verified V7 consolidation", "closed-loop synthetic", "CE", "outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/consolidation/result.json"),
]

EVOLUTION_SOURCES = [
    ("V2 CE", "outputs/stage2_runs_ours_verified_v2_ce/experiment_a/ours_verified/budget_128/seed_42/result.json"),
    ("V2 pairwise", "outputs/stage2_runs_ours_verified_v2_pairwise/experiment_a/ours_verified/budget_128/seed_42/result.json"),
    ("V3 CE", "outputs/stage2_runs_ours_verified_v3_ce/experiment_b/ours_verified/budget_128/seed_42/result.json"),
    ("V5 CE", "outputs/stage2_runs_ours_verified_v5_ce/experiment_b/ours_verified/budget_128/seed_42/result.json"),
    ("V7 consolidation", "outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/consolidation/result.json"),
    ("V7 continuation", "outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/continuation/result.json"),
]


def result_metrics(path: str) -> dict[str, Any]:
    payload = load_json(path)
    hans = payload.get("hans_final") or {}
    mnli = payload.get("mnli") or {}
    return {
        "path": path,
        "payload": payload,
        "hans": hans.get("overall_accuracy"),
        "ent": hans.get("entailment_accuracy"),
        "nonent": hans.get("non_entailment_accuracy"),
        "mnli_m": (mnli.get("matched") or {}).get("accuracy"),
        "mnli_mm": (mnli.get("mismatched") or {}).get("accuracy"),
        "budget": payload.get("actual_final_training_examples", payload.get("actual_training_examples", 0)),
        "status": payload.get("run_status", "complete"),
        "cost": (payload.get("assumed_cost") or {}).get("total"),
    }


def on_off_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("on", "off"):
        for seed in (42, 43, 44):
            result_path = f"outputs/stage2_runs_v7_training_verification_ablation/verification_{mode}/seed_{seed}/result.json"
            diagnostic_path = f"outputs/stage2_diagnostics/v7_hans_prediction_rules/verification_{mode}/seed_{seed}/diagnostic.json"
            result = load_json(result_path)
            diagnostic = load_json(diagnostic_path)
            direct = diagnostic["rounds"]["4"]["metrics"]["direct_binary"]
            resource = result["resource_usage"]
            teacher = resource["teacher"]
            gpu_proxy = sum(float(resource.get(key, 0.0)) for key in (
                "acquisition_gpu_wall_seconds", "training_gpu_wall_seconds",
                "validation_gpu_wall_seconds", "stopping_evaluation_gpu_wall_seconds",
                "final_evaluation_gpu_wall_seconds",
            ))
            states = (result.get("final_hypothesis_audit") or {}).get("states") or {}
            state_counts = {name: sum(item.get("state") == name for item in states.values()) for name in ("certified", "unresolved", "refuted")}
            rows.append({
                "mode": mode.upper(), "seed": seed, "path": result_path,
                "three": result["hans_final"]["overall_accuracy"],
                "three_nonent": result["hans_final"]["non_entailment_accuracy"],
                "direct": direct["overall_accuracy"],
                "direct_nonent": direct["non_entailment_accuracy"],
                "mnli_m": result["mnli"]["matched"]["accuracy"],
                "mnli_mm": result["mnli"]["mismatched"]["accuracy"],
                "tokens": float(teacher["input_tokens"]) + float(teacher["output_tokens"]),
                "calls": float(teacher["calls"]), "gpu_proxy": gpu_proxy,
                **state_counts,
                "learning_curve": result["learning_curve"],
                "heuristics": result["hans_final"]["accuracy_by_heuristic"],
            })
    aggregate: dict[str, Any] = {}
    for mode in ("ON", "OFF"):
        selected = [row for row in rows if row["mode"] == mode]
        aggregate[mode] = {key: mean_std(row[key] for row in selected) for key in (
            "three", "three_nonent", "direct", "direct_nonent", "mnli_m", "mnli_mm", "tokens", "calls", "gpu_proxy"
        )}
    return rows, aggregate


def extended_data() -> list[dict[str, Any]]:
    rows = []
    for seed in (42, 43):
        path = f"outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2/seed_{seed}/result.json"
        payload = load_json(path)
        direct = payload["hans_prediction_rules"]["direct_binary_probability_aggregation"]
        rows.append({
            "seed": seed, "path": path, "budget": payload["actual_training_examples"],
            "three": payload["hans_final"]["overall_accuracy"],
            "three_nonent": payload["hans_final"]["non_entailment_accuracy"],
            "direct": direct["overall_accuracy"], "direct_nonent": direct["non_entailment_accuracy"],
            "mnli_m": payload["mnli"]["matched"]["accuracy"],
            "mnli_mm": payload["mnli"]["mismatched"]["accuracy"],
            "stop_reason": payload["stop_reason"], "learning_curve": payload["learning_curve"],
        })
    return rows


def create_figures(main: list[dict[str, Any]], onoff: list[dict[str, Any]], extended: list[dict[str, Any]], evolution: list[dict[str, Any]], kl_rows: list[dict[str, Any]]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # 1: main comparison.
    selected = sorted(main, key=lambda row: row["hans"])
    colors = {"reference": "#64748b", "human MNLI": "#2563eb", "synthetic": "#f59e0b", "static verified synthetic": "#8b5cf6", "closed-loop synthetic": "#059669"}
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.barh([row["name"] for row in selected], [100 * row["hans"] for row in selected], color=[colors[row["source"]] for row in selected])
    ax.set_xlabel("Sealed HANS accuracy (%)"); ax.set_xlim(63, 68); ax.grid(axis="x", alpha=.2)
    ax.set_title("Budget-128 comparison (seed 42; implemented baselines)")
    save_figure(fig, "01_main_hans_comparison")

    # 2: robustness-retention trade-off.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for row in main:
        ax.scatter(100 * row["mnli_m"], 100 * row["nonent"], s=45, color=colors[row["source"]], alpha=.9)
        if row["name"] in {"Uncertainty MNLI", "Verified V7 continuation", "Core-set (synthetic)", "Failure-driven, no verification"}:
            ax.annotate(row["name"], (100 * row["mnli_m"], 100 * row["nonent"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("MNLI matched accuracy (%)"); ax.set_ylabel("HANS non-entailment accuracy (%)")
    ax.set_title("Robustness–retention trade-off at budget 128"); ax.grid(alpha=.2)
    save_figure(fig, "02_nonent_mnli_tradeoff")

    # 3: stopping-eval curves (not final test).
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for mode, color in (("ON", "#059669"), ("OFF", "#dc2626")):
        subset = [row for row in onoff if row["mode"] == mode]
        budgets = [0, 32, 64, 96, 128]
        matrix = np.array([[item["metrics"]["overall_accuracy"] for item in row["learning_curve"]] for row in subset]) * 100
        ax.plot(budgets, matrix.mean(axis=0), marker="o", color=color, label=f"Verification {mode}")
        ax.fill_between(budgets, matrix.mean(axis=0)-matrix.std(axis=0, ddof=1), matrix.mean(axis=0)+matrix.std(axis=0, ddof=1), color=color, alpha=.12)
    for row in extended:
        ax.plot([32 * item["round"] for item in row["learning_curve"]], [100 * item["metrics"]["overall_accuracy"] for item in row["learning_curve"]], linestyle="--", alpha=.38, label=f"Extended seed {row['seed']}")
    ax.set_xlabel("Cumulative acquired examples"); ax.set_ylabel("Stopping-eval accuracy (%)")
    ax.set_title("Learning curves on the fixed 900-example stopping subset")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.2)
    save_figure(fig, "03_learning_curves")

    # 4: ON/OFF rule-specific final test.
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    labels = ["Three-way→binary", "Direct binary"]
    x = np.arange(2); width = .34
    for index, (mode, color) in enumerate((("ON", "#059669"), ("OFF", "#dc2626"))):
        subset = [row for row in onoff if row["mode"] == mode]
        means = [100 * statistics.fmean(row[key] for row in subset) for key in ("three", "direct")]
        stds = [100 * statistics.stdev(row[key] for row in subset) for key in ("three", "direct")]
        ax.bar(x + (index-.5)*width, means, width, yerr=stds, capsize=3, color=color, label=mode)
    ax.set_xticks(x, labels); ax.set_ylabel("Sealed HANS accuracy (%)"); ax.set_ylim(67, 76)
    ax.set_title("Verification ablation is prediction-rule dependent (3 seeds)"); ax.legend(); ax.grid(axis="y", alpha=.2)
    save_figure(fig, "04_verification_ablation")

    # 5: historical evolution.
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(evolution)); ax.plot(x, [100*r["hans"] for r in evolution], marker="o", label="HANS overall")
    ax.plot(x, [100*r["nonent"] for r in evolution], marker="s", label="HANS non-entailment")
    ax.set_xticks(x, [r["name"] for r in evolution], rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Historical V2–V7 evolution (seed 42; protocol changes)")
    ax.legend(); ax.grid(alpha=.2)
    save_figure(fig, "05_v2_v7_evolution")

    # 6: resource comparison plus adaptive pilot compute.
    adaptive = load_json("outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/simulation.json")["summaries"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for index, mode in enumerate(("ON", "OFF")):
        subset = [row for row in onoff if row["mode"] == mode]
        axes[0].bar(index, statistics.fmean(row["tokens"] for row in subset)/1e3, color=("#059669" if mode=="ON" else "#dc2626"))
    axes[0].set_xticks([0,1],["ON","OFF"]); axes[0].set_ylabel("Teacher tokens (thousands, mean)"); axes[0].set_title("Verification teacher overhead")
    for index, seed in enumerate((42,43)):
        fixed = next(row for row in adaptive if row["seed"]==seed and row["strategy"]=="fixed_32")
        adapt = next(row for row in adaptive if row["seed"]==seed and row["strategy"]=="adaptive")
        axes[1].bar(index-.18, fixed["pilot_gpu_hours_used"], .36, label="Fixed-32" if index==0 else None, color="#64748b")
        axes[1].bar(index+.18, adapt["pilot_gpu_hours_used"], .36, label="Adaptive" if index==0 else None, color="#8b5cf6")
    axes[1].set_xticks([0,1],["Seed 42","Seed 43"]); axes[1].set_ylabel("Measured pilot GPU wall-hours"); axes[1].set_title("Retrospective adaptive-pilot simulation"); axes[1].legend()
    fig.tight_layout(); save_figure(fig, "06_resource_cost_breakdown")

    # 7: KL radii and projected-cost numerical decisions.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    labels = [f"{row['seed']}-R{row['round']}" for row in kl_rows]; x = np.arange(len(labels))
    axes[0].plot(x, [row["hoeffding_radius"] for row in kl_rows], marker="o", label="Range-aware Hoeffding")
    axes[0].plot(x, [row["kl_radius"] for row in kl_rows], marker="s", label="KL-Hoeffding")
    axes[0].set_xticks(x, labels, rotation=45, ha="right"); axes[0].set_ylabel("Confidence radius"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].bar(x, [row["projected_kl_net_ucb"] for row in kl_rows], color=["#dc2626" if row["projected_kl_net_ucb"] <= 0 else "#2563eb" for row in kl_rows])
    axes[1].axhline(0, color="black", linewidth=.8); axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].set_ylabel("KL Net-UCB using projected cost"); axes[1].set_title("Numerical only; no certified STOP")
    fig.tight_layout(); save_figure(fig, "07_kl_hoeffding_stopping")


def create_figures(main: list[dict[str, Any]], onoff: list[dict[str, Any]], extended: list[dict[str, Any]], evolution: list[dict[str, Any]], kl_rows: list[dict[str, Any]]) -> None:
    """Dependency-light SVG renderer; PNG conversion uses ImageMagick."""
    palette = {"reference":"#64748b","human MNLI":"#2563eb","synthetic":"#f59e0b","static verified synthetic":"#8b5cf6","closed-loop synthetic":"#059669"}

    # Figure 1: horizontal ranking.
    ordered = sorted(main, key=lambda row: row["hans"])
    width, height = 1100, 720; left, right, top, bottom = 270, 45, 60, 55
    low, high = 63.0, 67.5; plot_w = width-left-right; row_h = (height-top-bottom)/len(ordered)
    body = [text(width/2,30,"Budget-128 sealed-HANS comparison (seed 42)","title","middle")]
    for tick in np.arange(63,68):
        x = left+(tick-low)/(high-low)*plot_w; body += [f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height-bottom}" class="grid"/>',text(x,height-22,f"{tick:.0f}%","small","middle")]
    for idx,row in enumerate(ordered):
        y=top+idx*row_h+4; bar=(100*row["hans"]-low)/(high-low)*plot_w
        body += [text(left-8,y+row_h*.55,row["name"],"small","end"),f'<rect x="{left}" y="{y}" width="{max(bar,1)}" height="{row_h*.68}" rx="2" fill="{palette[row["source"]]}"/>',text(left+bar+5,y+row_h*.52,f"{100*row['hans']:.2f}","small")]
    save_svg("01_main_hans_comparison",width,height,body)

    # Figure 2: trade-off scatter.
    width,height=900,590; left,right,top,bottom=85,40,65,75; xmin,xmax=57,62; ymin,ymax=33,43.5
    sx=lambda v:left+(v-xmin)/(xmax-xmin)*(width-left-right); sy=lambda v:top+(ymax-v)/(ymax-ymin)*(height-top-bottom)
    body=[text(width/2,31,"HANS non-entailment vs MNLI retention","title","middle")]
    for tick in np.arange(57,63): body += [f'<line x1="{sx(tick)}" y1="{top}" x2="{sx(tick)}" y2="{height-bottom}" class="grid"/>',text(sx(tick),height-42,f"{tick}%","small","middle")]
    for tick in np.arange(34,44,2): body += [f'<line x1="{left}" y1="{sy(tick)}" x2="{width-right}" y2="{sy(tick)}" class="grid"/>',text(left-8,sy(tick)+4,f"{tick}%","small","end")]
    annotate={"Uncertainty MNLI","Verified V7 continuation","Core-set (synthetic)","Failure-driven, no verification"}
    for row in main:
        x,y=sx(100*row["mnli_m"]),sy(100*row["nonent"]); body.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{palette[row["source"]]}"/>')
        if row["name"] in annotate: body.append(text(x+7,y-6,row["name"],"small"))
    body += [text(width/2,height-12,"MNLI matched accuracy (%)","axis","middle"),text(20,height/2,"HANS non-entailment (%)","axis","middle",-90)]
    save_svg("02_nonent_mnli_tradeoff",width,height,body)

    # Figure 3: mean ON/OFF and individual extended stopping curves.
    series=[]
    for mode,color in (("ON","#059669"),("OFF","#dc2626")):
        subset=[r for r in onoff if r["mode"]==mode]
        vals=[100*statistics.fmean(r["learning_curve"][i]["metrics"]["overall_accuracy"] for r in subset) for i in range(5)]
        series.append((f"Verification {mode}",vals,color,""))
    # Render all series on 0..256 with missing later values held out manually.
    width,height=1000,600; left,right,top,bottom=90,40,70,80; low,high=60,77; sx=lambda v:left+v/256*(width-left-right); sy=lambda v:top+(high-v)/(high-low)*(height-top-bottom)
    body=[text(width/2,32,"Stopping-eval learning curves (not final test)","title","middle")]
    for tick in range(0,257,32): body.append(text(sx(tick),height-43,str(tick),"small","middle"))
    for tick in (60,64,68,72,76): body += [f'<line x1="{left}" y1="{sy(tick)}" x2="{width-right}" y2="{sy(tick)}" class="grid"/>',text(left-7,sy(tick)+4,f"{tick}%","small","end")]
    all_series=[(name,list(range(0,129,32)),vals,color,"") for name,vals,color,_ in series]
    for row,color in zip(extended,("#6366f1","#a855f7")):
        all_series.append((f"Extended seed {row['seed']}",[32*i["round"] for i in row["learning_curve"]],[100*i["metrics"]["overall_accuracy"] for i in row["learning_curve"]],color,"8 5"))
    for idx,(name,xs,vals,color,dash) in enumerate(all_series):
        dash = dash or "none"
        points=" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x,y in zip(xs,vals)); body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
        for x,y in zip(xs,vals): body.append(f'<circle cx="{sx(x)}" cy="{sy(y)}" r="4" fill="{color}"/>')
        body += [f'<line x1="{width-260}" y1="{82+idx*21}" x2="{width-230}" y2="{82+idx*21}" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>',text(width-220,86+idx*21,name,"small")]
    body += [text(width/2,height-12,"Cumulative acquired examples","axis","middle"),text(20,height/2,"Accuracy (%)","axis","middle",-90)]
    save_svg("03_learning_curves",width,height,body)

    # Figure 4: rule-specific ON/OFF means.
    width,height=820,520; left,right,top,bottom=90,35,70,80; low,high=67,76; pw=(width-left-right)/2
    body=[text(width/2,31,"Verification ablation by HANS prediction rule","title","middle")]
    for tick in range(67,77):
        y=top+(high-tick)/(high-low)*(height-top-bottom); body += [f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" class="grid"/>',text(left-7,y+4,str(tick),"small","end")]
    for gi,(label,key) in enumerate((("Three-way→binary","three"),("Direct binary","direct"))):
        center=left+pw*(gi+.5); body.append(text(center,height-42,label,"axis","middle"))
        for offset,(mode,color) in zip((-.19,.19),( ("ON","#059669"),("OFF","#dc2626") )):
            value=100*statistics.fmean(r[key] for r in onoff if r["mode"]==mode); y=top+(high-value)/(high-low)*(height-top-bottom)
            body += [f'<rect x="{center+offset*pw-48}" y="{y}" width="96" height="{height-bottom-y}" fill="{color}"/>',text(center+offset*pw,y-6,f"{mode} {value:.2f}","small","middle")]
    save_svg("04_verification_ablation",width,height,body)

    # Figure 5: evolution.
    line_chart("05_v2_v7_evolution","Historical V2–V7 evolution (seed 42)",list(range(len(evolution))),[("HANS overall",[100*r["hans"] for r in evolution],"#2563eb",""),("Non-entailment",[100*r["nonent"] for r in evolution],"#f59e0b","")],"Protocol version index","Accuracy (%)",32,67)

    # Figure 6: two-panel resources.
    adaptive=load_json("outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/simulation.json")["summaries"]
    width,height=1000,520; body=[text(width/2,31,"Resource diagnostics","title","middle")]
    # Tokens.
    for idx,(mode,color) in enumerate((("ON","#059669"),("OFF","#dc2626"))):
        value=statistics.fmean(r["tokens"] for r in onoff if r["mode"]==mode)/1000; x=130+idx*160; h=value/1100*330
        body += [f'<rect x="{x}" y="{420-h}" width="95" height="{h}" fill="{color}"/>',text(x+47,440,mode,"axis","middle"),text(x+47,410-h,f"{value:.0f}k","small","middle")]
    body.append(text(220,480,"Mean teacher tokens","axis","middle"))
    # Pilot hours.
    for si,seed in enumerate((42,43)):
        for oi,(strategy,color) in enumerate((("fixed_32","#64748b"),("adaptive","#8b5cf6"))):
            value=next(r["pilot_gpu_hours_used"] for r in adaptive if r["seed"]==seed and r["strategy"]==strategy); x=580+si*180+oi*60; h=value/14*330
            body += [f'<rect x="{x}" y="{420-h}" width="48" height="{h}" fill="{color}"/>',text(x+24,410-h,f"{value:.1f}h","small","middle")]
        body.append(text(640+si*180,440,f"Seed {seed}","axis","middle"))
    body += [text(725,480,"Pilot GPU wall-hours: fixed (gray) vs adaptive (purple)","axis","middle"),f'<line x1="500" y1="65" x2="500" y2="475" class="grid"/>']
    save_svg("06_resource_cost_breakdown",width,height,body)

    # Figure 7: KL comparison and projected net values.
    width,height=1200,560; body=[text(width/2,31,"KL-Hoeffding: tighter radii, numerical diagnostics only","title","middle")]; labels=[f"{r['seed']}-R{r['round']}" for r in kl_rows]
    for idx,row in enumerate(kl_rows):
        x=70+idx*48; scale=950
        body += [f'<rect x="{x}" y="{385-row["hoeffding_radius"]*scale}" width="16" height="{row["hoeffding_radius"]*scale}" fill="#94a3b8"/>',f'<rect x="{x+17}" y="{385-row["kl_radius"]*scale}" width="16" height="{row["kl_radius"]*scale}" fill="#2563eb"/>',text(x+16,405,labels[idx],"small","middle",-55)]
        net=row["projected_kl_net_ucb"]; nx=680+idx*43; zero=300; nh=net*700
        body.append(f'<rect x="{nx}" y="{min(zero,zero-nh)}" width="28" height="{abs(nh)}" fill="{"#dc2626" if net<=0 else "#059669"}"/>')
    body += [text(285,500,"Confidence radius: Range (gray), KL (blue)","axis","middle"),text(925,500,"KL Net-UCB using projected cost","axis","middle"),f'<line x1="650" y1="55" x2="650" y2="500" class="grid"/>',f'<line x1="675" y1="300" x2="1170" y2="300" class="base"/>']
    save_svg("07_kl_hoeffding_stopping",width,height,body)


def build() -> tuple[str, dict[str, Any]]:
    main = []
    for name, source, objective, path in MAIN_SOURCES:
        row = result_metrics(path); row.update(name=name, source=source, objective=objective)
        if row["hans"] is None:
            raise ValueError(f"Missing final HANS: {path}")
        main.append(row)
    evolution = []
    for name, path in EVOLUTION_SOURCES:
        row = result_metrics(path); row["name"] = name; evolution.append(row)
    onoff, onoff_agg = on_off_data()
    extended = extended_data()
    kl_payload = load_json("outputs/stage2_diagnostics/v7_kl_hoeffding/replay.json")
    kl_rows = kl_payload["per_round"]
    projected_cost = 0.8983888888888889
    projected_penalty = 0.2 * projected_cost
    for row in kl_rows:
        row["projected_kl_net_ucb"] = row["kl_gain_ucb"] - projected_penalty
        row["projected_numerical_stop"] = row["projected_kl_net_ucb"] <= 0
    create_figures(main, onoff, extended, evolution, kl_rows)

    target_rows = [result_metrics(f"outputs/stage2_runs_ours_verified_target_only/experiment_a/ours_verified/budget_128/seed_{seed}/result.json") for seed in (42,43,44)]
    target_hans = [r["hans"] for r in target_rows]
    first_numerical = {}
    for seed in (42,43):
        candidates = [r for r in kl_rows if r["seed"] == seed and r["projected_numerical_stop"]]
        first_numerical[seed] = min((r["round"] for r in candidates), default=None)
    numerical_rows = [r for r in kl_rows if r["projected_numerical_stop"]]

    main_rows = [[r["name"], r["source"], r["objective"], r["budget"], "42", pct(r["hans"]), pct(r["nonent"]), pct(r["mnli_m"]), pct(r["mnli_mm"]), "complete"] for r in main]
    onoff_rows = []
    for mode in ("ON", "OFF"):
        subset = [r for r in onoff if r["mode"] == mode]
        onoff_rows.append([f"V7 early-stop Verification {mode}", "closed-loop synthetic", "CE + epoch early stopping", 128, "42/43/44", pm(r["three"] for r in subset), pm(r["three_nonent"] for r in subset), pm(r["mnli_m"] for r in subset), pm(r["mnli_mm"] for r in subset), "complete"])
    extended_rows = [[f"V7 extended seed {r['seed']}", "closed-loop synthetic + pilots", 256, r["seed"], pct(r["three"]), pct(r["direct"]), pct(r["three_nonent"]), pct(r["mnli_m"]), r["stop_reason"]] for r in extended]
    evolution_rows = [[r["name"], pct(r["hans"]), pct(r["nonent"]), pct(r["mnli_m"]), pct(r["mnli_mm"]), r["path"]] for r in evolution]
    objective = load_json("reports/stage2_ours_verified_v2_ablation/seed_42_budget_128/objective_ablation.json")
    objective_rows = [[item["metric"], pct(item["ce"]), pct(item["pairwise"]), f"{100*item['pairwise_minus_ce']:+.04f} pp"] for item in objective["metrics"]]
    ver_rows = [[r["mode"], r["seed"], pct(r["three"]), pct(r["direct"]), pct(r["three_nonent"]), pct(r["mnli_m"]), f"{r['tokens']:.0f}", f"{r['calls']:.0f}"] for r in onoff]
    state_rows = [[r["seed"], r["certified"], r["unresolved"], r["refuted"]] for r in onoff if r["mode"] == "ON"]
    heuristic_rows = []
    zero = next(r for r in main if r["name"] == "Zero-shot Qwen")["payload"]["hans_final"]["accuracy_by_heuristic"]
    for heuristic in ("lexical_overlap", "subsequence", "constituent"):
        heuristic_rows.append([heuristic.replace("_", " "), pct(zero[heuristic]), pm(r["heuristics"][heuristic] for r in onoff if r["mode"]=="ON"), pm(r["heuristics"][heuristic] for r in onoff if r["mode"]=="OFF")])
    kl_table = [[r["seed"], r["round"], f"{r['pilot_mean_gain']:.6f}", f"{r['hoeffding_radius']:.6f}", f"{r['kl_radius']:.6f}", f"{r['radius_reduction_percent']:.2f}%", f"{r['projected_kl_net_ucb']:.6f}", "yes" if r["projected_numerical_stop"] else "no", "no"] for r in kl_rows]

    on_three = onoff_agg["ON"]["three"][0]; off_three = onoff_agg["OFF"]["three"][0]
    on_direct = onoff_agg["ON"]["direct"][0]; off_direct = onoff_agg["OFF"]["direct"][0]
    strongest = max(main, key=lambda r: r["hans"])
    best_synth = max((r for r in main if "synthetic" in r["source"]), key=lambda r: r["hans"])
    extended_three = [r["three"] for r in extended]
    adaptive = load_json("outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/simulation.json")["summaries"]
    radius_summary = kl_payload["summary"]["radius_reduction_percent"]

    report = f"""# Experimental Results Audit: Verified-Failure-Driven Active Learning (V7)

> **Artifact-only paper audit.** Generated from completed repository artifacts on the current `feat/v7-minimal-cost-aware` branch. No GPU, model inference, Teacher API, training, or sealed-test retuning was performed. Percentages are accuracy ×100; differences are percentage points (pp). `±` is the sample standard deviation across seeds; single-seed rows are explicitly marked rather than assigned a zero standard deviation.

## Executive evidence ledger

This audit found **30 unique completed result artifacts spanning 23 method/protocol labels**: {len(main)} core budget-128 comparison rows, six controlled Verification ON/OFF runs, three target-only verified runs, four additional V2/V3/V5 objective/evolution runs, and two completed 256-example extended trajectories. It distinguishes 15,000-example sealed HANS evaluation from the older 30,000-example baseline benchmark. Raw `result.json` takes precedence over stale prose: notably, all eight implemented Margin/Core-set/BADGE/CAL MNLI/synthetic runs are now complete.

The strongest single-seed budget-128 result in the common table is **{strongest['name']} ({pct(strongest['hans'])}% HANS)**, whereas the strongest synthetic row is **{best_synth['name']} ({pct(best_synth['hans'])}%)**. The newer three-seed early-stopping V7 reaches `{pm(r['three'] for r in onoff if r['mode']=='ON')}` with verification ON and `{pm(r['three'] for r in onoff if r['mode']=='OFF')}` with verification OFF under the current three-way evaluator. The two extended 256-example runs reach `{pm(extended_three)}` but terminate at maximum budget, not a certified Cost-Aware STOP.

## A. Experimental setup

**Research question.** Can confidence-screened, failure-directed synthetic acquisition improve HANS robustness while retaining MNLI performance under a strict evaluation firewall?

The student is Qwen3-1.7B with LoRA (`r=8`, `alpha=16`) and AdamW; the Teacher is Qwen3-14B. The canonical fixed-budget protocol uses four rounds ×32 examples (128 total), while the extended Cost-Aware trajectories allow eight rounds (256 total). Training uses three-way NLI labels; HANS is binary, mapping neutral and contradiction to non-entailment. The current evaluator predicts the three-way argmax then maps to binary; Direct Binary instead compares `P(E)` with `P(N)+P(C)` and is reported as a diagnostic.

The frozen 30,000 HANS corpus is partitioned into discovery (3,750), verification (3,750), intervention pool (3,750), gain evaluation (3,750), and sealed final test (15,000). Gain evaluation is further split into a 900-example `stopping_eval` subset and a 2,850-example held-out report subset. `stopping_eval` supports epoch-level early stopping; `gain_eval`/held-out reporting and sealed HANS are not used for training selection or Cost-Aware tuning. MNLI matched/mismatched are retention views. The separate historical HANS baseline benchmark evaluated all 30,000 records and is therefore not numerically interchangeable with sealed-HANS results.

| Component | Frozen protocol |
|---|---|
| Student / Teacher | Qwen3-1.7B LoRA / Qwen3-14B |
| Standard / extended budget | 128 (4×32) / 256 (8×32) examples |
| Primary robustness metric | sealed HANS 15k overall and non-entailment accuracy |
| Retention | MNLI matched / mismatched |
| Online model-selection view | HANS `stopping_eval`, 900 examples |
| Held-out views | 2,850 gain report; sealed HANS 15k; MNLI |
| Main seeds | historical comparisons: 42; ON/OFF: 42/43/44; extended: 42/43 |
| Firewall | no verification probe, stopping row, gain-report row, or sealed-test row enters the training loss |

Provenance: `flow_v7.md`, `README.md`, frozen manifests under `outputs/stage2_shared/`, and resolved run configs.

## B. Main results: comparison with existing methods

**Research question.** How does V7 compare with reproduced active-learning selectors under budget 128?

The published selectors below are **implemented/reproduced acquisition baselines in this codebase**, not numbers copied from the original papers. Human-MNLI rows have different data access from HANS-guided synthetic rows; they are diagnostic cross-source comparisons rather than equal-access controls.

{md_table(['Method','Acquisition source','Objective','Budget','Seed(s)','HANS','HANS non-ent.','MNLI-m','MNLI-mm','Status'], main_rows + onoff_rows)}

![Main HANS comparison](reports/experimental_result_figures/01_main_hans_comparison.svg)

*Figure 1: Sealed-HANS 15k accuracy for completed seed-42 budget-128 implemented baselines. The figure deliberately excludes the newer multi-seed early-stopping rows to avoid encoding unequal uncertainty as a single ranking.*

At seed 42, uncertainty sampling from MNLI exceeds historical V7 continuation by `{100*(next(r for r in main if r['name']=='Uncertainty MNLI')['hans']-next(r for r in main if r['name']=='Verified V7 continuation')['hans']):.4f}` pp overall and `{100*(next(r for r in main if r['name']=='Uncertainty MNLI')['nonent']-next(r for r in main if r['name']=='Verified V7 continuation')['nonent']):.4f}` pp on non-entailment. Core-set and CAL on the static verified synthetic pool are within 0.06 pp of historical V7 continuation, which is not evidence of a meaningful ordering from one seed. Failure-driven acquisition without verification is 0.4667 pp above historical V7 continuation; verification is therefore not uniformly beneficial in the old protocol.

![Robustness–retention trade-off](reports/experimental_result_figures/02_nonent_mnli_tradeoff.svg)

*Figure 2: HANS non-entailment versus MNLI matched accuracy. Human-MNLI access shifts both axes and must be disclosed.*

### Extended 256-example trajectories (separate budget)

{md_table(['Run','Source','Budget','Seed','HANS three-way','HANS direct','HANS non-ent.','MNLI-m','Terminal reason'], extended_rows)}

The two extended runs average `{pm(r['three'] for r in extended)}` three-way HANS and `{pm(r['direct'] for r in extended)}` Direct Binary. This is a strong completed result but not directly rankable against budget-128 methods. Both used all 256 examples and ended with `maximum_budget_reached_with_remaining_hypotheses`; they did not demonstrate Cost-Aware early stopping.

### Historical HANS-30k benchmark (not the sealed 15k protocol)

| Method | Budget | HANS-30k overall | Non-entailment | Lexical | Subsequence | Constituent |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0 | 63.6367 | 33.5067 | 82.9900 | 57.3200 | 50.6000 |
| Random | 32 | 63.7967 | 33.9200 | 83.3300 | 57.4200 | 50.6400 |
| Uncertainty | 32 | 64.4833 | 35.8067 | 84.3000 | 58.2200 | 50.9300 |

These early 30k results establish a baseline trend but cannot be merged with the sealed-15k tables because both evaluation membership and acquisition budget differ. Source: `reports/hans_baselines_seed42/data/run_summary.csv`.

## C. Ablation studies

### C.1 Verification ON versus OFF (controlled)

**Research question.** Does hypothesis verification improve V7 when training policy, seeds, selector, and budget are fixed?

{md_table(['Mode','Seed','HANS 3-way','HANS direct','Non-ent. 3-way','MNLI-m','Teacher tokens','Calls'], ver_rows)}

![Verification ablation](reports/experimental_result_figures/04_verification_ablation.svg)

*Figure 3: Mean ± sample standard deviation over seeds 42–44 on sealed HANS. The conclusion changes with the prediction rule.*

Under the current three-way-to-binary evaluator, ON averages `{100*on_three:.4f}%` and OFF `{100*off_three:.4f}%`, so ON is `{100*(on_three-off_three):+.4f}` pp lower. Under Direct Binary, ON averages `{100*on_direct:.4f}%` and OFF `{100*off_direct:.4f}%`, making ON `{100*(on_direct-off_direct):+.4f}` pp higher. The controlled ablation therefore does not support a rule-independent claim that verification improves accuracy. OFF also retains MNLI better on average (`{pm(r['mnli_m'] for r in onoff if r['mode']=='OFF')}` versus `{pm(r['mnli_m'] for r in onoff if r['mode']=='ON')}`).

### C.2 Verified versus unverified and V2–V7 evolution

**Research question.** Which improvements survive historical protocol changes?

{md_table(['Version','HANS','Non-ent.','MNLI-m','MNLI-mm','Raw source'], evolution_rows)}

![V2–V7 evolution](reports/experimental_result_figures/05_v2_v7_evolution.svg)

*Figure 4: Historical seed-42 evolution. This is a cross-protocol comparison, not a controlled component ablation.*

V2 CE to V7 continuation improves HANS by `{100*(evolution[-1]['hans']-evolution[0]['hans']):.4f}` pp and non-entailment by `{100*(evolution[-1]['nonent']-evolution[0]['nonent']):.4f}` pp. Continuation exceeds consolidation by `{100*(evolution[-1]['hans']-evolution[-2]['hans']):.4f}` pp using the same selected 128 examples, but only seed 42 is available. V4 has no distinct completed result; V6 is a preserved checkpoint-continuity failure and is excluded from performance claims.

### C.3 CE versus pairwise objective (controlled V2)

**Research question.** Does the implemented pairwise objective help when selected IDs are identical?

{md_table(['Metric','CE','Pairwise','Pairwise − CE'], objective_rows)}

The 128 selected IDs are identical and in the same order. Pairwise is 0.0400 pp lower on HANS overall and 0.0534 pp lower on non-entailment; the differences are tiny and single-seed. The report does not relabel CE runs as pairwise merely because pair structure was used during selection.

### C.4 Target-only verification and lifecycle

The early target-only protocol over seeds 42–44 yields `{pm(target_hans)}` HANS. In the newer Verification-ON final audits, hypothesis-state counts are:

{md_table(['Seed','Certified','Unresolved','Refuted'], state_rows)}

Unresolved hypotheses remain distinct from refuted hypotheses and can be retained for additional evidence. This is a protocol property, not evidence that every retained hypothesis is useful.

## D. Data efficiency and learning curves

**Research question.** How does performance evolve with cumulative acquisition, and is improvement monotonic?

![Learning curves](reports/experimental_result_figures/03_learning_curves.svg)

*Figure 5: Accuracy on the fixed 900-example stopping subset. Solid lines are means across seeds 42–44; dashed extended trajectories are individual seeds. These are not sealed-test curves.*

Both ON and OFF share their initial 32-example state within each seed, then diverge. The trajectories are non-monotonic: ON seed 42 peaks at 71.0% at 96 examples and falls to 66.67% at 128, while extended seed 42 falls at 160 before recovering at 224. This supports reporting every round rather than only the best retrospective checkpoint. It does not authorize selecting a checkpoint on sealed HANS. MNLI retention and final HANS are only available at finalized checkpoints, so they are not interpolated into this curve.

## E. HANS heuristic and failure-type analysis

**Research question.** Are gains distributed across HANS heuristics and labels?

{md_table(['Heuristic','Zero-shot seed 42','Verification ON (3 seeds)','Verification OFF (3 seeds)'], heuristic_rows)}

The label breakdown reveals the central trade-off: zero-shot is entailment-heavy (93.85% entailment, 33.75% non-entailment), whereas later V7 checkpoints substantially improve non-entailment. The three-way ON mean non-entailment is `{pm(r['three_nonent'] for r in onoff if r['mode']=='ON')}` and OFF is `{pm(r['three_nonent'] for r in onoff if r['mode']=='OFF')}`. Per-subcase artifacts exist for all sealed evaluations, but a 30-subcase ranking is placed in the appendix recommendation because multiple comparisons and seed variance make compact main-text claims fragile.

## F. Cost, resource efficiency, and stopping diagnostics

**Research question.** What resource overhead is measured, and does any confidence method establish an online STOP?

Verification ON uses `{pm((r['tokens'] for r in onoff if r['mode']=='ON'), scale=1.0, digits=0)}` teacher tokens versus `{pm((r['tokens'] for r in onoff if r['mode']=='OFF'), scale=1.0, digits=0)}` for OFF. These are run-specific teacher token totals; shared initial-pool costs are excluded. GPU wall-clock fields are proxies, not hardware-utilization integrals, and the seed-42 OFF run was affected by contention, so timing comparisons are descriptive rather than controlled cost measurements.

![Resource and pilot compute](reports/experimental_result_figures/06_resource_cost_breakdown.svg)

*Figure 6: Left, mean teacher tokens for ON/OFF. Right, measured cached-pilot wall-hours consumed by Fixed-32 versus the retrospective adaptive simulation.*

Adaptive allocation uses 20 versus 160 pilots for seed 42 and 24 versus 192 for seed 43, saving 7.40 and 11.19 measured GPU wall-hours relative to Fixed-32. Every adaptive outcome is `uncertified_continue`; these are retrospective compute savings, not evidence of safe stopping.

### F.1 Legacy, Range-Aware, and KL-Hoeffding

Legacy symmetric Hoeffding has radius 0.593443; Range-Aware Hoeffding reduces it to 0.296721. Across 11 complete 32-pilot decisions, KL-Hoeffding ranges from 0.138361 to 0.250092, reducing radius by `{radius_summary['minimum']:.2f}%–{radius_summary['maximum']:.2f}%` (mean `{radius_summary['mean']:.2f}%`).

{md_table(['Seed','Round','Mean gain','Range radius','KL radius','Reduction','KL Net-UCB*','Numerical STOP*','Certified STOP'], kl_table)}

`*` uses the frozen **projected marginal decision cost** `0.8983888889` and `lambda=0.2`. It is an optimistic hypothetical diagnostic, not a proved Cost-LCB. It yields `{len(numerical_rows)}` numerical candidates: {', '.join(f"seed {r['seed']} R{r['round']}" for r in numerical_rows)}. The first is R{first_numerical[42]} for seed 42 and R{first_numerical[43]} for seed 43, corresponding counterfactually to 64 and 128 already-trained examples before the next main update. These are not realized budget/cost savings.

![KL confidence and numerical decisions](reports/experimental_result_figures/07_kl_hoeffding_stopping.svg)

*Figure 7: Same-round conditional bounds. Red bars are numerical candidates only; no bar is a statistically certified pre-round STOP.*

Using the protocol-based conditional executable-round Cost-LCB `$0.2264` instead gives zero KL numerical STOPs. More fundamentally, current-round pilots are observed only after Teacher/generation/selection, conditional independence of deterministic pilot seeds is unproved, and no cross-round policy/drift theorem transfers them to a valid pre-round decision. Thus **certified STOP count is zero** and both extended runs end at budget 256.

## G. Key findings and limitations

| Claim | Classification | Evidence | Scope |
|---|---|---|---|
| Uncertainty MNLI is the strongest seed-42 budget-128 row (66.9933%). | Direct completed result | raw Stage-2 result | cross-source, n=1 |
| Verification ON is 0.4422 pp below OFF under three-way, but 0.6622 pp above under Direct Binary. | Controlled ablation | six ON/OFF runs + diagnostics | 3 seeds; rule-dependent |
| V7 extended reaches 75.91% ± 1.17% three-way at budget 256. | Direct completed result | two extended results | 2 seeds, unequal budget |
| V7 learning is non-monotonic across rounds. | Direct completed result | stopping-eval curves | validation subset, not sealed test |
| Pairwise does not improve V2 over CE in seed 42. | Controlled ablation | identical selected IDs | n=1 |
| Adaptive pilots greatly reduce retrospective pilot compute. | Retrospective diagnostic | cached pilot timing simulation | no certified decision |
| KL is tighter than Range-Aware Hoeffding. | Assumption-conditional result | 11 complete pilot sets | fixed shard; independence unproved |
| Cost-Aware stopping safely saves training budget. | Unsupported claim | no certified STOP | must not be claimed |

The principal limitations are unequal data access (human MNLI versus synthetic), unequal budgets (128 versus 256), only one seed for most historical methods, two seeds for extended runs, prediction-rule sensitivity, protocol changes across V2–V7, incomplete pilot evidence in three rounds, shared fixed evaluation shards, and a projected cost that is not a proven lower bound. Seed 42 was repeatedly used during development; it should not be described as untouched confirmatory evidence.

## Recommended main-paper tables and figures

1. Main budget-128 table, separated into human-MNLI and synthetic acquisition blocks.
2. Three-seed Verification ON/OFF table with both HANS rules.
3. Figure 2 (HANS non-entailment versus MNLI retention).
4. Figure 5 (validation learning curves), explicitly labeled `stopping_eval`.
5. A small separate table for the two budget-256 extended runs.

## Recommended appendix tables and figures

1. Full V2–V7 historical ledger and CE/pairwise controlled ablation.
2. HANS heuristic/state tables and full 30-subcase CSV artifacts.
3. Resource accounting and adaptive-pilot diagnostics.
4. KL/Range/legacy confidence-bound table and assumption audit.
5. HANS-30k baseline benchmark, clearly separated from sealed HANS-15k.

## Claim–evidence matrix

| Claim family | Primary artifact | Evaluation role | Admissible wording |
|---|---|---|---|
| Seed-42 baseline ranking | `outputs/stage2_runs*/.../result.json` | sealed HANS 15k / MNLI | completed single-seed comparison |
| ON/OFF effect | `outputs/stage2_runs_v7_training_verification_ablation/` | sealed HANS 15k | controlled three-seed ablation |
| Prediction-rule sensitivity | `outputs/stage2_diagnostics/v7_hans_prediction_rules/` | post-hoc frozen inference | diagnostic; do not select rule on final test |
| Extended performance | `outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2/` | final HANS/MNLI | completed two-seed budget-256 result |
| Cost-Aware decisions | `outputs/stage2_diagnostics/v7_*` | offline replay | retrospective/assumption-conditional only |
| HANS-30k baseline | `reports/hans_baselines_seed42/data/` | old full benchmark | separate protocol only |

## Missing or unverifiable results

- No distinct completed V4 result exists.
- V6 checkpoint continuity failed; its zero-shot-equivalent output is a preserved diagnostic, not a rankable run.
- Historical V2/V3/V5/V7 comparison lacks seeds 43/44.
- Extended seed 44 is absent.
- No statistically valid pre-round gain transfer or certified Cost-Aware STOP exists.
- Shared initial-pool and round-1 costs are incomplete in incremental cost accounting; total pipeline cost is partial.
- Fixed-shard pilot independence/unbiasedness is not established.
- Teacher generation and reviewer calls use the same teacher checkpoint in separate roles, not independent models.

## Draft: Experiments and Results

We evaluate a Qwen3-1.7B NLI student adapted with LoRA under a verified-failure-driven active-learning protocol. A Qwen3-14B teacher proposes failure hypotheses, generates target probes, and produces QC-filtered synthetic intervention pairs. The standard acquisition budget is 128 examples in four batches of 32; an extended protocol permits 256 examples. We partition HANS into disjoint discovery, verification, intervention, gain-evaluation, and sealed-test roles. Epoch-level early stopping observes only a fixed 900-example subset, while the 15,000-example sealed HANS test and MNLI matched/mismatched sets remain reporting-only. We report HANS overall and non-entailment accuracy, together with MNLI retention. Published active-learning methods are reproduced within our implementation and are not quoted results from their original papers.

At budget 128 and seed 42, uncertainty sampling from the human-labelled MNLI pool obtains the strongest sealed-HANS result (66.99%), followed by Margin-MNLI (66.13%) and BADGE-MNLI (65.70%). Historical V7 continuation reaches 65.00%, while Core-set and CAL on a static verified synthetic pool reach 65.05% and 65.03%, respectively. These small differences are not statistically meaningful at one seed, and comparisons with MNLI acquisition are cross-source because the methods have different data access. The main gap is concentrated in non-entailment: uncertainty-MNLI exceeds historical V7 continuation by 5.97 pp.

The controlled three-seed verification ablation is prediction-rule dependent. With the current three-way-to-binary rule, Verification ON averages {100*on_three:.2f}% and OFF {100*off_three:.2f}%; with Direct Binary aggregation, ON averages {100*on_direct:.2f}% and OFF {100*off_direct:.2f}%. Consequently, verification is not uniformly superior under every evaluation mapping. Learning curves are also non-monotonic: for example, seed 42 ON reaches 71.0% on the stopping subset after 96 examples and declines after the fourth batch. This motivates reporting full trajectories and predeclared checkpoint rules rather than retrospectively choosing the best sealed-test checkpoint.

With a maximum budget of 256, the two completed extended V7 runs obtain {pm(extended_three)} three-way HANS and {pm(r['direct'] for r in extended)} Direct Binary. Both terminate by budget exhaustion, not by a statistically certified gain–cost decision. Offline KL-Hoeffding replay tightens the Range-Aware radius by {radius_summary['mean']:.2f}% on average across 11 complete pilot sets. A projected-cost diagnostic produces numerical stopping candidates, but the projected marginal cost is not a proved lower confidence bound and same-round pilots are unavailable before Teacher acquisition. We therefore report zero certified STOPs and treat all apparent savings as counterfactual diagnostics.

Overall, the experiments demonstrate substantial HANS gains from the strengthened V7 training protocol and show that verified synthetic acquisition can approach reproduced synthetic active-learning baselines. They do not establish state of the art: most historical comparisons use one development seed, human-MNLI methods have stronger data access, and Cost-Aware stopping lacks the assumptions required for a valid pre-round certificate.

## Reproducibility and provenance

Generated by `scripts/build_experimental_result.py`. Supporting audit tables are stored in `reports/experimental_result_figures/`; raw outputs remain read-only. Figure SVGs are paper-editable and PNGs are rendered at 300 DPI.
"""

    audit_by_path: dict[str, dict[str, Any]] = {}
    def add_audit(category: str, method: str, seed: int, budget: int, row: dict[str, Any]) -> None:
        audit_by_path[row["path"]] = {"category":category, "method":method, "seed":seed, "budget":budget, "hans":row["hans"], "nonent":row["nonent"], "mnli_m":row["mnli_m"], "source_path":row["path"]}
    for row in main:
        add_audit("budget128_main", row["name"], 42, row["budget"], row)
    for row in onoff:
        audit_by_path[row["path"]] = {"category":"verification_ablation", "method":f"verification_{row['mode'].lower()}", "seed":row["seed"], "budget":128, "hans":row["three"], "nonent":row["three_nonent"], "mnli_m":row["mnli_m"], "source_path":row["path"]}
    for row in extended:
        audit_by_path[row["path"]] = {"category":"extended256", "method":"v7_extended", "seed":row["seed"], "budget":256, "hans":row["three"], "nonent":row["three_nonent"], "mnli_m":row["mnli_m"], "source_path":row["path"]}
    for seed, row in zip((42,43,44), target_rows):
        add_audit("target_only", "verified_target_only", seed, 128, row)
    for row in evolution:
        add_audit("historical_evolution", row["name"], 42, 128, row)
    audit_rows = sorted(audit_by_path.values(), key=lambda row: (row["category"], int(row["seed"]), row["method"]))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with (FIG_DIR / "audit_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(audit_rows)
    summary = {
        "completed_result_rows": len(audit_rows), "method_protocol_labels": 23, "main_methods": len(main),
        "tables": 12, "figures": 7, "numerical_kl_stops": len(numerical_rows),
        "certified_stops": 0, "first_numerical_stop": first_numerical,
    }
    (FIG_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    return report, summary


def main() -> None:
    report, summary = build()
    if report.count("## ") < 12:
        raise RuntimeError("Report section validation failed")
    expected = [REPORT, FIG_DIR / "audit_ledger.csv", FIG_DIR / "audit_summary.json"]
    expected.extend(FIG_DIR / f"{index:02d}_{name}.{suffix}" for index, name in (
        (1,"main_hans_comparison"),(2,"nonent_mnli_tradeoff"),(3,"learning_curves"),
        (4,"verification_ablation"),(5,"v2_v7_evolution"),(6,"resource_cost_breakdown"),
        (7,"kl_hoeffding_stopping"),
    ) for suffix in ("svg","png"))
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing deliverables: {missing}")
    print(json.dumps(summary, indent=2))
    print(f"[DONE] {REPORT}")
    print(f"[DONE] {FIG_DIR}")


if __name__ == "__main__":
    main()
