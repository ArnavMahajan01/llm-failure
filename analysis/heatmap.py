import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from config import PROCESSED_DIR, RESULTS_DIR

# config

CHARTS_DIR = os.path.join(PROCESSED_DIR, "charts")

# labels aligned with report
CONDITION_LABELS = {
    "zero_shot_acc": "Zero-Shot",
    "random_few_shot_acc": "Random ICL",
    "targeted_few_shot_acc": "Targeted ICL",
}

CONDITION_COLORS = {
    "Zero-Shot": "#B8C480",
    "Random ICL": "#922D50",
    "Targeted ICL": "#3C1B43",
}

ERROR_LABELS = {
    "confidently_wrong": "Confidently Wrong",
    "drifting": "Drifting",
    "distractor": "Distractor",
    "negation": "Negation",
    "multi_hop": "Multi-Hop",
    "hallucination": "Hallucination",
    "format_correct_wrong_answer": "Format / Extraction",
    "off_topic": "Off-Topic",
    "unknown": "Unknown",
}

STRATEGY_LABELS = {
    "random_few_shot": "Random ICL",
    "targeted_few_shot": "Targeted ICL",
    "targeted_few_shot_k5": "Targeted ICL (k=5)",
    "error_targeted_icl": "Error-Targeted",
    "error_targeted_icl_random": "Error-Targeted\n(Random)",
    "error_targeted_icl_correct_only": "Error-Targeted\n(Correct)",
    "self_consistency": "Self-Consistency",
}

BENCHMARK_LABELS = {
    "gsm_symbolic": "GSM-Symbolic",
    "gsm_plus": "GSM-Plus",
    "gsm_ic": "GSM-IC",
    "bigbench_hard": "BIG-Bench Hard",
    "bigbench_hard_tracking": "BIG-Bench Hard Tracking",
    "gsm8k": "GSM-8k",
    "folio": "FOLIO",
}

# style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "font.family": "sans-serif",
})

def load_csv(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {filename}: {len(df)} rows")
    return df


# display name formatter 
def friendly_model(name):
    parts = name.split("/")
    if len(parts) == 2:
        raw = parts[1]
        # Extract param size from end of string
        for size in ["1B", "1-5B", "3B", "4B", "8B", "9B"]:
            if size in raw:
                family = parts[0]
                display_size = size.replace("-", ".")
                return f"{family} {display_size}"
    return name


def friendly_benchmark(name):
    return BENCHMARK_LABELS.get(name, name)


# Fig 1: Error × Strategy Recovery Heatmap
def fig1_error_strategy_heatmap(df_recovery):
    """
    Heatmap: rows = error type, columns = ICL strategy, values = recovery rate.
    Shows which prompting strategies fix which failure modes.
    """
    df = df_recovery.copy()

    # Drop low-sample error types (n < 10 failures)
    max_failures = df.groupby("error_type")["failures"].max()
    df = df[df["error_type"].isin(max_failures[max_failures >= 10].index)]

    df["error_label"] = df["error_type"].map(ERROR_LABELS).fillna(df["error_type"])
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])

    pivot = df.pivot(index="error_label", columns="strategy_label", values="recovery_rate").fillna(np.nan) * 100
    failures_pivot = df.pivot(index="error_label", columns="strategy_label", values="failures")
    recovered_pivot = df.pivot(index="error_label", columns="strategy_label", values="recovered")

    # Sort rows: most frequent error type first
    row_order = (
        df.groupby("error_label")["failures"].max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    col_order = [
        STRATEGY_LABELS[s] for s in [
            "random_few_shot", "targeted_few_shot", "targeted_few_shot_k5",
            "error_targeted_icl", "error_targeted_icl_random",
            "error_targeted_icl_correct_only", "self_consistency",
        ]
        if STRATEGY_LABELS[s] in pivot.columns
    ]
    pivot = pivot.reindex(row_order)[col_order]
    failures_pivot = failures_pivot.reindex(row_order)[col_order]
    recovered_pivot = recovered_pivot.reindex(row_order)[col_order]

    # Add E/S numbering to axis labels
    pivot.index = [f"E{i+1}  {l}" for i, l in enumerate(pivot.index)]
    pivot.columns = [f"S{i+1}\n{l}" for i, l in enumerate(pivot.columns)]
    failures_pivot.index = pivot.index
    failures_pivot.columns = pivot.columns
    recovered_pivot.index = pivot.index
    recovered_pivot.columns = pivot.columns

    annot = np.empty(pivot.shape, dtype=object)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            rate = pivot.iloc[i, j]
            n = failures_pivot.iloc[i, j]
            if np.isnan(rate):
                annot[i, j] = "—"
            else:
                annot[i, j] = f"{rate:.1f}%\n({int(recovered_pivot.iloc[i, j])}/{int(n)})"

    cmap = LinearSegmentedColormap.from_list(
        "recovery", ["#d32f2f", "#ff9800", "#fdd835", "#8bc34a", "#2e7d32"]
    )

    fig, ax = plt.subplots(figsize=(14, max(5, len(pivot) * 0.85)))
    sns.heatmap(
        pivot, annot=annot, fmt="",
        cmap=cmap, vmin=0, vmax=55,
        linewidths=0.8, linecolor="white", ax=ax,
        cbar_kws={"label": "Recovery Rate (%)", "shrink": 0.75},
        annot_kws={"fontsize": 9, "fontweight": "bold"},
    )
    ax.set_title(
        "Fig 1: Which ICL Strategies Fix Which Errors?\n"
        "Recovery rate: % of zero-shot failures recovered by each prompting strategy",
        fontsize=13, pad=12,
    )
    ax.set_ylabel("Error Type", fontsize=11)
    ax.set_xlabel("ICL Strategy", fontsize=11)
    ax.tick_params(axis="y", labelsize=10, rotation=0)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "fig1_error_strategy_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# Fig 2: Model Family Comparison (grouped bar, one chart per parameter-size tier)
# DONE BY CLAUDE
def _parse_family_tier(model_name):
    """Return (family, tier_label) from raw model name."""
    parts = model_name.split("/")
    if len(parts) != 2:
        return model_name, "other"
    family, raw = parts[0], parts[1]
    for token, tier in [("1-5B", "1B–1.5B"), ("1B", "1B–1.5B"),
                        ("3B", "3B–4B"),   ("4B", "3B–4B")]:
        if token in raw:
            return family, tier
    return family, "other"


def fig2_family_comparison(df_summary):
    """
    Grouped bar chart per parameter-size tier.
    X = model family, bars = ICL conditions, Y = mean accuracy across benchmarks.
    Saves fig2.1 (1B-1.5B) and fig2.2 (3B-4B).
    """
    df = df_summary.copy()
    df["family"], df["tier"] = zip(*df["model"].apply(_parse_family_tier))

    conditions = [
        ("zero_shot_acc",          "Zero-Shot",        "#B8C480"),
        ("random_few_shot_acc",    "Random ICL",       "#922D50"),
        ("targeted_few_shot_acc",  "Targeted ICL",     "#3C1B43"),
        ("error_targeted_icl_acc", "Error-Targeted",   "#2E6AA6"),
    ]

    tiers = [("1B–1.5B", "fig2.1"), ("3B–4B", "fig2.2")]

    for tier_name, fig_id in tiers:
        tier_df = df[df["tier"] == tier_name]
        if tier_df.empty:
            continue

        families = sorted(tier_df["family"].unique())
        agg = tier_df.groupby("family")[[c for c, _, _ in conditions]].mean()

        x = np.arange(len(families))
        n_conds = len(conditions)
        width = 0.18
        offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * width

        fig, ax = plt.subplots(figsize=(8, 5))

        for offset, (col, label, color) in zip(offsets, conditions):
            vals = [agg.loc[f, col] if (f in agg.index and col in agg.columns and not pd.isna(agg.loc[f, col])) else 0
                    for f in families]
            bars = ax.bar(x + offset, vals, width, label=label,
                          color=color, edgecolor="white", linewidth=0.6)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.012,
                            f"{v:.1%}", ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(families, fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (mean across benchmarks)", fontsize=10)
        ax.set_title(
            f"Fig 2: Model Family Comparison at {tier_name}\n"
            "Which family benefits most from ICL?",
            fontsize=12,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, min(1.0, agg[[c for c, _, _ in conditions]].max().max() + 0.18))
        ax.legend(loc="upper right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        path = os.path.join(CHARTS_DIR, f"{fig_id}_family_comparison.png")
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


# Fig 3: Strategy Progression Line Chart
# Does each model get better after each strategy step?
# One colour per family, two line styles for the two sizes within each family
FAMILY_STYLES = {
    "Gemma": {"color": "#4CAF50", "sizes": {"1B": "-",   "4B": "--"}},
    "Llama": {"color": "#E53935", "sizes": {"1.5B": "-", "3B": "--"}},
    "Qwen":  {"color": "#7B1FA2", "sizes": {"1.5B": "-", "3B": "--"}},
}

def fig3_strategy_progression(df_summary):
    """
    Line chart: X = strategy steps (ordered baseline -> sophisticated),
    Y = mean accuracy across benchmarks, one line per model.
    Shows whether each model improves after each prompting step.
    """
    df = df_summary.copy()
    df["model_label"] = df["model"].apply(friendly_model)

    # Build ordered steps: zero-shot first, then each strategy from STRATEGY_LABELS
    all_steps = [("zero_shot_acc", "Zero-Shot")] + [
        (f"{k}_acc", label) for k, label in STRATEGY_LABELS.items()
    ]
    cols     = [c     for c, _     in all_steps if c in df.columns]
    x_labels = [label for c, label in all_steps if c in df.columns]

    # Micro-average: (sum of correct items) / (sum of total items) across benchmarks
    # correct_items = accuracy * n_samples  -> sum / sum(n_samples)
    grouped = df.groupby("model_label")
    agg = grouped.apply(
        lambda g: pd.Series({
            c: (g[c] * g["n_samples"]).sum() / g["n_samples"].sum()
            for c in cols
        })
    )
    x = np.arange(len(cols))

    fig, ax = plt.subplots(figsize=(13, 6))

    for model_label, row in agg.iterrows():
        # Derive family and size from label e.g. "Gemma 1B", "Llama 1.5B"
        parts = model_label.split()
        family = parts[0] if parts else model_label
        size   = parts[1] if len(parts) > 1 else ""

        style = FAMILY_STYLES.get(family, {"color": "#888", "sizes": {}})
        color = style["color"]
        ls    = style["sizes"].get(size, "-")
        marker = "o" if ls == "-" else "s"

        vals = [row[c] for c in cols]
        ax.plot(x, vals, linestyle=ls, marker=marker, color=color,
                linewidth=2, markersize=6, label=model_label)

        # Annotate final point
        ax.annotate(f"{vals[-1]:.0%}", xy=(x[-1], vals[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Accuracy (mean across benchmarks)", fontsize=11)
    ax.set_xlabel("Prompting Strategy  ->  increasing sophistication", fontsize=10)
    ax.set_title(
        "Fig 3: Does Each Model Improve Across Strategy Steps?\n"
        "Solid line = smaller model (1B / 1.5B)   Dashed = larger (3B / 4B)",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.3, len(cols) - 0.3)
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "fig3_strategy_progression.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# Fig 4: Radar Chart, per-model, accuracy across datasets by strategy
RADAR_STRATEGIES = [
    ("zero_shot_baseline_acc",                "Zero-Shot\nBaseline"),  
    ("zero_shot_acc",                         "Zero-Shot"),
    ("targeted_few_shot_answer_only_acc",     "Targeted ICL\n(Ans Only)"),
    ("random_few_shot_acc",                   "Random ICL"),
    ("targeted_few_shot_acc",                 "Targeted ICL"),
    ("targeted_few_shot_k5_acc",              "Targeted ICL (k=5)"),
    ("error_targeted_icl_acc",                "Error-Targeted"),
    ("error_targeted_icl_random_acc",         "ET (Random)"),
    ("error_targeted_icl_correct_only_acc",   "ET (Correct)"),
    ("self_consistency_acc",                  "Self-Consistency"),
]
BOLD_SPOKE_COL = "zero_shot_baseline_acc"


def fig4_radar_model(df_summary, model_name):
    """
    One radar chart per dataset for the given model.
    Spokes   = ICL strategies (8 spokes, always consistent).
    Polygon  = model accuracy on each strategy for that dataset.
    Duplicate runs for the same benchmark are micro-averaged first.
    Usage: python3 -m analysis.heatmap -Llama/Llama3-2-3B
    """
    available = df_summary["model"].unique()
    if model_name not in available:
        print(f"  ERROR: '{model_name}' not found.")
        print(f"  Available models: {sorted(available)}")
        return

    df = df_summary[df_summary["model"] == model_name].copy()
    acc_cols = [col for col, _ in RADAR_STRATEGIES if col in df.columns]

    # Micro-average duplicate runs per benchmark
    df = (
        df.groupby("benchmark")
        .apply(lambda g: pd.Series({
            c: (g[c] * g["n_samples"]).sum() / g["n_samples"].sum()
            for c in acc_cols
        }))
        .reset_index()
    )

    model_label = friendly_model(model_name)
    safe_model  = model_label.replace(" ", "_").replace("/", "-")
    family      = model_label.split()[0]
    color       = {"Gemma": "#4CAF50", "Llama": "#E53935", "Qwen": "#7B1FA2"}.get(family, "#2E6AA6")

    # Spokes = strategies present in data
    strategies   = [(col, label) for col, label in RADAR_STRATEGIES if col in acc_cols]
    spoke_labels = [label for _, label in strategies]
    N      = len(strategies)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # One file per benchmark
    for _, row in df.iterrows():
        benchmark_key   = row["benchmark"]
        benchmark_label = friendly_benchmark(benchmark_key)
        safe_bm         = benchmark_key.replace("/", "-")

        vals = [row[col] for col, _ in strategies] + [row[strategies[0][0]]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spoke_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7, color="grey")
        ax.tick_params(pad=10)

        bold_idx = next(
            (i for i, (col, _) in enumerate(strategies) if col == BOLD_SPOKE_COL),
            None,
        )
        if bold_idx is not None:
            for i, lbl in enumerate(ax.get_xticklabels()):
                if i == bold_idx:
                    lbl.set_fontweight("bold")
            ax.scatter(
                angles[bold_idx],
                vals[bold_idx],
                s=90,
                color="#212121",
                zorder=5,
                edgecolors="white",
                linewidth=1.2,
            )

        ax.plot(angles, vals, "o-", linewidth=2.5, color=color, markersize=6)
        ax.fill(angles, vals, alpha=0.2, color=color)

        # Accuracy label at each spoke
        for angle, val in zip(angles[:-1], vals[:-1]):
            ax.annotate(
                f"{val:.0%}",
                xy=(angle, val),
                xytext=(angle, min(val + 0.09, 0.96)),
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=color,
            )

        ax.set_title(
            f"Fig 4: {model_label}  ·  {benchmark_label}\nAccuracy by ICL Strategy",
            fontsize=11, pad=22,
        )
        plt.tight_layout()

        path = os.path.join(CHARTS_DIR, f"fig4_{safe_model}_{safe_bm}_radar.png")
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


# Fig 5: Cumulative accuracy per question, one line per dataset
# BENCHMARK_COLORS keeps lines visually distinct regardless of order
BENCHMARK_COLORS = {
    "gsm8k":                "#E53935",
    "gsm_symbolic":         "#F9A825",
    "gsm_plus":             "#FB8C00",
    "gsm_ic":               "#43A047",
    "bigbench_hard":        "#1E88E5",
    "bigbench_hard_tracking":"#8E24AA",
    "folio":                "#00ACC1",
}

def moving_average(values, window=5):
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values  # not enough points to smooth
    return np.convolve(values, np.ones(window) / window, mode="same")


def fig5_cumulative_accuracy(model_name, condition="zero_shot"):
    """
    Line chart: X = question index, Y = running accuracy up to that question.
    One line per dataset (benchmark) for the given model.
    Reads raw JSON result files directly from RESULTS_DIR.
    Usage: python3 -m analysis.heatmap -Llama/Llama3-2-3B
    """
    # Filename format: {Family}_{ModelName}__{benchmark}__{timestamp}.json
    # "/" → "_" and "." → "-" so e.g. Llama/Llama3.2-1.5B → Llama_Llama3-2-1-5B
    file_prefix = model_name.replace("/", "_", 1).replace(".", "-")
    pattern = os.path.join(RESULTS_DIR, "**", f"{file_prefix}__*.json")
    # Sort descending so the latest timestamp file comes first
    files = sorted(glob.glob(pattern, recursive=True), reverse=True)

    if not files:
        print(f"  ERROR: No result files found for model '{model_name}'")
        print(f"  Searched: {pattern}")
        return

    model_label = friendly_model(model_name)
    fallback_colors = list(plt.cm.tab10.colors)

    fig, ax = plt.subplots(figsize=(13, 6))
    plotted = 0

    # Track seen benchmarks — latest-timestamp file wins (files sorted descending)
    seen_benchmarks = set()

    for filepath in files:
        filename  = os.path.basename(filepath)
        parts     = filename.replace(".json", "").split("__")
        # parts: [model_prefix, benchmark, timestamp]
        benchmark = parts[1] if len(parts) >= 2 else filename

        if benchmark in seen_benchmarks:
            continue
        seen_benchmarks.add(benchmark)

        with open(filepath, encoding="utf-8") as f:
            results = json.load(f)

        # Extract majority_correct for the requested condition, in question order
        correct_flags = [
            item.get("conditions", {}).get(condition, {}).get("majority_correct", False)
            for item in results
        ]

        if not correct_flags:
            print(f"  WARNING: no '{condition}' data in {filename}, skipping")
            continue

        running_acc = np.cumsum(correct_flags) / np.arange(1, len(correct_flags) + 1)
        smoothed_acc = moving_average(running_acc, window=5)
        x = np.arange(1, len(running_acc) + 1)

        color = BENCHMARK_COLORS.get(benchmark, fallback_colors[plotted % len(fallback_colors)])
        bm_label = friendly_benchmark(benchmark)
        final_acc = running_acc[-1]
        ax.plot(x, running_acc, linewidth=1.8, color=color, label=f"{bm_label} (final {final_acc:.1%})" )
        #ax.plot(x, smoothed_acc, linewidth=2.2, color=color, label=f"{bm_label} (final {final_acc:.1%})")

        # Mark the final accuracy with a dot
        ax.scatter([x[-1]], [running_acc[-1]], s=40, color=color, zorder=5)
        plotted += 1

    if plotted == 0:
        print(f"  No data plotted for {model_name}.")
        plt.close()
        return

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Question Index", fontsize=11)
    ax.set_ylabel("Cumulative Accuracy", fontsize=11)
    ax.set_title(
        f"Fig 5: Running Accuracy: {model_label}\n"
        f"Cumulative accuracy after each question  "
        f"[condition: {condition.replace('_', ' ')}]",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(left=1)
    plt.tight_layout()

    safe_model = model_label.replace(" ", "_").replace("/", "-")
    path = os.path.join(CHARTS_DIR, f"fig5_{safe_model}_cumulative_accuracy.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# Fig 6: Cumulative accuracy per question, one line per model, fixed benchmark
def fig6_cumulative_by_benchmark(benchmark, condition="zero_shot"):
    """
    Line chart: X = question index, Y = running accuracy up to that question.
    One line per model for the given benchmark/dataset.
    Reads raw JSON result files directly from RESULTS_DIR.
    Usage: python3 -m analysis.heatmap --gsm8k
    """
    # Files live in RESULTS_DIR/{benchmark}/ with names {model}__{benchmark}__{timestamp}.json
    bm_dir  = os.path.join(RESULTS_DIR, benchmark)
    pattern = os.path.join(bm_dir, "*.json")
    # Sort descending so the latest-timestamp file comes first per model
    files = sorted(glob.glob(pattern), reverse=True)

    if not files:
        print(f"  ERROR: No result files found for benchmark '{benchmark}'")
        print(f"  Searched: {pattern}")
        return

    benchmark_label = friendly_benchmark(benchmark)
    fallback_colors = list(plt.cm.tab10.colors)

    fig, ax = plt.subplots(figsize=(13, 6))
    plotted = 0
    seen_models = set()

    for filepath in files:
        filename = os.path.basename(filepath)
        parts    = filename.replace(".json", "").split("__")
        # parts: [model_prefix, benchmark, timestamp]
        if len(parts) < 2:
            continue

        # Reconstruct model name the same way loadResults() does
        modelname = parts[0].replace("_", "/", 1)

        # Skip duplicate files for the same model — latest timestamp wins (sorted desc)
        if modelname in seen_models:
            continue
        seen_models.add(modelname)

        with open(filepath, encoding="utf-8") as f:
            results = json.load(f)

        correct_flags = [
            item.get("conditions", {}).get(condition, {}).get("majority_correct", False)
            for item in results
        ]
        if not correct_flags:
            print(f"  WARNING: no '{condition}' data in {filename}, skipping")
            continue

        running_acc = np.cumsum(correct_flags) / np.arange(1, len(correct_flags) + 1)
        x = np.arange(1, len(running_acc) + 1)

        # Derive colour + line style from family (same scheme as Fig 3)
        model_label  = friendly_model(modelname)
        label_parts  = model_label.split()
        family       = label_parts[0] if label_parts else model_label
        size         = label_parts[1] if len(label_parts) > 1 else ""
        style        = FAMILY_STYLES.get(family, {"color": fallback_colors[plotted % len(fallback_colors)], "sizes": {}})
        color        = style["color"]
        ls           = style["sizes"].get(size, "-")
        marker       = "o" if ls == "-" else "s"

        final_acc = running_acc[-1]
        ax.plot(x, running_acc, linewidth=1.8, color=color, linestyle=ls,
                label=f"{model_label}  ({final_acc:.1%})")
        ax.scatter([x[-1]], [running_acc[-1]], s=45, color=color,
                   marker=marker, zorder=5)
        plotted += 1

    if plotted == 0:
        print(f"  No data plotted for benchmark '{benchmark}'.")
        plt.close()
        return

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Question Index", fontsize=11)
    ax.set_ylabel("Cumulative Accuracy", fontsize=11)
    ax.set_title(
        f"Fig 6: Running Accuracy: {benchmark_label}\n"
        f"Cumulative accuracy after each question  "
        f"[condition: {condition.replace('_', ' ')}]",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(left=1)
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, f"fig6_{benchmark}_cumulative_accuracy.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# Main
def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("Loading processed CSVs from processResults.py...")
    df_summary = load_csv("summary.csv")
    df_errors = load_csv("error_distribution.csv")
    df_recovery = load_csv("error_strategy_recovery.csv")

    if df_summary is None or df_recovery is None:
        print("\nMissing CSVs. Run processResults.py first:")
        print("  python3 -m results.processResults <files...>")
        return

    # CLI args:
    #   -<model>      → Fig 4 (radar) + Fig 5 (cumulative per dataset)
    #   --<benchmark> → Fig 6 (cumulative per model for that dataset)
    model_arg = next(
        (arg.lstrip("-") for arg in sys.argv[1:]
         if arg.startswith("-") and not arg.startswith("--")),
        None,
    )
    benchmark_arg = next(
        (arg.lstrip("-") for arg in sys.argv[1:] if arg.startswith("--")),
        None,
    )

    print("\nGenerating figures...")
    fig1_error_strategy_heatmap(df_recovery)
    fig2_family_comparison(df_summary)
    fig3_strategy_progression(df_summary)

    if model_arg:
        print(f"\nGenerating Fig 4 radar chart for model: {model_arg}")
        fig4_radar_model(df_summary, model_arg)
        print(f"\nGenerating Fig 5 cumulative accuracy for model: {model_arg}")
        fig5_cumulative_accuracy(model_arg)

    if benchmark_arg:
        print(f"\nGenerating Fig 6 cumulative accuracy for benchmark: {benchmark_arg}")
        fig6_cumulative_by_benchmark(benchmark_arg)

    if not model_arg and not benchmark_arg:
        print("\nTip: pass optional args to generate per-model/benchmark figures:")
        print("  -<model>      → Fig 4 + Fig 5  e.g. -Llama/Llama3-2-3B")
        print("  --<benchmark> → Fig 6           e.g. --gsm8k")

    print(f"\nDone. All figures saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
