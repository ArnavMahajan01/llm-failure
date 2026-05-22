import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from config import PROCESSED_DIR

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
        for size in ["270M", "1B", "1-5B", "3B", "8B", "9B"]:
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

    annot = np.empty(pivot.shape, dtype=object)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            rate = pivot.iloc[i, j]
            n = failures_pivot.iloc[i, j]
            if np.isnan(rate):
                annot[i, j] = "—"
            else:
                annot[i, j] = f"{rate:.1f}%\n(n={int(n)})"

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

    print("\nGenerating figures...")
    fig1_error_strategy_heatmap(df_recovery)

    print(f"\nDone. All figures saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
