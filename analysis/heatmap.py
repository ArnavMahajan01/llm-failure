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

BENCHMARK_LABELS = {
    "gsm_symbolic": "GSM-Symbolic",
    "gsm_plus": "GSM-Plus",
    "gsm_ic": "GSM-IC",
    "bigbench_hard": "BIG-Bench Hard",
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


# Fig 1: Baseline vs ICL Accuracy
def fig1_accuracy(df_summary):
    df = df_summary.copy()
    df["model_label"] = df["model"].apply(friendly_model)
    models = df["model_label"].tolist()

    columns = ["zero_shot_acc", "random_few_shot_acc", "targeted_few_shot_acc"]
    labels = ["Zero-Shot", "Random ICL", "Targeted ICL"]
    colors = [CONDITION_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.25

    for i, (col, label, color) in enumerate(zip(columns, labels, colors)):
        vals = df[col].values
        bars = ax.bar(
            x + i * width, vals, width,
            label=label, color=color,
            edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline vs In-Context Learning Accuracy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "fig1_accuracy.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# Fig 2: Error Taxonomy (zero-shot failures)
def fig2_error_taxonomy(df_errors):
    df = df_errors.copy()
    df["error_label"] = df["error_type"].map(ERROR_LABELS)
    df["model_label"] = df["model"].apply(friendly_model)

    # Keep only error types that actually appear
    active = df.groupby("error_label")["count"].sum()
    active = active[active > 0].index.tolist()
    df_plot = df[df["error_label"].isin(active)]

    pivot = df_plot.pivot(
        index="model_label", columns="error_label", values="percentage"
    ).fillna(0)

    # Annotate with count + percentage
    annot_rows = []
    for _, row in df_plot.iterrows():
        annot_rows.append({
            "model_label": row["model_label"],
            "error_label": row["error_label"],
            "label": f"{row['percentage']:.1f}%\n(n={int(row['count'])})" if row["count"] > 0 else "—",
        })
    annot_df = pd.DataFrame(annot_rows)
    annot_pivot = annot_df.pivot(
        index="model_label", columns="error_label", values="label"
    ).fillna("—")
    annot_pivot = annot_pivot[pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot, annot=annot_pivot.values, fmt="",
        cmap="YlOrRd", vmin=0, vmax=100,
        linewidths=0.8, linecolor="white", ax=ax,
        cbar_kws={"label": "% of Zero-Shot Errors", "shrink": 0.8},
    )
    ax.set_title("Error Taxonomy: Zero-Shot Failure Classification")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "fig2_error_taxonomy.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# Fig 3: ICL Recovery Heatmap
def fig3_recovery_heatmap(df_summary):
    """
    Shows what prompting can and cannot fix.
    """
    df = df_summary.copy()
    df["model_label"] = df["model"].apply(friendly_model)
    df["benchmark_label"] = df["benchmark"].apply(friendly_benchmark)

    benchmarks = df["benchmark_label"].unique()

    if len(benchmarks) == 1:
        # Single benchmark: show model × strategy
        _recovery_heatmap_single_benchmark(df)
    else:
        # Multiple benchmarks: show model × benchmark (targeted recovery)
        _recovery_heatmap_multi_benchmark(df)


def _recovery_heatmap_single_benchmark(df):
    benchmark_name = df["benchmark_label"].iloc[0]

    models = df["model_label"].tolist()
    data = pd.DataFrame({
        "Random ICL": df["recovery_rate_random"].values * 100,
        "Targeted ICL": df["recovery_rate_targeted"].values * 100,
    }, index=models)

    # Compute zero-shot failure counts for annotation
    n_failures = (df["n_samples"] * (1 - df["zero_shot_acc"])).astype(int).values

    # Annotation: recovery rate + failure count context
    annot = np.empty(data.shape, dtype=object)
    for i in range(len(models)):
        for j, col in enumerate(data.columns):
            rate = data.iloc[i, j]
            n = n_failures[i]
            recovered = int(round(rate * n / 100))
            annot[i, j] = f"{rate:.1f}%\n({recovered}/{n})"

    # Custom diverging colormap: red -> yellow -> green
    cmap = LinearSegmentedColormap.from_list(
        "recovery", ["#d32f2f", "#ff9800", "#fdd835", "#8bc34a", "#2e7d32"]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        data, annot=annot, fmt="",
        cmap=cmap, vmin=0, vmax=70,
        linewidths=1.0, linecolor="white", ax=ax,
        cbar_kws={"label": "Recovery Rate %", "shrink": 0.8},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title(
        f"ICL Recovery Heatmap: {benchmark_name}\n"
        f"What fraction of zero-shot failures does prompting fix?",
        fontsize=13,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=11)
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "fig3_recovery_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def _recovery_heatmap_multi_benchmark(df):
    """Model x Benchmark recovery heatmap (targeted ICL recovery)."""
    pivot = df.pivot(
        index="model_label", columns="benchmark_label",
        values="recovery_rate_targeted",
    ).fillna(0) * 100

    # Annotation with rates
    annot = pivot.map(lambda v: f"{v:.1f}%")

    cmap = LinearSegmentedColormap.from_list(
        "recovery", ["#d32f2f", "#ff9800", "#fdd835", "#8bc34a", "#2e7d32"]
    )

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2.5), max(4, len(pivot) * 1.2)))
    sns.heatmap(
        pivot, annot=annot, fmt="",
        cmap=cmap, vmin=0, vmax=70,
        linewidths=1.0, linecolor="white", ax=ax,
        cbar_kws={"label": "Recovery Rate %", "shrink": 0.8},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title(
        "ICL Recovery Heatmap — Targeted Few-Shot\n"
        "What fraction of zero-shot failures does error-matched prompting fix?",
        fontsize=13,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "fig3_recovery_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

# Fig 4: Failure Mode Profile, same data as fig 2 but displayed as horizontal stacked bars
def fig4_failure_profile(df_errors):
    df = df_errors.copy()
    df["error_label"] = df["error_type"].map(ERROR_LABELS)
    df["model_label"] = df["model"].apply(friendly_model)

    active = df.groupby("error_label")["count"].sum()
    active = active[active > 0].index.tolist()
    df_plot = df[df["error_label"].isin(active)]

    pivot = df_plot.pivot(
        index="model_label", columns="error_label", values="percentage"
    ).fillna(0)

    palette = sns.color_palette("Set2", n_colors=len(active))
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(kind="barh", stacked=True, ax=ax, color=palette, edgecolor="white", linewidth=0.5)

    for i, model in enumerate(pivot.index):
        cumulative = 0
        for j, col in enumerate(pivot.columns):
            val = pivot.loc[model, col]
            if val > 5:
                ax.text(
                    cumulative + val / 2, i,
                    f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                )
            cumulative += val

    ax.set_xlabel("% of Zero-Shot Errors")
    ax.set_title("Failure Mode Profile by Model")
    ax.legend(title="Error Taxonomy", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "fig4_failure_profile.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# Main
def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("Loading processed CSVs from processResults.py...")
    df_summary = load_csv("summary.csv")
    df_errors = load_csv("error_distribution.csv")

    if df_summary is None or df_errors is None:
        print("\nMissing CSVs. Run processResults.py first:")
        print("  python3 -m results.processResults <files...>")
        return

    models = df_summary["model"].apply(friendly_model).tolist()
    benchmarks = df_summary["benchmark"].apply(friendly_benchmark).unique().tolist()
    print(f"\nModels: {models}")
    print(f"Benchmarks: {benchmarks}")

    print("\nGenerating figures...")
    fig1_accuracy(df_summary)
    fig2_error_taxonomy(df_errors)
    fig3_recovery_heatmap(df_summary)
    fig4_failure_profile(df_errors)

    print(f"\nDone. All figures saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
