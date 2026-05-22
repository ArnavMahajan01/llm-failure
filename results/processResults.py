import json
import os
import glob
import sys
import pandas as pd
from config import RESULTS_DIR, PROCESSED_DIR
from evaluation.taxonomy import ERROR_TYPES


def loadResults() -> list:

    path = os.path.join(RESULTS_DIR, "**", "*.json")
    files = glob.glob(path, recursive=True)

    if not files:
        print(f"No files found in {RESULTS_DIR}")
        print(f"Run run_experiment first")
        return []
    
    loadedFiles = []

    for filePath in sorted(files):
        filename = os.path.basename(filePath)
        parts = filename.replace(".json", "").split("__")
        if len(parts) < 2:
            print(f"WARNING: Unexpected file format {filename}. Skipping it")
            continue

        modelname = parts[0].replace("_", "/", 1)
        benchmark = parts[1]

        with open(filePath, encoding="utf-8") as f:
            results = json.load(f)

        loadedFiles.append((modelname, benchmark, results))
        print(f"  Loaded: {filename} ({len(results)} samples)")

    return loadedFiles

def accuracy(result: list, condition: str) -> float:
    if not result:
        return 0.0
    correctVal = sum(1 for item in result if item["conditions"].get(condition, {}).get("majority_correct", False))

    return correctVal / len(result)

def errorDistribution(result: list) -> dict:
    counts = {errorType: 0 for errorType in ERROR_TYPES}
    total = 0

    for item in result:
        zeroShotRun = item["conditions"].get("zero_shot", {}).get("runs", [])
        for run in zeroShotRun:
            if not run.get("correct") and run.get("error_type"):
                errorType = run["error_type"]
                if errorType in counts:
                    counts[errorType] += 1
                    total += 1

    if total == 0:
        return counts

    return {errorType: {"count": c, "pct": round(c / total * 100, 1)} for errorType, c in counts.items()}

def getErrors(results: list) -> pd.DataFrame:
    rows = []

    for modename, benchmark, result in results:
        errors = errorDistribution(result)
        for errorType, stats in errors.items():
            if isinstance(stats, dict):
                rows.append({
                    "model": modename,
                    "benchmark": benchmark,
                    "error_type": errorType,
                    "count": stats["count"],
                    "percentage": stats["pct"]
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()

# DONE BY CLAUDE
def recoveryRate(results: list, from_condition: str = "zero_shot", to_condition: str = "targeted_few_shot") -> float:
    """
    Recovery rate = fraction of from_condition failures that succeed under to_condition.
    This is the key metric for the heatmap.
    """
    failures = [
        item for item in results
        if not item["conditions"].get(from_condition, {}).get("majority_correct", False)
        and to_condition in item["conditions"]
    ]

    if not failures:
        return 0.0

    recovered = sum(
        1 for item in failures
        if item["conditions"][to_condition].get("majority_correct", False)
    )

    return recovered / len(failures)

ALL_CONDITIONS = [
    "zero_shot_baseline",
    "zero_shot",
    "targeted_few_shot_answer_only",
    "random_few_shot",
    "targeted_few_shot",
    "targeted_few_shot_k5",
    "error_targeted_icl",
    "error_targeted_icl_random",
    "error_targeted_icl_correct_only",
    "self_consistency",
]

def getSummary(results: list) -> pd.DataFrame:
    rows = []

    for modelname, benchmark, result in results:
        if not result:
            continue

        # Only include conditions that actually appear in this file
        present = {c for item in result for c in item["conditions"]}

        row = {"model": modelname, "benchmark": benchmark, "n_samples": len(result)}

        for cond in ALL_CONDITIONS:
            if cond in present:
                row[f"{cond}_acc"] = accuracy(result, cond)

        # Recovery rate vs zero_shot for every non-baseline strategy
        recovery_targets = [c for c in ALL_CONDITIONS
                            if c not in ("zero_shot_baseline", "zero_shot") and c in present]
        for cond in recovery_targets:
            row[f"recovery_{cond}"] = recoveryRate(result, "zero_shot", cond)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if df[c].dtype == float]
    df[numeric_cols] = df[numeric_cols].round(3)

    return df


def getCategoryBreakdown(results: list) -> pd.DataFrame:
    from collections import defaultdict

    rows = []

    for modelname, benchmark, result in results:
        if not result:
            continue

        # Group items by category
        by_cat = defaultdict(list)
        for item in result:
            cat = item.get("benchmark_category") or "unknown"
            by_cat[cat].append(item)

        present = {c for item in result for c in item["conditions"]}

        for cat, items in by_cat.items():
            row = {
                "model": modelname,
                "benchmark": benchmark,
                "category": cat,
                "n_samples": len(items),
            }

            for cond in ALL_CONDITIONS:
                if cond in present:
                    row[f"{cond}_acc"] = accuracy(items, cond)

            # Dominant zero-shot error type for this category slice
            error_counts = {}
            for item in items:
                zs_runs = item["conditions"].get("zero_shot", {}).get("runs", [])
                for run in zs_runs:
                    if not run.get("correct") and run.get("error_type"):
                        et = run["error_type"]
                        error_counts[et] = error_counts.get(et, 0) + 1
            if error_counts:
                top_error = max(error_counts, key=error_counts.get)
                total_errors = sum(error_counts.values())
                row["top_error_type"] = top_error
                row["top_error_pct"] = round(error_counts[top_error] / total_errors * 100, 1)
            else:
                row["top_error_type"] = None
                row["top_error_pct"] = None

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if df[c].dtype == float]
    df[numeric_cols] = df[numeric_cols].round(3)

    return df


def recoveryByErrorAndStrategy(results: list) -> pd.DataFrame:
    from collections import defaultdict

    strategies = [
        "random_few_shot", "targeted_few_shot", "targeted_few_shot_k5",
        "error_targeted_icl", "error_targeted_icl_random",
        "error_targeted_icl_correct_only", "self_consistency",
    ]

    data = defaultdict(lambda: defaultdict(lambda: {"failures": 0, "recovered": 0}))

    for _, _, result in results:
        for item in result:
            conds = item["conditions"]
            zs = conds.get("zero_shot", {})
            if zs.get("majority_correct", True):
                continue
            error_type = None
            for run in zs.get("runs", []):
                if run.get("error_type"):
                    error_type = run["error_type"]
                    break
            if not error_type:
                error_type = "unknown"

            for strat in strategies:
                if strat in conds:
                    data[error_type][strat]["failures"] += 1
                    if conds[strat].get("majority_correct", False):
                        data[error_type][strat]["recovered"] += 1

    rows = []
    for error_type, strat_data in data.items():
        for strat, counts in strat_data.items():
            if counts["failures"] > 0:
                rows.append({
                    "error_type": error_type,
                    "strategy": strat,
                    "failures": counts["failures"],
                    "recovered": counts["recovered"],
                    "recovery_rate": round(counts["recovered"] / counts["failures"], 3),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def loadSpecificResults(filenames: list) -> list:
    loadedFiles = []

    for filename in filenames:
        parts = filename.replace(".json", "").split("__")
        if len(parts) < 2:
            print(f"WARNING: Unexpected file format {filename}. Skipping it")
            continue

        modelname = parts[0].replace("_", "/", 1)
        benchmark = parts[1]

        filePath = os.path.join(RESULTS_DIR, benchmark, filename)

        if not os.path.exists(filePath):
            print(f"WARNING: File not found: {filename}. Skipping.")
            continue

        with open(filePath, encoding="utf-8") as f:
            results = json.load(f)

        loadedFiles.append((modelname, benchmark, results))
        print(f"  Loaded: {filename} ({len(results)} samples)")

    return loadedFiles


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if sys.argv[1:]:
        print(f"Loading {len(sys.argv[1:])} specified file(s)...")
        allResults = loadSpecificResults(sys.argv[1:])
    else:
        print("Loading all result files...")
        allResults = loadResults()

    if not allResults:
        return

    print("\nBuilding summary table...")
    summary = getSummary(allResults)

    if summary.empty:
        print("No data to process.")
        return

    summaryPath = os.path.join(PROCESSED_DIR, "summary.csv")
    summary.to_csv(summaryPath, index=False)
    print(f"\nSummary table saved to: {summaryPath}")
    print("\n" + summary.to_string(index=False))

    print("\nBuilding error distribution table...")
    errors = getErrors(allResults)
    if not errors.empty:
        errorPath = os.path.join(PROCESSED_DIR, "error_distribution.csv")
        errors.to_csv(errorPath, index=False)
        print(f"Error table saved to: {errorPath}")

    print("\nBuilding per-category breakdown table...")
    catBreakdown = getCategoryBreakdown(allResults)
    if not catBreakdown.empty:
        catPath = os.path.join(PROCESSED_DIR, "category_breakdown.csv")
        catBreakdown.to_csv(catPath, index=False)
        print(f"Category breakdown saved to: {catPath}")
        print("\n" + catBreakdown.to_string(index=False))

    print("\nBuilding error × strategy recovery table...")
    recovery = recoveryByErrorAndStrategy(allResults)
    if not recovery.empty:
        recoveryPath = os.path.join(PROCESSED_DIR, "error_strategy_recovery.csv")
        recovery.to_csv(recoveryPath, index=False)
        print(f"Recovery table saved to: {recoveryPath}")

    print("\nNext step: run python analysis/heatmap.py")


if __name__ == "__main__":
    main()