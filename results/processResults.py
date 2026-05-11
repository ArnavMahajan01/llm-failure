import json
import os
import glob
import sys
import pandas as pd
from config import RESULTS_DIR, PROCESSED_DIR
from evaluation.taxonomy import ERROR_TYPES


def loadResults() -> list:

    path = os.path.join(RESULTS_DIR, "*.json")
    files = glob.glob(path)

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

def getSummary(results: list) -> pd.DataFrame:
    rows = []

    for modelname, benchmark, result in results:
        if not result:
            continue

        row = {
            "model": modelname,
            "benchmark": benchmark,
            "n_samples": len(result),
            "zero_shot_acc": accuracy(result, "zero_shot"),
            "random_few_shot_acc": accuracy(result, "random_few_shot"),
            "targeted_few_shot_acc": accuracy(result, "targeted_few_shot"),
            "recovery_rate_random": recoveryRate(
                result, "zero_shot", "random_few_shot"
            ),
            "recovery_rate_targeted": recoveryRate(
                result, "zero_shot", "targeted_few_shot"
            ),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Round all numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype == float]
    df[numeric_cols] = df[numeric_cols].round(3)

    return df


def loadSpecificResults(filenames: list) -> list:
    loadedFiles = []

    for filename in filenames:
        filePath = os.path.join(RESULTS_DIR, filename)

        if not os.path.exists(filePath):
            print(f"WARNING: File not found: {filename}. Skipping.")
            continue

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

    print("\nNext step: run python analysis/heatmap.py")


if __name__ == "__main__":
    main()