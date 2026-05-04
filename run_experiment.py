import json
import os
import sys
import argparse
import time
from datetime import datetime
import re

from config import (
    SMOKE_TEST, SMOKE_TEST_SAMPLES, NUM_SAMPLES, RESULTS_DIR, MAX_NEW_TOKENS
)
from models.model_loader import load_model, generate_response
from data.data_loader import load_benchMark
from prompts.zeroShot import zeroShotPrompt
from evaluation.scorer import score


def run_liad():
    model, _ = load_model("Qwen/Qwen3.5-9B-Instruct")
    print(f"Running model: {model}")

def check_answer(predicted:str, groundTruth: str) -> bool:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", predicted)
    if not numbers:
        return False
    lastNumber = numbers[-1]
    return lastNumber.strip() == groundTruth.strip()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM failure mode experiments")
    parser.add_argument("--smoke", action="store_true", help="Quick test with 5 samples")
    parser.add_argument("--model", type=str, default=None, help="Run only this model family")
    parser.add_argument("--benchmark", type=str, default=None, help="Run only this benchmark")
    parser.add_argument("--condition", type=str, default=None, help="Run only this condition")
    return parser.parse_args()

def isMajorityCorrect(totalRuns: list) -> bool:
    """Just checking if more than half of the questions are correct or not
    """

    totalCount = sum(1 for i in totalRuns if i["correct"])
    return totalCount > len(totalRuns) / 2

def saveResult(results: list, model_name: str, benchmark: str):
    """Save results to a JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = model_name.replace("/", "_").replace(".", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{RESULTS_DIR}/{name}__{benchmark}__{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {filename}")
    return filename

def run_single_question(
    model,
    tokenizer,
    question: str,
    answer: str,
    condition: str,
    benchmark: str,
    example_pool: list,
    run_idx: int
) -> dict:
    """
    Run inference on a single question under one condition.
    Returns a result dict.
    """
    assigned_failure_type = None

    # Build the prompt
    if condition == "zero_shot":
        prompt = zeroShotPrompt(question, benchmark)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Generate response
    try:
        response = generate_response(model, prompt)
    except Exception as e:
        print(f"      ERROR during inference: {e}")
        response = ""

    # Score the response
    scored = score(response, answer, benchmark)

    return {
        "run": run_idx,
        "condition": condition,
        "correct": scored["correct"],
        "predicted": scored["predicted"],
        "ground_truth": answer,
        "assigned_failure_type": assigned_failure_type,
        "full_response": response
    }

def main():
    args = parse_arguments()

    smoke = args.smoke or SMOKE_TEST
    num_samples = SMOKE_TEST_SAMPLES if smoke else NUM_SAMPLES

    """
        This is basic example on how to call the model from ollama and print the result
    """
    # model, _ = load_model("Qwen/Qwen3.5-9B-Instruct")
    
    # response = generate_response(
    #     model,
    #     "What is 2 + 2? Give me just the number."
    # )
    
    # print(f"Response: {response}")

    """
        This is a basic example to see first few questions in the dataset loaded. Change numSamples to change the number of question
    """
    # model, tokenizer = load_model("Qwen/Qwen3.5-9B-Instruct")
    
    # # Load just 3 questions from the simplest benchmark
    # data = load_benchMark("gsm_symbolic", numSamples=3)
    
    # # Print the first question to see what it looks like
    # print("Question:", data[0]["question"])
    # print("Answer:", data[0]["answer"])
    # print("Category:", data[0]["category"])
    # print()
    
    # # Now send it to the model
    # response = generate_response(
    #     model,
    #     data[0]["question"]
    # )
    
    # print(f"Model Response: {response}")

    """
        This is a basic example to see whether the predicted and actual answer are matchng or not
    """
    model, tokenizer = load_model("Qwen/Qwen3.5-9B-Instruct")
    data = load_benchMark("gsm_symbolic", numSamples=30)
    
    all_results = []
    
    for i, item in enumerate(data):
        print(f"[{i+1}/30] Processing...")
        result = run_single_question(
            model, tokenizer,
            item["question"],
            item["answer"],
            condition="zero_shot",
            benchmark="gsm_symbolic",
            example_pool=[],
            run_idx=i
        )

        all_results.append(result)
    
    saveResult(all_results, "qwen3.5", "gsm_symbolic")
    
    # Print summary
    correct = sum(1 for r in all_results if r["correct"])
    print(f"\nAccuracy: {correct}/30 ({correct/30*100:.1f}%)")


if __name__ == "__main__":
    main()
