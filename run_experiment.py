import json
import os
import sys
import argparse
import time
from datetime import datetime
import re

from config import (
    SMOKE_TEST, SMOKE_TEST_SAMPLES, NUM_SAMPLES, RESULTS_DIR, MAX_NEW_TOKENS, NUM_EXAMPLES, FEW_SHOT_POOL_SIZE, NUM_RUNS, MODELS, BENCHMARKS, CONDITIONS
)
from models.model_loader import load_model, generate_response, free_model
from data.data_loader import load_benchMark
from prompts.zeroShot import zeroShotPrompt
from prompts.randomFewShot import randomFewShotPrompt
from prompts.targetedFewShots import targetedFewShotPrompt
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

    elif condition == "random_few_shot":
        prompt = randomFewShotPrompt(question=question, exampleList=example_pool, benchmark=benchmark, numExamples=NUM_EXAMPLES, seed=run_idx)
    
    elif condition == "targeted_few_shot":
        prompt, failureType = targetedFewShotPrompt(question=question, benchmark=benchmark, numExamples=NUM_EXAMPLES)

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

def runbenchmarkOnModel(model, tokenizer, modelname: str, benchmark: str, conditionRunning: list, numSamples: int) -> list:
    # This runs all the conditions on all the samples for one model and benchmark combination

    print(f"\n Loading benchmark: {benchmark}")
    data = load_benchMark(benchmark, numSamples=numSamples + FEW_SHOT_POOL_SIZE)

    if not data:
        print(f"ERROR: There was an error in loading data for {benchmark}.")
        return []
    
    poolSize = data[:FEW_SHOT_POOL_SIZE]
    evaluationSize = data[FEW_SHOT_POOL_SIZE:FEW_SHOT_POOL_SIZE + numSamples]

    examplespool = []
    for item in poolSize:
        examplespool.append({
            "question": item["question"],
            "answer": item["answer"],
            "solution": f"Working through this problem step by step and carefully, the answer is {item['answer']}"
        })

    allResults = []
    total = len(evaluationSize)

    for i, item in enumerate(evaluationSize):
        question = item["question"]
        answer = item["answer"]

        print(f"[{i+1} / {total}] Evaluting Wuestion")

        result = {
            "question": question,
            "ground_truth": answer,
            "benchmark_category": item.get("category", "unknown"),
            "conditions": {}
        }

        for condition in conditionRunning:
            runs = []
            for run_idx in range(NUM_RUNS):
                itemResult = run_single_question(
                    model = model,
                    tokenizer = tokenizer,
                    question = question,
                    answer = answer,
                    condition=condition,
                    benchmark=benchmark,
                    example_pool=examplespool,
                    run_idx=run_idx
                )
                runs.append(itemResult)

            result["conditions"][condition] = {
                "majority_correct": isMajorityCorrect(runs),
                "correct_count": sum(1 for res in runs if res["correct"]),
                "total_runs": NUM_RUNS,
                "runs": runs
            }
        
        allResults.append(result)
    
    print(f" Completed {total} questions for {benchmark}")
    return allResults

def main():
    args = parse_arguments()

    smoke = args.smoke or SMOKE_TEST
    num_samples = SMOKE_TEST_SAMPLES if smoke else NUM_SAMPLES

    modelToRun = MODELS
    if args.model:
        if args.model not in MODELS:
            print(f"ERROR: The given model is not defined in the config list. Please check if the model '{args.model}' exists.  \n")
            print(f"Choose model from: {list(MODELS.keys())}")
            sys.exit(1)
        modelToRun = {args.model: MODELS[args.model]}

    benchmarksToRun = BENCHMARKS
    if args.benchmark:
        if args.benchmark not in BENCHMARKS:
            print(f"ERROR: The given benchmark is not defined in the config list. Please check if the benchmark '{args.benchmark}' exists.  \n")
            print(f"Choose benchmark from: {BENCHMARKS}")
            sys.exit(1)
        benchmarksToRun = [args.benchmark]

    conditionToRun = CONDITIONS
    if args.condition:
        if args.condition not in CONDITIONS:
            print(f"ERROR: The given condition is not defined in the config list. Please check if the condition '{args.condition}' exists")
            print(f"Choose condition from: {CONDITIONS}")
            sys.exit(1)
        conditionToRun = [args.condition]

    if smoke:
        print(f"SMOKE TEST MODE: {num_samples} samples per benchmark")
    else:
        print(f"FULL RUN: {num_samples} samples per benchmark")

    print(f"Models: {list(modelToRun.keys())}")
    print(f"Benchmarks: {benchmarksToRun}")
    print(f"Conditions: {conditionToRun}")
    print(f"Runs per questions: {NUM_RUNS}")

    startTime = time.time()

    for modelFamily, modelList in modelToRun.items():
        for modelname in modelList:
            print(f"\n===========")
            print(f"MODEL: {modelname}")
            print(f"\n===========")

            # Load Model Here
            try:
                model, tokenizer = load_model(modelname)
            except Exception as error:
                print(f"ERROR: Error loading model {modelname}: {error}")
                print(f"Skipping the model.")
                continue
            
            for benchmark in benchmarksToRun:
                results = runbenchmarkOnModel(
                    model=model,
                    tokenizer=tokenizer,
                    modelname=modelname,
                    benchmark=benchmark,
                    conditionRunning=conditionToRun,
                    numSamples=num_samples
                )

                if results:
                    saveResult(results, modelname, benchmark)

            free_model(model)

    elapsedTime = time.time() - startTime
    print(f"=============")
    print(f"All experiments are completed. Total time taken: {elapsedTime/60:.1f} minutes")
    print(f"Results are saved to: {RESULTS_DIR}/")


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
    # conditions = ["zero_shot", "random_few_shot", "targeted_few_shot"]
    # model, tokenizer = load_model("Qwen/Qwen3.5-9B-Instruct")
    # data = load_benchMark("gsm_symbolic", numSamples=25)
    # example_pool = data[:20]   # first 20 as pool
    # eval_data = data[20:]
    
    # all_results = []

    # for i, item in enumerate(eval_data):
    #     print(f"[{i+1}/{len(eval_data)}] Processing...")
    #     item_result = {
    #         "question": item["question"],
    #         "ground_truth": item["answer"],
    #         "conditions": {}
    #     }

    #     for condition in conditions:
    #         result = run_single_question(
    #             model, tokenizer,
    #             item["question"],
    #             item["answer"],
    #             condition=condition,
    #             benchmark="gsm_symbolic",
    #             example_pool=example_pool,
    #             run_idx=i
    #         )
    #         item_result["conditions"][condition] = result

    #     all_results.append(item_result)

    #     # Save after each question
    #     with open(f"{RESULTS_DIR}/qwen3.5__gsm_symbolic.json", "w", encoding="utf-8") as f:
    #         json.dump(all_results, f, indent=2, ensure_ascii=False)

    # # Print summary AFTER everything is done
    # total = len(all_results)
    # print("\n--- RESULTS SUMMARY ---")
    # for condition in conditions:
    #     correct = sum(
    #         1 for r in all_results 
    #         if r["conditions"][condition]["correct"]
    #     )
    #     print(f"{condition}: {correct}/{total} ({correct/total*100:.1f}%)")





if __name__ == "__main__":
    main()
