import json
import os
import sys
import argparse
import time
from datetime import datetime
import re

from config import (
    SMOKE_TEST, SMOKE_TEST_SAMPLES, NUM_SAMPLES
)
from models.model_loader import load_model, generate_response
from data.data_loader import load_benchMark


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
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test with 5 samples")
    parser.add_argument("--model", type=str, default=None, help="Run only this model family")
    parser.add_argument("--benchmark", type=str, default=None, help="Run only this benchmark")
    parser.add_argument("--condition", type=str, default=None, help="Run only this condition")
    return parser.parse_args()

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
    data = load_benchMark("gsm_symbolic", numSamples=5)

    for i, item in enumerate(data):
        print(f"\nQuestion {i+1}: {item['question']}")
        print(f"Expected Answer: {item['answer']}")
        
        response = generate_response(model, item["question"])
        print(f"Model Response: {response}")
        
        correct = check_answer(response, item["answer"])
        print(f"Correct: {correct}")
        print("-" * 50)



if __name__ == "__main__":
    main()
