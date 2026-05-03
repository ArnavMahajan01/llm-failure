import json
import os
import sys
import argparse
import time
from datetime import datetime

from config import (
    SMOKE_TEST, SMOKE_TEST_SAMPLES, NUM_SAMPLES
)
from models.model_loader import load_model, generate_response

def run_liad():
    model, _ = load_model("Qwen/Qwen3.5-9B-Instruct")
    print(f"Running model: {model}")

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

    # model, _ = load_model("Qwen/Qwen3.5-9B-Instruct")
    
    # response = generate_response(
    #     model,
    #     "What is 2 + 2? Give me just the number."
    # )
    
    # print(f"Response: {response}")

if __name__ == "__main__":
    main()
