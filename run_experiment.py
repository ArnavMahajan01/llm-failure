import json
import os
import sys
import argparse
import time
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM failure mode experiments")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test with 5 samples")
    parser.add_argument("--model", type=str, default=None, help="Run only this model family")
    parser.add_argument("--benchmark", type=str, default=None, help="Run only this benchmark")
    parser.add_argument("--condition", type=str, default=None, help="Run only this condition")
    return parser.parse_args()