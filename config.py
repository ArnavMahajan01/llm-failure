SMOKE_TEST = False
SMOKE_TEST_SAMPLES = 2

# Number of samples to evaluate per benchmark
NUM_SAMPLES = 200
MAX_NEW_TOKENS = 8192

# OLLAMA BASE URL
OLLAMA_BASE_URL = "http://localhost:11434"

RESULTS_DIR = "results/raw"
PROCESSED_DIR = "results/processed"

MODELS = {
    "qwen": [
         "Qwen/Qwen2-math-1.5B",
    ],
    "gemma": [
        "Gemma/Gemma3-1B"
    ],
    "llama": [
        "Llama/Llama3.2-1.5B"
    ]
}

BENCHMARKS = {
    "gsm_symbolic",
    "gsm_plus",
    "gsm_ic",
    "bigbench_hard",
    "folio"
}

CONDITIONS = [
    "zero_shot_baseline",
    "zero_shot",
    "targeted_few_shot_answer_only",
    "random_few_shot",
    "targeted_few_shot",
    "targeted_few_shot_k5",
]

NUM_EXAMPLES = 3
NUM_EXAMPLES_K5 = 5
# Number of examples reserved from benchmark data for random few-shot pool
FEW_SHOT_POOL_SIZE = 20
# Number of times to run each prompt on same model
NUM_RUNS = 1
