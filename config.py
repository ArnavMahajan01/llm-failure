SMOKE_TEST = False
SMOKE_TEST_SAMPLES = 10

# Number of samples to evaluate per benchmark
NUM_SAMPLES = 200
MAX_NEW_TOKENS = 8192

# OLLAMA BASE URL
OLLAMA_BASE_URL = "http://localhost:11434"

RESULTS_DIR = "results/raw"

MODELS = {

}

BENCHMARKS = {
    "gsm_symbolic",
    "gsm_plus",
    "gsm_ic",
    "bigbench_hard",
    "folio"
}

CONDITIONS = [
    "zero_shot",
    "random_few_shot",
    "targeted_few_shot"
]