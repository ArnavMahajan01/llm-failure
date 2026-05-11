# LLM Failure Mode Analysis
## COMP6242 Deep Learning Group Project

---

## PROJECT OVERVIEW

This project investigates where large language models fail on reasoning. (MORE DETAIL IN THE FUTURE)

At the moment, it is working for **gsm-symbolic, gsm_plus and folio with Qwen3.5 9B parameter model**.

## SETUP

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up ollama Model
At the moment it only supports installing models from ollama locally on the machine. Capturing the models from hugging face and running on the server is not made.

Go to ollama website and download ollama. once done check with **ollama** if the ollama is installed or not

To download the model locally, run **ollama run <model_name>**

For Qwuen3.5 run **ollama run qwen3.5**. This will install latest model of ollama with 9B parameters. To take a smaller or a larger model just type in **ollama run <model_name>:<parameter_size>**.
For example, for qwen3.5 with 0.8b parametr run **ollama run qwen3.5:0.8b**.

If or when you install a new model you need to add them in **config.py and model_loader.py** in the similar manner as given.

## Running Experiments

At any time the config values can be changed.

### Quick Smoke Test (To check if everything works or not)
```bash
python3 run_experiment.py --smoke
```

### Single model family

```bash
python3 run_experiment.py --model qwen
```

**NOTE: Please note that it takes the model family as paramter in --model and not a single model. Giving the model family like qwen or gemma will process results on all the models present in the model family in config**

### Single benchmark

```bash
python3 run_experiment.py --benchmark folio
```

### Single condition

```bash
python3 run_experiment.py --condition zero_shot
```

### Full run (all models, all benchmarks, all conditions)

```bash
python3 run_experiment.py
```



**NOTE: You can make a combination of the args to check and test different scenarios**

**Expected runtime:** Full run takes several hours depending on GPU.

Recommended order: run smoke test first, then run one model family at a time.

## Generating Results

(WORK IN PROGESS FROM THIS POINT ONWARDS)

If you want to process results run **processResults.py**. it can be a wither all the files in the results/raw folder or a combination of files. You can run it as follows

```bash
python3 -m results.processResults Gemma_Gemma3-1B__gsm_symbolic__20260511_2100.json

```

### Raw Results

At the moment the full and final raw files are as follows

**FOR ~1 Billion Params**

1. Gemma_Gemma3-1B__gsm_symbolic__20260511_2100.json
2. Qwen_Qwen2-math-1-5B__gsm_symbolic__20260511_2148.json
3. Llama_Llama3-2-1-5B__gsm_symbolic__20260511_2210.json