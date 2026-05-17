from datasets import load_dataset

def load_benchMark(benchMarkName: str, numSamples: int = 200) -> list:
    """
    Load a benchmark and return standardised list of samples.

    Args:
        benchMarkName (str): One of the dataset which are used from hugging face
        numSamples (int, optional): Max number of samples to return. Defaults to 200.

    Returns:
        list: List of dicts with keys: question, answer, category
    """

    if benchMarkName == "gsm_symbolic":
        return _load_gsm_symbolic(numSamples)
    elif benchMarkName == "gsm_plus":
        return _load_gsm_plus(numSamples)
    elif benchMarkName == "gsm_ic":
        return _load_gsm_ic(numSamples)
    elif benchMarkName == "bigbench_hard":
        return _load_bigbench_hard(numSamples, "logical_deduction_five_objects", "logical_deduction")
    elif benchMarkName == "bigbench_hard_tracking":
        return _load_bigbench_hard(numSamples, "tracking_shuffled_objects_five_objects", "object_tracking")
    elif benchMarkName == "folio":
        return _load_folio(numSamples)
    elif benchMarkName == "gsm8k":
        return _load_gsm8k(numSamples)
    else:
        raise ValueError(f"Unknown benchmark: {benchMarkName}")
    
def _load_gsm_symbolic(numSamples: int) -> list:
    try: 
        dataset = load_dataset("apple/GSM-Symbolic", split="test")
        sample = []

        # print(f"Samples {dataset[0]}" )

        for item in dataset:
            rawAnswer = item.get("answer", "")
            if "####" in rawAnswer:
                answer = rawAnswer.split("####")[-1].strip()
            else:
                answer = rawAnswer

            sample.append({
                "question": item.get("question", ""),
                # "answer": str(item.get("answer", "")),
                "answer": answer,
                "category": "arithmetic_symbolic"
            })
            if len(sample) >= numSamples:
                break
        
        return sample
    except Exception as error:
        print(f"ERROR: Could not load gsm-symbolic: {error}")
        return list
    
def _load_gsm_plus(numSamples: int) -> list:
    try: 
        dataset = load_dataset("qintongli/GSM-Plus", split="test")
        sample = []

        # print(f"Samples {dataset[0]}" )

        for item in dataset:
            rawAnswer = item.get("answer", "")
            if "####" in rawAnswer:
                answer = rawAnswer.split("####")[-1].strip()
            else:
                answer = rawAnswer

            sample.append({
                "question": item.get("question", ""),
                "answer": str(item.get("answer", "")),
                # "answer": answer,
                "category": "arithmetic_perturbed"
            })
            if len(sample) >= numSamples:
                break
        
        return sample
    except Exception as error:
        print(f"ERROR: Could not load gsm-symbolic: {error}")
        return list
    
def _load_folio(num_samples: int) -> list:
    try:
        dataset = load_dataset("yale-nlp/FOLIO", split="validation")
        samples = []

        for item in dataset:
            # Combine premises and conclusion into a single question
            premises = item.get("premises", "")
            conclusion = item.get("conclusion", "")

            if isinstance(premises, list):
                premises = " ".join(premises)
            question = f"Given the following statements:\n{premises}\n\nIs the following true, false, or uncertain?\n{conclusion}"

            samples.append({
                "question": question,
                "answer": str(item.get("label", "")),
                "category": "formal_logic"
            })
            if len(samples) >= num_samples:
                break
        return samples
    except Exception as e:
        print(f"  WARNING: Could not load FOLIO: {e}")
        return []

def _load_gsm_ic(numSamples: int) -> list:
    try:
        dataset = load_dataset("voidful/GSM-IC", split="validation")
        sample=[]

        for item in dataset:
            rawAnswer = item.get("answer", "")
            if "####" in rawAnswer:
                answer = rawAnswer.split("####")[-1].strip()
            else:
                answer = rawAnswer

            sample.append({
                "question": item.get("question", ""),
                "answer": str(item.get("answer", "")),
                # "answer": answer,
                "category": "arithmetic_word_problem"
            })
            if len(sample) >= numSamples:
                break
        
        return sample
    except Exception as error:
        print(f"ERROR: Could not load gsm-ic: {error}")
        return list
    
def _load_bigbench_hard(numSamples: int, subset: str, category: str) -> list:
    try:
        dataset = load_dataset("lukaemon/bbh", subset, split="test")
        sample = []

        for item in dataset:
            sample.append({
                "question": item.get("input", ""),
                "answer": item.get("target", ""),
                "category": category
            })
            if len(sample) >= numSamples:
                break
                
        return sample
    except Exception as error:
        print(f"ERROR: Could not load bigbench-hard {subset}: {error}")
        return []
    
def _load_gsm8k(numSamples: int) -> list:
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        sample = []

        for item in dataset:
            rawAnswer = item.get("answer", "")
            if "####" in rawAnswer:
                answer = rawAnswer.split("####")[-1].strip()
            else:
                answer = rawAnswer

            sample.append({
                "question": item.get("question", ""),
                "answer": answer,
                "category": "arithmetic_word_problem"
            })
            if len(sample) >= numSamples:
                break

        return sample
    except Exception as error:
        print(f"ERROR: Could not load gsm8k: {error}")
        return []