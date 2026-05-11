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
    elif benchMarkName == "folio":
        return _load_folio(numSamples)
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
                "category": item.get("perturbation_type", "arithmetic_perturbed")
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
