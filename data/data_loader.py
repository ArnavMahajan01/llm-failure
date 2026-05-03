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
                "answer": str(item.get("answer", "")),
                # "answer": answer,
                "category": "arithmetic_symbolic"
            })
            if len(sample) >= numSamples:
                break
        
        return sample
    except Exception as error:
        print(f"ERROR: Could not load gsm-symbolic: {error}")
        return list