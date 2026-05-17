import re
from collections import Counter

def extractResponse(response: str, benchmark: str) -> str:

    if not response:
        return ""

    # To remove the whitespaces from front and back
    strippedResponse = response.strip();

    # Search for all "Answer: ..." occurrences and pick the first one that
    # isn't inside a LaTeX \text{} macro (those start with a stray '}').
    for matchAnswer in re.finditer(
        r"answer\s*:\s*(.+?)(?:\n|$)",
        strippedResponse,
        re.IGNORECASE
    ):
        value = matchAnswer.group(1).strip()
        # Skip captures that are continuations of \text{Answer:} LaTeX noise
        if value.startswith("}"):
            continue
        value = re.sub(r"[.,;*]+$", "", value).strip()
        # Strip common LaTeX wrappers like \boxed{...} and extract inner text
        boxed = re.search(r"\\boxed\{([^}]+)\}", value)
        if boxed:
            value = boxed.group(1).strip()
        if value:
            return value
    
    # For FOLIO / logic benchmarks, look for True/False/Uncertain
    if benchmark == "folio":
        for label in ["True", "False", "Uncertain"]:
            if re.search(rf"\b{label}\b", strippedResponse, re.IGNORECASE):
                return label

    if benchmark in ["gsm_symbolic", "gsm_plus", "gsm_ic", "gsm8k"]:
        numVal = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", strippedResponse)
        if numVal:
            return numVal[-1].replace(",", "")

    if benchmark in ["bigbench_hard", "bigbench_hard_tracking"]:
        match = re.search(r"\\boxed\{([A-Ea-e])\}", strippedResponse)
        if match:
            return f"({match.group(1).upper()})"
        
    lastLine = [line.strip() for line in strippedResponse.split("\n") if line.strip()]
    if lastLine:
        return lastLine[-1]

    return ""

def normalize(answer: str) -> str:
    if not answer:
        return ""
    
    answer = answer.strip().lower().replace(",", "")
    answer = re.sub(r"[$£€¥%]", "", answer)
    answer = re.sub(r"\.$", "", answer).strip()

    return answer


def isCorrect(predictedAnswer: str, actualAnswer: str) -> bool:
    predVal = normalize(predictedAnswer)
    actualVal = normalize(actualAnswer)

    if not predVal or not actualVal:
        return False
    
    if predVal == actualVal:
        return True
    
    # For numeric Values
    try:
        pred_num = float(predVal)
        truth_num = float(actualVal)
        if abs(pred_num - truth_num) < 1e-5:
            return True
    except ValueError:
        pass

    # For Folio
    for label in ["true", "false", "uncertain"]:
        if label in predVal and label in actualVal:
            return True
        
    # Substring containment only for non-numeric answers (e.g. FOLIO labels)
    # Avoided for numbers: "7" would incorrectly match inside "17"
    def is_numeric(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    if not is_numeric(actualVal) and not is_numeric(predVal):
        if actualVal in predVal or predVal in actualVal:
            return True

    return False

def majorityVote(predictions: list) -> str:
    cleaned = [normalize(p) for p in predictions if p]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def score(modelResponse: str, actualVal: str, benchmark: str = "general") -> dict:
    """
    Full scoring pipeline for a single model response.

    Args:
        modelResponse:      Full model response string
        actualVal:  True answer
        benchmark:     Benchmark name for extraction strategy

    Returns:
        Dict with keys: predicted, actualVal, correct, full response
    """

    predictedVal = extractResponse(modelResponse, benchmark)
    correct = isCorrect(predictedVal, actualVal)

    return {
        "predicted": predictedVal,
        "ground_truth": actualVal,
        "correct": correct,
        "full_response": modelResponse
    }