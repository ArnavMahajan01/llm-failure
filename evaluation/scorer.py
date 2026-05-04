import re

def extractResponse(response: str, benchmark: str) -> str:

    if not response:
        return ""

    # To remove the whitespaces from front and back
    strippedResponse = response.strip();

    matchAnswer = re.search(
        r"answer\s*:\s*(.+?)(?:\n|$)",
        strippedResponse,
        re.IGNORECASE
    )

    # if matchAnswer:
    #     value = answerMatch() # SOMETHEING HERE

    #     return value

    if matchAnswer:
        value = matchAnswer.group(1).strip()
        value = re.sub(r"[.,;]$", "", value).strip()
        if value:
            return value
    

    if benchmark in ["gsm_symbolic"]:
        numVal = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", strippedResponse)
        if numVal:
            return numVal[-1].replace(",", "")
        
    lastLine = [line.strip() for line in strippedResponse.split("\n") if line.strip()]
    if lastLine:
        return lastLine[-1]

    return ""

def normalize(answer: str) -> str:
    if not answer:
        return ""
    
    answer = answer.strip().lower().replace(",", "")
    answer = re.sub(r"[$£€¥]", "", answer)
    answer = re.sub(r"\.$", "", answer).strip()

    return answer


def isCorrect(predictedAnswer: str, actualAnswer: str) -> bool:
    predVal = normalize(predictedAnswer)
    actualVal = normalize(actualAnswer)

    if not predVal or not actualVal:
        return False
    
    if predVal == actualVal:
        return True
    
    try:
        pred_num = float(predVal)
        truth_num = float(actualVal)
        if abs(pred_num - truth_num) < 1e-5:
            return True
    except ValueError:
        pass

    # Substring containment (for multi-word answers)
    if actualVal in predVal or predVal in actualVal:
        return True
    #  NNED TO CHECK FOR OTHER DATASET TYPES
    return False

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

