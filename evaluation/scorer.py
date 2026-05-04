import re

def extractResponse(response: str, benchMark: str) -> str:

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
    

    if benchMark in ["gsm_symbolic"]:
        numVal = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", strippedResponse)
        if numVal:
            return numVal
        
    lastLine = [line.strip() for line in strippedResponse.split("\n") if line.strip()]
    if lastLine:
        return lastLine[-1]

    return ""

def normalize(answer: str) -> str:
    if not answer:
        return ""
    
    answer = answer.strip().lower().replace(",", "")
    answer = answer.re.sub(r"[$£€¥]", "", answer)
    answer = re.sub(r"\.$", "", answer).strip()

    return answer


def isCorrect(predictedAnswer: str, actualAnswer: str) -> bool:

    return True

