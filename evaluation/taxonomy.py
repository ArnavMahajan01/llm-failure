from evaluation.scorer import extractResponse, normalize
import re

ERROR_TYPES = {
    "confidently_wrong": (
        "Model gives a wrong answer confidently, without any uncertinity. This means there are some hallucination steps"
    ),
    "drifting": (
        "Model started reasoning correctly but lost it's way midway. Some of the steps are correct but a mistakes appears partway through."
    ),
    "distractor": (
        "Model is mislead by irrelevant information in the question. Model is capturing irrelevant and non-essentail detail too."
    ),
    "negation": (
        "Model handles negation incorrectly. Either ignores negation eintrely or interpret it wrongly."
    ),
    "multi_hop": (
        "Models drops one or more steps in a chain of reasoning. Skips from an early step to the final answer without completing the chain."
    ),
    "hallucination": (
        "Model introduces facts, numbers, or entities that are not present in the question"
    ),
    "format_correct_wrong_answer": (
         "Model reasoning appears correct but final answer extraction fails. Wrong number pulled from correct reasoning."
    ),
    "off_topic": (
        "Model does not address the question at all. Producing irrelevant context"
    ),
    "premise_order_sensitivity": (
        "Model fails when the order of premises or information is changed, even though all the information is still present."
    ),
    "arithmetic_slip": (
        "Model set up the problem correctly with the right plan but made a calculation mistake in one of the arithmetic steps."
    ),
    "unknown": (
        "The error is not listed in the model at all."
    )
}


def checkArithmeticSlip(response: str) -> bool:
    pattern = r'(\d+(?:\.\d+)?)\s*([+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
    matches = re.findall(pattern, response)
    for a, op, b, result in matches:
        a, b, result = float(a), float(b), float(result)
        if op == '+':
            expected = a + b
        elif op == '-':
            expected = a - b
        elif op == '*':
            expected = a * b
        elif op == '/':
            if b == 0:
                continue
            expected = a / b
        else:
            continue
        if abs(expected - result) > 0.01:
            return True
    return False


def classifyError(question: str, response: str, actualAnswer: str, benchmark: str) -> str:
    if not response or not response.strip():
        return "unknown"

    questionLower = question.lower()
    responseLower = response.lower()

    turthNorm = normalize(actualAnswer)
    if turthNorm and turthNorm in responseLower:
        extractedResponse = extractResponse(response, benchmark)
        if not extractedResponse or normalize(extractedResponse) != turthNorm:
            return "format_correct_wrong_answer"

    # Check off topic
    # Very short response or no reasoning attempt
    if len(response.strip()) < 30:
        return "off_topic"
    if "i cannot" in responseLower or "i don't know" in responseLower or "i do not know" in responseLower or "i can't" in responseLower:
        return "off_topic"

    # negation
    negatedWords = ["not", "neither", "nor", "never", "no one", "nobody",
                    "nothing", "not the case", "not all"]
    question_has_negation = any(w in questionLower for w in negatedWords)
    if question_has_negation:
        # Check if model mentions the negation at all in its reasoning
        responseNegation = any(w in responseLower for w in negatedWords)
        if not responseNegation:
            return "negation"

    # Logical reversal - affirming the consequent (if A then B, B therefore A)
    # Model reverses a conditional even when negation words are present
    conditionalInQuestion = "if " in questionLower and " then " in questionLower
    if conditionalInQuestion:
        reversalPhrases = ["since", "because", "therefore", "thus", "hence"]
        affirmingWords = ["is true", "must be true", "answer: true", "answer: yes", "so yes", "therefore yes"]
        # Check if model uses reversal reasoning AND affirms something it should not
        if any(p in responseLower for p in reversalPhrases) and any(a in responseLower for a in affirmingWords):
            return "negation"

    # Distractor
    distractorWords = [
        "favourite", "favorite", "years old", "works as", "lives in",
        "was born", "colour", "color", "named", "married", "hobby",
        "works at", "has been working", "founded in", "located", "made in",
        "profession", "generation", "acres", "floors", "days a week",
        "years of experience", "age", "hair", "height", "weight"
    ]
    irrelevant_in_question = [m for m in distractorWords if m in questionLower]
    if irrelevant_in_question:
        # Check if model incorporated these irrelevant details in reasoning
        irrelevant_in_response = [m for m in irrelevant_in_question if m in responseLower]
        if irrelevant_in_response:
            return "distractor"

    # Hallucination
    # Check if model introduces numbers or assumptions not present in the question
    questionNumbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', question))
    responseNumbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    newNumbers = [n for n in responseNumbers if n not in questionNumbers]
    hallucinationPhrases = [
        "assuming", "assume", "we know that", "it is given that",
        "by definition", "it is known that", "typically", "in general",
        "as we know", "from our knowledge"
    ]
    phraseInResponse = any(p in responseLower for p in hallucinationPhrases)
    if newNumbers and phraseInResponse:
        return "hallucination"

    # Arithmetic slip
    # Model had the right plan but made a wrong calculation
    if checkArithmeticSlip(response):
        return "arithmetic_slip"

    chainkeywords = [
        "than", "times as", "twice", "therefore", "greater", "taller",
        "triple", "double", "as fast as", "as many as", "as much as",
        "half as", "per hour", "per week", "per day", "each day",
        "for every", "rate of"
    ]
    if any(kw in questionLower for kw in chainkeywords):
        has_steps = (
            "step" in responseLower or
            any(f"step {i}" in responseLower for i in range(1, 6)) or
            response.count("\n") >= 3
        )
        if not has_steps:
            return "multi_hop"

    multipleSteps = (
        "step 1" in responseLower or
        ("first" in responseLower and "then" in responseLower) or
        response.count("\n") >= 3
    )
    if multipleSteps:
        return "drifting"

    # Premise order sensitivity
    # Heuristic for logic benchmarks - multiple premises present, no other error matched
    if benchmark in ["folio", "bigbench_hard"]:
        premises = [s.strip() for s in re.split(r'[.\n]', question) if s.strip()]
        if len(premises) >= 3:
            return "premise_order_sensitivity"

    return "confidently_wrong"

def summarise_errors(results: list) -> dict:

    error_counts = {et: 0 for et in ERROR_TYPES}
    total_errors = 0

    for item in results:
        for condition_data in item.get("conditions", {}).values():
            for run in condition_data.get("runs", []):
                error_type = run.get("error_type")
                if error_type and error_type in error_counts:
                    error_counts[error_type] += 1
                    total_errors += 1

    summary = {}
    for et, count in error_counts.items():
        summary[et] = {
            "count": count,
            "percentage": round(count / total_errors * 100, 1) if total_errors > 0 else 0.0,
            "description": ERROR_TYPES[et]
        }

    return summary
    
