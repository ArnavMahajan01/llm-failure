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
    "unknown": (
        "The error is not listed in the model at all."
    )
}


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
        
    # Distractor
    distractorWords = [
        "favourite", "favorite", "years old", "works as", "lives in",
        "was born", "colour", "color", "named", "married", "hobby"
    ]
    irrelevant_in_question = [m for m in distractorWords if m in questionLower]
    if irrelevant_in_question:
        # Check if model incorporated these irrelevant details in reasoning
        irrelevant_in_response = [m for m in irrelevant_in_question if m in responseLower]
        if irrelevant_in_response:
            return "distractor"
    
    chainkeywords = ["than", "times as", "twice", "therefore", "greater", "taller"]
    if any(kw in questionLower for kw in chainkeywords):

        has_steps = (
            "step" in responseLower or
            any(f"step {i}" in responseLower for i in range(1, 6)) or
            response.count("\n") < 2
        )
        if not has_steps:
            return "multi_hop"
        
    multipleSteps = (
        "step 1" in responseLower or
        "first" in responseLower and "then" in responseLower or
        response.count("\n") >= 3
    )
    if multipleSteps:
        return "drifting"

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
    
