# Builds prompts with no examples.
# zeroShotBaselinePrompt = S0: bare question, no CoT instruction.
# zeroShotPrompt         = S1: zero-shot with explicit "step by step" CoT trigger.
#
# Category-specific instructions take priority over benchmark-level defaults.
# Add new categories to CATEGORY_INSTRUCTIONS_BASELINE / CATEGORY_INSTRUCTIONS_COT
# to customise behaviour without touching the function logic.

CATEGORY_INSTRUCTIONS_BASELINE = {
    "arithmetic_symbolic": (
        "You are a math teacher. Solve the following math problem.\n"
        "At the end, give the final answer in a new line starting with 'Answer: '"
    ),
    "arithmetic_word_problem": (
        "You are a math teacher. Read the word problem carefully and identify only the relevant numbers.\n"
        "At the end, give the final answer in a new line starting with 'Answer: '"
    ),
    "arithmetic_perturbed": (
        "You are a math teacher. Solve the following math problem.\n"
        "At the end, give the final answer in a new line starting with 'Answer: '"
    ),
    "logical_deduction": (
        "You are a logical reasoning expert. Read the clues and deduce the correct ordering or selection.\n"
        "At the end, give the final answer's option in a new line starting with 'Answer: '"
    ),
    "object_tracking": (
        "You are a careful analyst. Track each object or entity through every step described.\n"
        "At the end, give the final answer's option in a new line starting with 'Answer: '"
    ),
    "formal_logic": (
        "Read the following statements carefully and determine whether the conclusion is True, False, or Uncertain.\n"
        "At the end, give the final answer in a new line starting with 'Answer: '\n"
        "The answer must be exactly one of: True, False, Uncertain."
    ),
}

CATEGORY_INSTRUCTIONS_COT = {
    "arithmetic_symbolic": (
        "You are a math teacher. Solve the following math problem step by step.\n"
        "Think about each step carefully.\n"
        "At the end, give the final answer in a new line starting with 'Answer: '"
    ),
    "arithmetic_word_problem": (
        "You are a math teacher. Read the word problem step by step, identify only the relevant numbers, and ignore any distractors.\n"
        "Show all arithmetic clearly.\n"
        "At the end, give the final answer in a new line starting with 'Answer: '"
    ),
    "arithmetic_perturbed": (
        "You are a math teacher. Solve the following math problem step by step.\n"
        "Think about each step carefully.\n"
        "At the end, give the final answer in a new line starting with 'Answer: '"
    ),
    "logical_deduction": (
        "You are a logical reasoning expert. Work through the clues one by one to deduce the answer.\n"
        "Think the whole process step by step.\n"
        "At the end, give the final answer's option in a new line starting with 'Answer: '"
    ),
    "object_tracking": (
        "You are a careful analyst. Simulate each swap or move in order, tracking every object's position.\n"
        "Think the whole process step by step.\n"
        "At the end, give the final answer's option in a new line starting with 'Answer: '"
    ),
    "formal_logic": (
        "Read the following statements carefully and determine whether the conclusion is True, False, or Uncertain.\n"
        "Think the whole process step by step.\n"
        "At the end, give the final answer in a new line starting with 'Answer: '\n"
        "The answer must be exactly one of: True, False, Uncertain."
    ),
}

# Benchmark-level fallbacks (used when category is None or unrecognised)
BENCHMARK_BASELINE_FALLBACK = {
    "gsm_symbolic": CATEGORY_INSTRUCTIONS_BASELINE["arithmetic_symbolic"],
    "gsm_plus": CATEGORY_INSTRUCTIONS_BASELINE["arithmetic_perturbed"],
    "gsm_ic": CATEGORY_INSTRUCTIONS_BASELINE["arithmetic_word_problem"],
    "gsm8k": CATEGORY_INSTRUCTIONS_BASELINE["arithmetic_word_problem"],
    "folio": CATEGORY_INSTRUCTIONS_BASELINE["formal_logic"],
    "bigbench_hard": CATEGORY_INSTRUCTIONS_BASELINE["logical_deduction"],
    "bigbench_hard_tracking": CATEGORY_INSTRUCTIONS_BASELINE["object_tracking"],
}

BENCHMARK_COT_FALLBACK = {
    "gsm_symbolic": CATEGORY_INSTRUCTIONS_COT["arithmetic_symbolic"],
    "gsm_plus": CATEGORY_INSTRUCTIONS_COT["arithmetic_perturbed"],
    "gsm_ic": CATEGORY_INSTRUCTIONS_COT["arithmetic_word_problem"],
    "gsm8k": CATEGORY_INSTRUCTIONS_COT["arithmetic_word_problem"],
    "folio": CATEGORY_INSTRUCTIONS_COT["formal_logic"],
    "bigbench_hard": CATEGORY_INSTRUCTIONS_COT["logical_deduction"],
    "bigbench_hard_tracking": CATEGORY_INSTRUCTIONS_COT["object_tracking"],
}

DEFAULT_BASELINE = (
    "Answer the following question.\n"
    "At the end, give the final answer in a new line starting with 'Answer: '"
)

DEFAULT_COT = (
    "Answer the following question step by step.\n"
    "At the end, give the final answer in a new line starting with 'Answer: '"
)


def resolveInstructions(lookup: dict, fallback: dict, category, benchmark) -> str:
    if category and category in lookup:
        return lookup[category]
    if benchmark and benchmark in fallback:
        return fallback[benchmark]
    return None


def zeroShotBaselinePrompt(question: str, benchmark: str, category: str = None) -> str:
    instructions = (
        resolveInstructions(CATEGORY_INSTRUCTIONS_BASELINE, BENCHMARK_BASELINE_FALLBACK, category, benchmark)
        or DEFAULT_BASELINE
    )

    prompt = f"""{instructions}

    Problem: {question}

    Solution:
    """

    return prompt


def zeroShotPrompt(question: str, benchmark: str, category: str = None) -> str:
    instructions = (
        resolveInstructions(CATEGORY_INSTRUCTIONS_COT, BENCHMARK_COT_FALLBACK, category, benchmark)
        or DEFAULT_COT
    )

    prompt = f"""{instructions}

    Problem: {question}

    Solution:
    """

    return prompt