# Builds prompts with no examples.
# This is the baseline condition for all experiments.

def zeroShotPrompt(question: str, benchmark: str) -> str:
    if benchmark in ["gsm_symbolic", "gsm_plus"]:
        instructions = (
            "You are a math teacher. Solve the following math problem step by step. Final answer will not be given\n"
            "Think about each step carefully\n"
            "At the end, Give the final answer in a new line starting with 'Answer: '"
        )
    elif benchmark in ["folio"]:
        instructions = (
            "Read the following statements carefully and determine whether the conclusion or answer per se is True or False, or Uncertain based on the information given to you. \n"
            "Think the whole process step by step. \n"
            "At the end, give the final answer in a new line starting with 'Answer: '\n"
            "The answer must be exactly one of: True, False, Uncertain."
        )
    else: 
        instructions = (
            "Answer the following question step by step. Final answer will not be given\n"
            "At the end, Give the final answer in a new line starting with 'Answer: '"
        )

    prompt = f"""{instructions} 

    Problem: {question}

    Solution:
    """

    return prompt