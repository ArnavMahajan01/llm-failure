# Builds prompts with no examples.
# This is the baseline condition for all experiments.

def zeroShotPrompt(question: str, benchmark: str) -> str:
    if benchmark == "gsm_symbolic":
        instructions = (
            "You are a math teacher. Solve the following math problem step by step. Final answer will not be given\n"
            "Think about each step carefully\n"
            "At the end, Give the final answer in a new line starting with 'Answer: '"
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