
def zeroShotPrompt(question: str, benchmark: str) -> str:
    if benchmark == "gsm_symbolic":
        prompt = (
            "You are a math teacher. Solve the following match problem step by step. Final answer will not be given",
            "Think about each step carefully",
            "At the end, Give the final answer in a new line starting with 'Answer: '"
        )
    else: 
        prompt = (
            "Answer the following question step by step. Final answer will not be given",
            "At the end, Give the final answer in a new line starting with 'Answer: '"
        )

    prompt = f"""{prompt} 

    Problem: {question}

    Solution:
    """

    return prompt