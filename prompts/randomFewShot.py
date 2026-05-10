import random
from prompts.zeroShot import zeroShotPrompt

def randomFewShotPrompt(
        question: str,
        exampleList: list,
        benchmark: str,
        numExamples: int = 3,
        seed: int = None
) -> str:
        if seed is not None:
            random.seed(seed)

        availableExamples = [example for example in exampleList if example.get("question", "") != question]
        numToSample = min(numExamples, len(availableExamples))

        if numToSample == 0:
            return zeroShotPrompt(question, benchmark)
        
        selectedQuestions = random.sample(availableExamples, numToSample)

        examples = ""
        for i, ex in enumerate(selectedQuestions, 1):
            ques = ex.get("question", "")
            ans = ex.get("answer", "")
            solution = ex.get("solution", f"Working through the question step by step the answer is {ans}")
            examples += f"Example {i}:\nProblem: {ques}\nSolution: {solution}\nAnswer: {ans}\n\n"

        if benchmark in ["gsm_symbolic", "gsm_plus"]:
            instruction = (
                "Given are some of the math problems with their solution\n"
                "Follow the same step by step approach to solve the new problem.\n"
                "At the end, write the final numeric answer starting with 'Answer:'"
            )
        elif benchmark in ["folio"]:
             instruction = (
                "Given are some reasoning problems. \n"
                "For each question, determine whether the conclusion is True, False, or Uncertain. \n"
                "Follow the same format and at the end write 'Answer: True', 'Answer: False', or 'Answer: Uncertain'. "
             )
        else:
            instruction = (
                "Given are some problems with their solution\n"
                "Follow the same step by step approach to solve the new problem.\n"
                "At the end, write the final answer starting with 'Answer:'"
            )

        prompt = f"""{instruction} 
        {examples}.

        Now solve this problem using the same approach

        Problem: {question}

        Solution:
        """

        return prompt

    
