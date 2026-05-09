TARGETED_EXAMPLES = {
    "distractor": [
        {
            "question": (
                "John Doe has 18 apples in his bag. His favourite color is red and he lives in Canberra.\n"
                "He then proceeds to give 6 apples to Monica Geller, who is 26 years old and work as a software developer\n"
                "How many apples does John Doe have left?"
            ),
            "solution": (
                "I will identify the relevant information and ignore the rest of the information.\n"
                "Relevant: John Does has 18 apples, and gives 6 to Monica Geller.\n"
                "Irrelevant (ignored): John Doe's favourite color and place of residence and Monica Geller's age and profession\n"
                "Calculation: 18 - 6 = 12"
            ),
            "answer": "12",
        }
    ],
    # Demonstrating step by step solution
    "multi_hop": [
        {
            "question": (
                "Ross Geller earns twice as much as Chandler Bing. Chandler Bing earns three times as much as Joey Tribbiani.\n"
                "Joey earns $500 per weel. How much does Ross Geller earn per week?"
            ),
            "solution": (
                "I will solve the problem one step at a time.\n"
                "Step 1: Joey Tribbiani earns $500 per week.\n"
                "Step 2: Chandler Bing earns 3 times as compared to Joey per week= 3 * 500 = $1500.\n"
                "Step 3: Ross earns 2 times as compared to Chandler = 2 * 1500 = $3000\n"
                "Answer: Ross Geller earns $3000 per week."
            ),
            "answer": "3000",
        }
    ],

    # Single negated or Double neagted conditions
    "negation": [
        {
            "question": (
                "All mammals are not reptiles. Dogs are mammals.\n "
                "Is a dog a repltile?"
            ),
            "solution": (
                "Step 1: All mammals are not reptiles. This means that is something is a mammal, it is not a reptile.\n"
                "Step 2: Dogs are mammals.\n"
                "Step 3: Applying this rule, dogs are not reptiles.\n"
                "Answer: No, a dog is not a repltile."
            ),
            "answer": "No"
        }
    ],

    # Arithmetic / Default questions
    "arithmetic": [
        {
            "question": (
                "A rectangle has a length og 8 cm and a width of 5 cm. What is the area of the rectangle?"
            ),
            "solution": (
                "Area  of rectangle = length * width\n"
                " = 8 * 5\n"
                " = 40 cm squared"
            ),
            "answer": "40"
        }
    ]
}

DISTRACTOR_KEYWORDS = [
    "favorite", "favourite", "lives in", "works as", "years old"
]

NEGATION_KEYWORDS = [
    "not", "nor", "nothing"
]

MULTI_HOP_KEYWORDS = [
    "greater than", "more than", "less than", "earns", "times as",
    "twice", "three times"
]

def classifyQuestions(question: str) -> str:
    questionLowered = question.lower()

    distractor_score = sum( 1 for i in DISTRACTOR_KEYWORDS if i in questionLowered)
    negation_score = sum(1 for i in NEGATION_KEYWORDS if i in questionLowered)
    multi_hop_score = sum(1 for i in MULTI_HOP_KEYWORDS if i in questionLowered)

    scores = {
        "distractor": distractor_score,
        "negation": negation_score,
        "multi_hop": multi_hop_score,
        "arithmetic": 0  # default
    }

    bestCategory = max(scores, key = scores.get)

    if scores[bestCategory] == 0:
        return "arithmatic"
    
    return bestCategory


def targetedFewShotPrompt(question: str, benchmark: str, numExamples: int = 3) -> str:
    failureType = classifyQuestions(question)

    examples = TARGETED_EXAMPLES.get(failureType, TARGETED_EXAMPLES["arithmetic"])
    exampleToUse = examples[:numExamples]

    exampleBlock = ""
    for i, example in enumerate(exampleToUse, 1):
        exampleBlock += (
            f"Example {i}:\n"
            f"Problem: {example['question']}"
            f"Solution: {example['solution']}"
            f"Answer: {example['answer']}"
        )

    failureInstruction = {
        "distractor": (
            "The following examples show how to identify RELEVANT information and ignore IRRELEVANT information in a problem. Focus only on what is necessary."
        ),
        "multi_hop": (
            "The following example show how to solve problems that require reasoning steps. Work through each step explicitly before moving to the next step."
        ),
        "negation": (
            "The following examples show how to carefully handle negations and negative conditions. Always unpack negaton step by step before concluding"
        ),
        "arithmetic": (
            "The following example show steo by step arithmetic reasoning. Show every calculation step."
        )
    }

    instruction = failureInstruction.get(failureType, failureInstruction["arithmetic"])

    suffix = "Write your final answer as 'Answer: [number of value]'"

    prompt = f"""{instruction}

    {exampleBlock}Now apply the same approach to solve this problem:

    Problem:
    {question}

    Solution:
    ({suffix})
    """
    return prompt, failureType


