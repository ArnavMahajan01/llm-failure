TARGETED_EXAMPLES = {
    "distractor": [
        {
            "question": (
                "John Doe has 18 apples in his bag. His favourite color is red and he lives in Canberra.\n"
                "He then proceeds to give 6 apples to Monica Geller, who is 26 years old and works as a software developer.\n"
                "How many apples does John Doe have left?"
            ),
            "solution": (
                "I will identify the relevant information and ignore the rest of the information.\n"
                "Relevant: John Doe has 18 apples, and gives 6 to Monica Geller.\n"
                "Irrelevant (ignored): John Doe's favourite color and place of residence and Monica Geller's age and profession.\n"
                "Calculation: 18 - 6 = 12"
            ),
            "answer": "12",
        },
        {
            "question": (
                "Sarah has 24 cookies in her jar. Her favourite TV show is Friends and she works as a nurse.\n"
                "She gives 9 cookies to her colleague Tom, who has been working at the hospital for 5 years.\n"
                "How many cookies does Sarah have left?"
            ),
            "solution": (
                "I will identify the relevant information and ignore the rest.\n"
                "Relevant: Sarah has 24 cookies and gives 9 to Tom.\n"
                "Irrelevant (ignored): Sarah's favourite show, her profession, Tom's years of experience.\n"
                "Calculation: 24 - 9 = 15"
            ),
            "answer": "15",
        },
        {
            "question": (
                "Mark collected 45 stamps over the summer. He is 14 years old and has brown hair.\n"
                "His friend Jake, who lives on Oak Street, borrowed 12 stamps from him.\n"
                "How many stamps does Mark have now?"
            ),
            "solution": (
                "I will identify the relevant information and ignore the rest.\n"
                "Relevant: Mark has 45 stamps and Jake borrowed 12.\n"
                "Irrelevant (ignored): Mark's age, hair colour, Jake's street address.\n"
                "Calculation: 45 - 12 = 33"
            ),
            "answer": "33",
        },
        {
            "question": (
                "A store has 80 shirts in stock. The store was founded in 1995 and is located downtown.\n"
                "After a weekend sale, 35 shirts were sold.\n"
                "How many shirts remain in stock?"
            ),
            "solution": (
                "I will identify the relevant information and ignore the rest.\n"
                "Relevant: 80 shirts in stock, 35 sold.\n"
                "Irrelevant (ignored): founding year and store location.\n"
                "Calculation: 80 - 35 = 45"
            ),
            "answer": "45",
        },
        {
            "question": (
                "Emma has $120 in her wallet. She bought a blue dress made in Italy that costs $45.\n"
                "The dress is a size medium and was on display at the front of the shop.\n"
                "How much money does Emma have left?"
            ),
            "solution": (
                "I will identify the relevant information and ignore the rest.\n"
                "Relevant: Emma has $120 and spends $45.\n"
                "Irrelevant (ignored): colour of dress, country of manufacture, size, display position.\n"
                "Calculation: 120 - 45 = 75"
            ),
            "answer": "75",
        },
        {
            "question": (
                "A farmer has 60 chickens on his farm. The farm has been in his family for 3 generations and covers 50 acres.\n"
                "He sold 25 chickens at the Saturday market.\n"
                "How many chickens does he have now?"
            ),
            "solution": (
                "I will identify the relevant information and ignore the rest.\n"
                "Relevant: 60 chickens, 25 sold.\n"
                "Irrelevant (ignored): farm history and acreage.\n"
                "Calculation: 60 - 25 = 35"
            ),
            "answer": "35",
        },
    ],

    # Demonstrating step by step solution
    "multi_hop": [
        {
            "question": (
                "Ross Geller earns twice as much as Chandler Bing. Chandler Bing earns three times as much as Joey Tribbiani.\n"
                "Joey earns $500 per week. How much does Ross Geller earn per week?"
            ),
            "solution": (
                "I will solve the problem one step at a time.\n"
                "Step 1: Joey Tribbiani earns $500 per week.\n"
                "Step 2: Chandler Bing earns 3 times as much as Joey = 3 * 500 = $1500.\n"
                "Step 3: Ross earns 2 times as much as Chandler = 2 * 1500 = $3000.\n"
                "Answer: Ross Geller earns $3000 per week."
            ),
            "answer": "3000",
        },
        {
            "question": (
                "Anna runs twice as fast as Bob. Bob runs three times as fast as Carol.\n"
                "Carol runs at 4 km/h. How fast does Anna run?"
            ),
            "solution": (
                "I will solve the problem one step at a time.\n"
                "Step 1: Carol runs at 4 km/h.\n"
                "Step 2: Bob runs 3 times as fast as Carol = 3 * 4 = 12 km/h.\n"
                "Step 3: Anna runs 2 times as fast as Bob = 2 * 12 = 24 km/h.\n"
                "Answer: Anna runs at 24 km/h."
            ),
            "answer": "24",
        },
        {
            "question": (
                "Tom earns four times as much as Sam. Sam earns twice as much as Lily.\n"
                "Lily earns $300 per week. How much does Tom earn per week?"
            ),
            "solution": (
                "I will solve the problem one step at a time.\n"
                "Step 1: Lily earns $300 per week.\n"
                "Step 2: Sam earns 2 times as much as Lily = 2 * 300 = $600.\n"
                "Step 3: Tom earns 4 times as much as Sam = 4 * 600 = $2400.\n"
                "Answer: Tom earns $2400 per week."
            ),
            "answer": "2400",
        },
        {
            "question": (
                "A car travels at twice the speed of a bicycle. The bicycle travels at three times the speed of a person walking.\n"
                "The person walks at 5 km/h. How fast does the car travel?"
            ),
            "solution": (
                "I will solve the problem one step at a time.\n"
                "Step 1: Walking speed = 5 km/h.\n"
                "Step 2: Bicycle speed = 3 times walking = 3 * 5 = 15 km/h.\n"
                "Step 3: Car speed = 2 times bicycle = 2 * 15 = 30 km/h.\n"
                "Answer: The car travels at 30 km/h."
            ),
            "answer": "30",
        },
        {
            "question": (
                "Box A contains five times as many balls as Box B. Box B contains twice as many balls as Box C.\n"
                "Box C contains 8 balls. How many balls are in Box A?"
            ),
            "solution": (
                "I will solve the problem one step at a time.\n"
                "Step 1: Box C contains 8 balls.\n"
                "Step 2: Box B contains 2 times as many as Box C = 2 * 8 = 16 balls.\n"
                "Step 3: Box A contains 5 times as many as Box B = 5 * 16 = 80 balls.\n"
                "Answer: Box A contains 80 balls."
            ),
            "answer": "80",
        },
        {
            "question": (
                "A factory produces three times as many widgets as a workshop. The workshop produces half as many widgets as a studio.\n"
                "The studio produces 200 widgets per day. How many widgets does the factory produce per day?"
            ),
            "solution": (
                "I will solve the problem one step at a time.\n"
                "Step 1: The studio produces 200 widgets per day.\n"
                "Step 2: The workshop produces half as many as the studio = 200 / 2 = 100 widgets.\n"
                "Step 3: The factory produces 3 times as many as the workshop = 3 * 100 = 300 widgets.\n"
                "Answer: The factory produces 300 widgets per day."
            ),
            "answer": "300",
        },
    ],

    # Single negated or double negated conditions
    "negation": [
        {
            "question": (
                "All mammals are not reptiles. Dogs are mammals.\n"
                "Is a dog a reptile?"
            ),
            "solution": (
                "Step 1: All mammals are not reptiles : if something is a mammal, it is not a reptile.\n"
                "Step 2: Dogs are mammals.\n"
                "Step 3: Applying this rule, dogs are not reptiles.\n"
                "Answer: No, a dog is not a reptile."
            ),
            "answer": "No",
        },
        {
            "question": (
                "No birds are mammals. Penguins are birds.\n"
                "Is a penguin a mammal?"
            ),
            "solution": (
                "Step 1: No birds are mammals : being a bird rules out being a mammal.\n"
                "Step 2: Penguins are birds.\n"
                "Step 3: Therefore, penguins are not mammals.\n"
                "Answer: No, a penguin is not a mammal."
            ),
            "answer": "No",
        },
        {
            "question": (
                "All fish cannot breathe air. Salmon are fish.\n"
                "Can a salmon breathe air?"
            ),
            "solution": (
                "Step 1: All fish cannot breathe air.\n"
                "Step 2: Salmon are fish.\n"
                "Step 3: Therefore, salmon cannot breathe air.\n"
                "Answer: No, a salmon cannot breathe air."
            ),
            "answer": "No",
        },
        {
            "question": (
                "Nothing in the locked box is available for purchase. The diamond ring is in the locked box.\n"
                "Is the diamond ring available for purchase?"
            ),
            "solution": (
                "Step 1: Nothing in the locked box is available for purchase.\n"
                "Step 2: The diamond ring is in the locked box.\n"
                "Step 3: Therefore, the diamond ring is not available for purchase.\n"
                "Answer: No, the diamond ring is not available for purchase."
            ),
            "answer": "No",
        },
        {
            "question": (
                "No students who failed the exam passed the course. Alice failed the exam.\n"
                "Did Alice pass the course?"
            ),
            "solution": (
                "Step 1: Failing the exam means not passing the course.\n"
                "Step 2: Alice failed the exam.\n"
                "Step 3: Therefore, Alice did not pass the course.\n"
                "Answer: No, Alice did not pass the course."
            ),
            "answer": "No",
        },
        {
            "question": (
                "None of the red items are on sale. The red jacket is a red item.\n"
                "Is the red jacket on sale?"
            ),
            "solution": (
                "Step 1: None of the red items are on sale.\n"
                "Step 2: The red jacket is a red item.\n"
                "Step 3: Therefore, the red jacket is not on sale.\n"
                "Answer: No, the red jacket is not on sale."
            ),
            "answer": "No",
        },
    ],

    # Arithmetic / Default questions
    "arithmetic": [
        {
            "question": (
                "A rectangle has a length of 8 cm and a width of 5 cm. What is the area of the rectangle?"
            ),
            "solution": (
                "Area of rectangle = length * width\n"
                " = 8 * 5\n"
                " = 40 cm squared"
            ),
            "answer": "40",
        },
        {
            "question": (
                "A train travels at 60 km/h for 3 hours. How far does it travel?"
            ),
            "solution": (
                "Distance = speed * time\n"
                " = 60 * 3\n"
                " = 180 km"
            ),
            "answer": "180",
        },
        {
            "question": (
                "A store buys products for $25 each and sells them for $40 each.\n"
                "How much total profit is made on 10 products?"
            ),
            "solution": (
                "Profit per product = selling price - cost price\n"
                " = 40 - 25 = $15\n"
                "Total profit = profit per product * number of products\n"
                " = 15 * 10 = $150"
            ),
            "answer": "150",
        },
        {
            "question": (
                "A tank holds 500 litres. It is currently 40 percent full.\n"
                "How many litres are in the tank?"
            ),
            "solution": (
                "Litres in tank = total capacity * percentage full\n"
                " = 500 * 0.40\n"
                " = 200 litres"
            ),
            "answer": "200",
        },
        {
            "question": (
                "A worker earns $15 per hour and works 8 hours a day for 5 days.\n"
                "What are the total earnings?"
            ),
            "solution": (
                "Daily earnings = hourly rate * hours per day\n"
                " = 15 * 8 = $120\n"
                "Total earnings = daily earnings * number of days\n"
                " = 120 * 5 = $600"
            ),
            "answer": "600",
        },
        {
            "question": (
                "A recipe requires 2.5 cups of flour for 12 cookies.\n"
                "How much flour is needed for 36 cookies?"
            ),
            "solution": (
                "Scale factor = 36 / 12 = 3\n"
                "Flour needed = 2.5 * 3 = 7.5 cups"
            ),
            "answer": "7.5",
        },
    ],
}

DISTRACTOR_KEYWORDS = [
    "favorite", "favourite", "lives in", "works as", "years old",
    "founded in", "located", "made in", "size medium", "size large",
    "has been working", "generations", "covers"
]

NEGATION_KEYWORDS = [
    "not", "nor", "nothing", "no ", "none", "never", "cannot", "can't",
    "did not", "is not", "are not"
]

MULTI_HOP_KEYWORDS = [
    "greater than", "more than", "less than", "earns", "times as",
    "twice", "three times", "four times", "five times",
    "as fast as", "as many as", "as much as", "half as"
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


def targetedFewShotPrompt(question: str, benchmark: str, numExamples: int = 3, use_cot: bool = True) -> str:
    failureType = classifyQuestions(question)

    examples = TARGETED_EXAMPLES.get(failureType, TARGETED_EXAMPLES["arithmetic"])
    exampleToUse = examples[:numExamples]

    exampleBlock = ""
    for i, example in enumerate(exampleToUse, 1):
        if use_cot:
            exampleBlock += (
                f"Example {i}:\n"
                f"Problem: {example['question']}"
                f"Solution: {example['solution']}"
                f"Answer: {example['answer']}\n\n"
            )
        else:
            exampleBlock += (
                f"Example {i}:\n"
                f"Problem: {example['question']}"
                f"Answer: {example['answer']}\n\n"
            )

    if use_cot:
        failureInstruction = {
            "distractor": (
                "The following examples show how to identify RELEVANT information and ignore IRRELEVANT information in a problem. Focus only on what is necessary."
            ),
            "multi_hop": (
                "The following examples show how to solve problems that require multiple reasoning steps. Work through each step explicitly before moving to the next step."
            ),
            "negation": (
                "The following examples show how to carefully handle negations and negative conditions. Always unpack negation step by step before concluding."
            ),
            "arithmetic": (
                "The following examples show step by step arithmetic reasoning. Show every calculation step."
            )
        }
    else:
        failureInstruction = {
            "distractor": (
                "The following examples show problems with irrelevant information and their final answers. Focus only on what is necessary to find the answer."
            ),
            "multi_hop": (
                "The following examples show problems that require multiple reasoning steps and their final answers."
            ),
            "negation": (
                "The following examples show problems involving negations and their final answers."
            ),
            "arithmetic": (
                "The following examples show arithmetic problems and their final answers."
            )
        }

    instruction = failureInstruction.get(failureType, failureInstruction["arithmetic"])

    suffix = "Write your final answer as 'Answer: [number or value]'"

    prompt = f"""{instruction}

    {exampleBlock}Now apply the same approach to solve this problem:

    Problem:
    {question}

    Solution:
    ({suffix})
    """
    return prompt, failureType


