import random

EXEMPLAR_BANK = {

    "confidently_wrong": [
        {
            "question": (
                "A dolphin is 120 feet long. 4 remoras are attached to it, each 18 inches long.\n"
                "What percentage of the dolphin's body length is the combined length of the remoras?"
            ),
            "error_demonstration": (
                "This is a confidently_wrong error : the model skips reasoning and states a wrong answer as if it is certain.\n"
                "INCORRECT:\n"
                "Working through this problem step by step and carefully, the answer is 20\n"
                "Answer: 20"
            ),
            "solution": (
                "Step 1: Convert dolphin length to inches: 120 feet * 12 = 1440 inches.\n"
                "Step 2: Combined remora length: 4 * 18 = 72 inches.\n"
                "Step 3: Percentage: (72 / 1440) * 100 = 5percent.\n"
                "Answer: 5"
            ),
            "answer": "5",
        },
        {
            "question": (
                "A store sells notebooks for $3 each and pens for $1.50 each.\n"
                "Lisa buys 4 notebooks and 6 pens. How much does she spend in total?"
            ),
            "error_demonstration": (
                "This is a confidently_wrong error : the model states a wrong total without showing any calculation.\n"
                "INCORRECT:\n"
                "The total cost is $18.\n"
                "Answer: 18"
            ),
            "solution": (
                "Step 1: Cost of notebooks: 4 * $3 = $12.\n"
                "Step 2: Cost of pens: 6 * $1.50 = $9.\n"
                "Step 3: Total: $12 + $9 = $21.\n"
                "Answer: 21"
            ),
            "answer": "21",
        },
    ],

    "drifting": [
        {
            "question": (
                "A shark is 90 feet long. 4 remoras are attached to it, each 54 inches long.\n"
                "What percentage of the shark's body length is the combined length of the remoras?"
            ),
            "error_demonstration": (
                "This is a drifting error : the model starts reasoning correctly but makes a wrong calculation midway and carries the mistake forward.\n"
                "INCORRECT:\n"
                "Step 1: Convert shark length to inches: 90 * 12 = 1080 inches. (correct)\n"
                "Step 2: Combined remora length: 4 * 54 = 216 inches. (correct)\n"
                "Step 3: Percentage: (216 / 1080) * 100 = 27percent. (wrong : divided incorrectly)\n"
                "Answer: 27"
            ),
            "solution": (
                "Step 1: Convert shark length to inches: 90 * 12 = 1080 inches.\n"
                "Step 2: Combined remora length: 4 * 54 = 216 inches.\n"
                "Step 3: Percentage: (216 / 1080) * 100 = 20percent.\n"
                "Answer: 20"
            ),
            "answer": "20",
        },
        {
            "question": (
                "Alice earns $800 per week. She saves 25percent of her earnings each week.\n"
                "How much does she save in 6 weeks?"
            ),
            "error_demonstration": (
                "This is a drifting error : the model correctly finds the weekly saving but then multiplies incorrectly.\n"
                "INCORRECT:\n"
                "Step 1: Weekly saving: 25percent of $800 = $200. (correct)\n"
                "Step 2: Savings over 6 weeks: $200 * 6 = $1400. (wrong arithmetic)\n"
                "Answer: 1400"
            ),
            "solution": (
                "Step 1: Weekly saving: 25percent of $800 = $200.\n"
                "Step 2: Savings over 6 weeks: $200 * 6 = $1200.\n"
                "Answer: 1200"
            ),
            "answer": "1200",
        },
    ],

    "distractor": [
        {
            "question": (
                "John, who is 34 years old and lives in Melbourne, has 50 marbles.\n"
                "He gives 15 marbles to his friend Sam, who works as a teacher.\n"
                "How many marbles does John have left?"
            ),
            "error_demonstration": (
                "This is a distractor error : the model uses irrelevant numbers from the question instead of the relevant ones.\n"
                "INCORRECT:\n"
                "John is 34 years old. He gave away 15 marbles.\n"
                "Marbles left: 34 - 15 = 19.\n"
                "Answer: 19"
            ),
            "solution": (
                "I will identify relevant information and ignore irrelevant details.\n"
                "Relevant: John has 50 marbles and gives 15 to Sam.\n"
                "Irrelevant (ignored): John's age, city, Sam's profession.\n"
                "Calculation: 50 - 15 = 35.\n"
                "Answer: 35"
            ),
            "answer": "35",
        },
        {
            "question": (
                "A shop founded in 2005 stocks 200 items. It sells 45 items on Monday and 30 items on Tuesday.\n"
                "The shop has 3 floors and is open 7 days a week.\n"
                "How many items remain after Monday and Tuesday?"
            ),
            "error_demonstration": (
                "This is a distractor error : the model incorporates the number of floors or days into its calculation.\n"
                "INCORRECT:\n"
                "Items sold: 45 + 30 = 75. Items on 3 floors: 200 / 3 = 66 per floor.\n"
                "Remaining: 66 * 3 - 75 = 123. (wrong : used floor count)\n"
                "Answer: 123"
            ),
            "solution": (
                "Relevant: 200 items, 45 sold Monday, 30 sold Tuesday.\n"
                "Irrelevant (ignored): founding year, number of floors, days open per week.\n"
                "Calculation: 200 - 45 - 30 = 125.\n"
                "Answer: 125"
            ),
            "answer": "125",
        },
    ],

    "negation": [
        {
            "question": (
                "No reptiles are warm-blooded. Snakes are reptiles.\n"
                "Are snakes warm-blooded?"
            ),
            "error_demonstration": (
                "This is a negation error : the model ignores the negation 'No' and treats the statement as a positive rule.\n"
                "INCORRECT:\n"
                "Reptiles are warm-blooded. Snakes are reptiles.\n"
                "Therefore snakes are warm-blooded.\n"
                "Answer: True"
            ),
            "solution": (
                "P1: No reptiles are warm-blooded. (Universal negation : all reptiles are excluded from warm-blooded.)\n"
                "P2: Snakes are reptiles.\n"
                "Therefore: Snakes are not warm-blooded.\n"
                "Answer: False"
            ),
            "answer": "False",
        },
        {
            "question": (
                "None of the students who passed the exam failed the course.\n"
                "Maria passed the exam. Did Maria fail the course?"
            ),
            "error_demonstration": (
                "This is a negation error : the model reverses the negation and concludes the opposite of what the rule states.\n"
                "INCORRECT:\n"
                "Students who passed the exam failed the course. Maria passed the exam.\n"
                "Therefore Maria failed the course.\n"
                "Answer: Yes"
            ),
            "solution": (
                "Step 1: None of the students who passed the exam failed the course.\n"
                "        This means: passed exam → did NOT fail course.\n"
                "Step 2: Maria passed the exam.\n"
                "Step 3: Therefore Maria did not fail the course.\n"
                "Answer: No"
            ),
            "answer": "No",
        },
    ],

    "multi_hop": [
        {
            "question": (
                "Leon rolls a 10-sided die. How much more likely is it (as a percentage) that he rolls\n"
                "a number greater than 2 than that he rolls two even numbers in a row?"
            ),
            "error_demonstration": (
                "This is a multi_hop error : the model computes one probability correctly but skips calculating the second and jumps to the answer.\n"
                "INCORRECT:\n"
                "Numbers greater than 2: {3,4,5,6,7,8,9,10} = 8 out of 10 = 80percent.\n"
                "Difference = 80 - 25 = 55. (wrong : never calculated the two-even probability)\n"
                "Answer: 28"
            ),
            "solution": (
                "Step 1: P(greater than 2) = 8/10 = 80percent.\n"
                "Step 2: Even numbers: {2,4,6,8,10} = 5 out of 10. P(two evens in a row) = (5/10) * (5/10) = 25percent.\n"
                "Step 3: Difference = 80percent - 25percent = 55percent.\n"
                "Answer: 55"
            ),
            "answer": "55",
        },
        {
            "question": (
                "Factory A produces 3 times as many parts as Factory B.\n"
                "Factory B produces twice as many parts as Factory C.\n"
                "Factory C produces 40 parts per day. How many parts does Factory A produce per day?"
            ),
            "error_demonstration": (
                "This is a multi_hop error : the model skips Factory B and applies Factory A's multiplier directly to Factory C.\n"
                "INCORRECT:\n"
                "Factory C = 40. Factory A = 3 * 40 = 120. (skipped Factory B entirely)\n"
                "Answer: 120"
            ),
            "solution": (
                "Step 1: Factory C produces 40 parts per day.\n"
                "Step 2: Factory B produces 2 * 40 = 80 parts per day.\n"
                "Step 3: Factory A produces 3 * 80 = 240 parts per day.\n"
                "Answer: 240"
            ),
            "answer": "240",
        },
    ],

    "hallucination": [
        {
            "question": (
                "A car travels at 60 km/h for 2 hours, then at 80 km/h for 3 hours.\n"
                "What is the total distance travelled?"
            ),
            "error_demonstration": (
                "This is a hallucination error : the model introduces a rest stop or extra segment that is not mentioned in the question.\n"
                "INCORRECT:\n"
                "Segment 1: 60 * 2 = 120 km.\n"
                "Rest stop: 15 minutes = 0.25 hours at 0 km/h. (not in question)\n"
                "Segment 2: 80 * 3 = 240 km.\n"
                "Total: 120 + 240 = 360 km.\n"
                "Answer: 360"
            ),
            "solution": (
                "Step 1: Distance in first segment: 60 * 2 = 120 km.\n"
                "Step 2: Distance in second segment: 80 * 3 = 240 km.\n"
                "Step 3: Total distance: 120 + 240 = 360 km.\n"
                "Answer: 360"
            ),
            "answer": "360",
        },
        {
            "question": (
                "A bag contains 5 red balls and 3 blue balls.\n"
                "What is the probability of drawing a red ball?"
            ),
            "error_demonstration": (
                "This is a hallucination error : the model adds green balls that do not exist in the question.\n"
                "INCORRECT:\n"
                "The bag has 5 red, 3 blue, and 2 green balls. (green balls not in question)\n"
                "Total = 10. P(red) = 5/10 = 0.5.\n"
                "Answer: 0.5"
            ),
            "solution": (
                "Step 1: Total balls = 5 red + 3 blue = 8.\n"
                "Step 2: P(red) = 5/8.\n"
                "Answer: 5/8"
            ),
            "answer": "5/8",
        },
    ],

    "format_correct_wrong_answer": [
        {
            "question": (
                "A dolphin is 100 feet long. 3 remoras are attached to it, each 8 inches long.\n"
                "What percentage of the dolphin's body length is the combined length of the remoras?"
            ),
            "error_demonstration": (
                "This is a format_correct_wrong_answer error : the reasoning steps are correct but the wrong intermediate value is reported as the final answer.\n"
                "INCORRECT:\n"
                "Step 1: Convert dolphin to inches: 100 * 12 = 1200 inches.\n"
                "Step 2: Combined remora length: 3 * 8 = 24 inches.\n"
                "Step 3: Percentage: (24 / 1200) * 100 = 2percent. (correct)\n"
                "Answer: 6.7percent (wrong : reported a different number, not the final result)\n"
            ),
            "solution": (
                "Step 1: Convert dolphin to inches: 100 * 12 = 1200 inches.\n"
                "Step 2: Combined remora length: 3 * 8 = 24 inches.\n"
                "Step 3: Percentage: (24 / 1200) * 100 = 2percent.\n"
                "Answer: 2"
            ),
            "answer": "2",
        },
        {
            "question": (
                "A worker earns $18 per hour. She works 7 hours on Monday and 5 hours on Tuesday.\n"
                "How much does she earn in total over the two days?"
            ),
            "error_demonstration": (
                "This is a format_correct_wrong_answer error : the model does the right calculation but extracts an intermediate value as the final answer.\n"
                "INCORRECT:\n"
                "Monday: 18 * 7 = $126.\n"
                "Tuesday: 18 * 5 = $90.\n"
                "Total: $126 + $90 = $216.\n"
                "Answer: 126 (wrong : stated Monday's earnings instead of the total)\n"
            ),
            "solution": (
                "Step 1: Monday earnings: 18 * 7 = $126.\n"
                "Step 2: Tuesday earnings: 18 * 5 = $90.\n"
                "Step 3: Total: $126 + $90 = $216.\n"
                "Answer: 216"
            ),
            "answer": "216",
        },
    ],

    "off_topic": [
        {
            "question": (
                "A train leaves Station A at 9:00 AM travelling at 90 km/h.\n"
                "Station B is 270 km away. What time does the train arrive at Station B?"
            ),
            "error_demonstration": (
                "This is an off_topic error : the model gives a general explanation about trains instead of solving the problem.\n"
                "INCORRECT:\n"
                "Trains are an important mode of transport used worldwide. They connect cities and reduce road congestion.\n"
                "I cannot determine the exact arrival time without more information.\n"
                "Answer: Unknown"
            ),
            "solution": (
                "Step 1: Time = distance / speed = 270 / 90 = 3 hours.\n"
                "Step 2: Departure 9:00 AM + 3 hours = 12:00 PM.\n"
                "Answer: 12:00 PM"
            ),
            "answer": "12:00 PM",
        },
    ],

    "unknown": [
        {
            "question": (
                "A rectangle has a perimeter of 36 cm. Its length is twice its width.\n"
                "What is the area of the rectangle?"
            ),
            "error_demonstration": (
                "This is an unknown error : the reasoning does not match any clear error pattern but the answer is still wrong.\n"
                "INCORRECT:\n"
                "Perimeter = 36. Length = 2 * width. Area = length * width = 2 * 36 = 72.\n"
                "Answer: 72"
            ),
            "solution": (
                "Step 1: Let width = w, length = 2w.\n"
                "Step 2: Perimeter = 2 * (w + 2w) = 6w = 36, so w = 6 cm.\n"
                "Step 3: Length = 2 * 6 = 12 cm.\n"
                "Step 4: Area = 12 * 6 = 72 cm squared.\n"
                "Answer: 72"
            ),
            "answer": "72",
        },
    ],
}


def classifyErrorType(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["times as", "twice", "three times", "four times", "five times", "as fast as", "as many as", "half as"]):
        return "multi_hop"
    if any(k in q for k in ["favourite", "favorite", "years old", "works as", "lives in", "founded in", "located", "made in"]):
        return "distractor"
    if any(k in q for k in ["no ", "none", "nothing", "not ", "cannot", "never"]):
        return "negation"
    return "drifting"


def errorTargetedICLPrompt(
        question: str,
        benchmark: str,
        numExamples: int = 3,
        showError: bool = True,
        randomClass: bool = False
) -> str:
    errorClass = classifyErrorType(question)

    if randomClass:
        otherClasses = [c for c in EXEMPLAR_BANK if c != errorClass]
        errorClass = random.choice(otherClasses)

    examples = EXEMPLAR_BANK.get(errorClass, EXEMPLAR_BANK["confidently_wrong"])
    exampleToUse = examples[:numExamples]

    if showError:
        instruction = (
            f"The following examples show a common '{errorClass}' error that models make, followed by the correct reasoning. Learn from the mistake shown and avoid repeating it."
        )
    else:
        instruction = (
            f"The following examples demonstrate how to correctly solve problems where models typically make a '{errorClass}' error. Follow the same reasoning approach."
        )

    exampleBlock = ""
    for i, example in enumerate(exampleToUse, 1):
        if showError:
            exampleBlock += (
                f"Example {i}:\n"
                f"Problem: {example['question']}\n"
                f"{example['error_demonstration']}\n\n"
                f"CORRECT:\n"
                f"{example['solution']}\n"
                f"Answer: {example['answer']}\n\n"
            )
        else:
            exampleBlock += (
                f"Example {i}:\n"
                f"Problem: {example['question']}\n"
                f"Solution:\n"
                f"{example['solution']}\n"
                f"Answer: {example['answer']}\n\n"
            )

    suffix = "Write your final answer as 'Answer: [number or value]'"

    prompt = f"""{instruction}

    {exampleBlock}Now apply the same approach to solve this problem:

    Problem:
    {question}

    Solution:
    ({suffix})
    """
    return prompt
