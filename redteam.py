import dspy
import instructor
import json
import openai
import os
from dotenv import load_dotenv
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPRO
from openai import OpenAI
from tqdm import tqdm
from program import AttackProgram
from utils import get_response, judge_prompt

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
instructor_client = instructor.from_openai(OpenAI())

attack_model = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=512)
dspy.settings.configure(lm=attack_model)

target_client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)
target_model_name = "lmsys/vicuna-7b-v1.5"


def metric(intent, attack_prompt, trace=None, eval_round=True):
    response = get_response(
        target_client,
        target_model_name,
        attack_prompt,
        inference_params={"max_tokens": 512, "temperature": 0},
    )
    score = judge_prompt(instructor_client, intent, response)[0]
    if eval_round:
        score = round(score)
    return score


def eval_program(prog, eval_set):
    evaluate = Evaluate(
        devset=eval_set,
        metric=lambda x, y: metric(x, y),
        num_threads=4,
        display_progress=True,
        display_table=0,
    )
    evaluate(prog)


def main():
    with open("advbench_subset.json", "r") as f:
        goals = json.load(f)["goals"]

    trainset = [
        dspy.Example(harmful_intent=goal).with_inputs("harmful_intent")
        for goal in goals
    ]

    # Evaluate baseline: directly passing in harmful intent strings
    base_score = 0
    for ex in tqdm(trainset, desc="Raw Input Score"):
        base_score += metric(
            intent=ex.harmful_intent, attack_prompt=ex.harmful_intent, eval_round=True
        )
    base_score /= len(trainset)
    print(f"--- Raw Harmful Intent Strings ---")
    print(f"Baseline Score: {base_score}")

    attacker_prog = AttackProgram(target_client, target_model_name, layers=5)
    print(f"--- Evaluating Initial Architecture ---")
    eval_program(attacker_prog, trainset)

    optimizer = MIPRO(metric=metric, verbose=True, view_data_batch_size=3)
    best_prog = optimizer.compile(
        attacker_prog,
        trainset=trainset,
        max_bootstrapped_demos=2,
        max_labeled_demos=0,
        num_trials=30,
        requires_permission_to_run=False,
        eval_kwargs=dict(num_threads=16, display_progress=True, display_table=0),
    )

    print(f"--- Evaluating Optimized Architecture ---")
    eval_program(best_prog, trainset)


if __name__ == "__main__":
    main()
