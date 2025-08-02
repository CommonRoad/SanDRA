import json
import os
import random

import pandas as pd
from openai import OpenAI
from openai.types import Batch

from sandra.common.config import SanDRAConfiguration
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.utility.general import extract_scenario_and_planning_problem


def instantiate_cot_output():
    example_output_cot = f"""{{
  "thoughts": {{
    "observation": [
      "The car is currently driving on the highway at 25.0 m/s with no immediate acceleration or steering.",
      "Three other cars (6, 9, 12) are on the same lane, with decreasing time-to-collision values, indicating potential risks ahead.",
      "Car 13 is on the left-adjacent lane, moving slightly slower but with infinite time-to-collision, suggesting no immediate threat."
    ],
    "conclusion": "The primary risk comes from car 6, which is closest with a 4.3 seconds time-to-collision. Car 9 and 12 are further back but still on the same lane. Car 13 is on the left lane but not an immediate threat. The driver needs to consider lane changes or adjustments to avoid potential collisions with car 6."
  }},
  "best_combination": {{
    "lateral_action": "left",
    "longitudinal_action": "decelerate"
  }}
  ,
  "second_best_combination": {{
    "lateral_action": "left",
    "longitudinal_action": "keep"
  }}
  ,
  "third_best_combination": {{
    "lateral_action": "left",
    "longitudinal_action": "accelerate"
  }}
}}"""


def instantiate_normal_output(lateral_actions: list[tuple[str, str]]):
    assert len(lateral_actions) == 3
    lat1, lon1 = lateral_actions[0]
    lat2, lon2 = lateral_actions[1]
    lat3, lon3 = lateral_actions[2]
    return f"""{{
  "best_combination": {{
    "lateral_action": "{lat1}",
    "longitudinal_action": "{lon1}"
  }}
  ,
  "second_best_combination": {{
    "lateral_action": "{lat2}",
    "longitudinal_action": "{lon2}"
  }}
  ,
  "third_best_combination": {{
    "lateral_action": "{lat3}",
    "longitudinal_action": "{lon3}"
  }}
}}"""


def extract_available_actions(text: str) -> tuple[list[str], list[str]]:
    """
    Extract longitudinal and lateral action lists from structured text.

    Returns:
        tuple: (longitudinal_actions, lateral_actions)
    """
    lines = [line.strip() for line in text.split("\n")]

    longitudinal_actions = []
    lateral_actions = []

    current_section = None

    for line in lines:
        line = line.strip()

        if line.startswith("Feasible longitudinal actions:"):
            current_section = "longitudinal"
        elif line.startswith("Feasible lateral actions:"):
            current_section = "lateral"
        elif line.startswith("- ") and current_section:
            # Extract the action (remove "- " prefix)
            action = line[2:].strip()

            if current_section == "longitudinal":
                longitudinal_actions.append(action)
            elif current_section == "lateral":
                lateral_actions.append(action)

    return longitudinal_actions, lateral_actions


def pick_remaining_actions(
    combination: tuple[str, str],
    available_strings1: list[str],
    available_strings2: list[str],
    n: int = 2,
) -> list[tuple[str, str]]:
    """
    Randomly pick n combinations of strings where the first part is from available_strings1
    and the second is from available_strings2, excluding the already chosen combination.
    """
    all_combinations = [
        (s1, s2) for s1 in available_strings1 for s2 in available_strings2
    ]
    remaining_combinations = [
        combo for combo in all_combinations if combo != combination
    ]

    if len(remaining_combinations) <= n:
        raise ValueError("Not enough remaining combinations")
    return random.sample(remaining_combinations, n)


def generate_conversations(source_path: str, save_path: str = None, qwen=False) -> list[list[dict]]:
    """
    Extract prompts and labels, formulate a correct response, and save the resulting conversations as jsonl.
    """
    df = pd.read_csv(source_path)
    conversations = []
    for row in df.itertuples():
        # full_prompt = row.Prompt
        # split_sentence = "Here is an overview of your environment:"
        # split_prompt = full_prompt.split(split_sentence)
        system_prompt = row.system_prompt
        if not qwen:
            system_prompt = system_prompt.rsplit('\n', 1)[0]
        user_prompt = row.user_prompt

        available_longitudinal_actions, available_lateral_actions = (
            extract_available_actions(system_prompt)
        )

        # remove "stop"
        available_longitudinal_actions.remove("stop")
        lateral_label = row.Trajectory_Lateral
        longitudinal_label = row.Trajectory_Longitudinal
        actions = [(lateral_label, longitudinal_label)] + pick_remaining_actions(
            lateral_label, available_lateral_actions, available_longitudinal_actions
        )
        response = instantiate_normal_output(actions)
        item = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]
        conversations.append({"messages": item})
    if save_path:
        with open(save_path, "w") as f:
            json.dump(conversations, f)

    return conversations


def load_jsonl(filepath):
    """Load data from JSONL format."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, filepath):
    """Save data as JSONL format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_batch(batch_file_name: str, api_key_name="OPENAI_API_KEY") -> Batch:
    client = OpenAI(api_key=os.getenv(api_key_name))
    path_to = "batch_files"
    batch_file = client.files.create(
        file=open(os.path.join(path_to, batch_file_name), "rb"), purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"BATCH ID: {batch.id}")
    # client.batches.retrieve(batch.id)
    return batch


def get_batch_status(batch_id: str, api_key_name="OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv(api_key_name))
    batch = client.batches.retrieve(batch_id)

    print(f"STATUS: {batch.status}")
    if batch.status == "failed":
        print(str(batch.errors))
        # client.files.retrieve(batch.error_file_id)


def download_batch_result(
    batch_id: str, api_key_name="OPENAI_API_KEY", save_name="batch_result.jsonl"
):
    client = OpenAI(api_key=os.getenv(api_key_name))
    batch = client.batches.retrieve(batch_id)
    result_file = client.files.content(batch.output_file_id)
    content = result_file.read().decode("utf-8")
    path_to = "batch_results"
    save_path = os.path.join(path_to, save_name)
    with open(save_path, "w") as f:
        f.write(content)

    print(f"Batch result saved to {save_name}")


def create_finetuning_job(
    training_file_name: str,
    validation_file_name: str,
    api_key_name="OPENAI_API_KEY",
    model="gpt-4o",
):
    client = OpenAI(api_key=os.getenv(api_key_name))
    path_to = "finetuning_files"
    training_file = client.files.create(
        file=open(os.path.join(path_to, training_file_name), "rb"), purpose="fine-tune"
    )

    validation_file = client.files.create(
        file=open(os.path.join(path_to, validation_file_name), "rb"),
        purpose="fine-tune",
    )
    print(training_file.id)
    print(validation_file.id)
    fine_tuning_job = client.fine_tuning.jobs.create(
        training_file=training_file.id,
        validation_file=validation_file.id,  # This is where you specify the validation set
        model=model,  # Or your base model of choice
    )
    print(fine_tuning_job.id)


def split_fine_tuning_samples(sample_path: str, train_size: int = 2000):
    save_folder = "finetuning_files"
    samples = load_jsonl(sample_path)[0]
    val_size = int(train_size * 0.1)

    all_samples = random.sample(samples, train_size + val_size)
    val_samples = random.sample(all_samples, val_size)
    val_indices = set()
    for val_sample in val_samples:
        for i, sample in enumerate(all_samples):
            if sample == val_sample:
                val_indices.add(i)
                break

    # Create train samples by excluding validation samples
    train_samples = [
        sample for i, sample in enumerate(all_samples) if i not in val_indices
    ]

    train_path = os.path.join(save_folder, "train.jsonl")
    val_path = os.path.join(save_folder, "val.jsonl")
    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)


if __name__ == "__main__":

    def simple_jsonl_to_list(input_file: str, output_file: str):
        """
        Simple version: Read JSONL file to list and save as JSON array.

        Args:
            input_file (str): Path to input JSONL file
            output_file (str): Path to output JSON file

        Returns:
            List[Any]: List of parsed JSON objects
        """
        data = []

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        # data = data[0]
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        return data

    simple_jsonl_to_list("finetuning_files/train-new-gpt2.jsonl", "finetuning_files/train-new-gpt3.jsonl")
    # generate_conversations(
    #     "validation_with_prompts.csv",
    #     "finetuning_files/val-new-gpt.jsonl",
    # )
    # df = pd.read_csv("validation.csv")
#
    # # Initialize lists to store the prompts
    # system_prompts = []
    # user_prompts = []
#
    # for row in df.itertuples():
    #     scenario_id = row.ScenarioID
    #     scenario_path = f"/home/sebastian/Documents/Uni/Sandra/highD-sandra-0.04/{scenario_id}.xml"
    #     scenario, planning_problem = extract_scenario_and_planning_problem(
    #         scenario_path
    #     )
    #     describer = CommonRoadDescriber(
    #         scenario,
    #         planning_problem,
    #         0,
    #         SanDRAConfiguration()
    #     )
    #     system_prompt = describer.system_prompt()
    #     user_prompt = describer.user_prompt()
#
    #     # Append prompts to lists
    #     system_prompts.append(system_prompt)
    #     user_prompts.append(user_prompt)
#
    # # Add new columns to dataframe
    # df['system_prompt'] = system_prompts
    # df['user_prompt'] = user_prompts
#
    # # Save the updated dataframe
    # df.to_csv("validation_with_prompts.csv", index=False)


    # generate_conversations("conversations-new.jsonl")
    # split_fine_tuning_samples("conversations-new.jsonl")
