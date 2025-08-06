import json
import os
import random

import pandas as pd


def instantiate_response(actions: list[tuple[str, str]]):
    assert len(actions) == 3
    lat1, lon1 = actions[0]
    lat2, lon2 = actions[1]
    lat3, lon3 = actions[2]
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
    Extract available longitudinal and lateral action lists from system prompt.

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
        system_prompt = row.system_prompt
        if not qwen:  # remove '/no_think' from system prompt
            system_prompt = system_prompt.rsplit('\n', 1)[0]
        user_prompt = row.user_prompt

        available_longitudinal_actions, available_lateral_actions = (
            extract_available_actions(system_prompt)
        )

        # remove "stop" as it is not in any label distribution
        available_longitudinal_actions.remove("stop")
        lateral_label = row.Trajectory_Lateral
        longitudinal_label = row.Trajectory_Longitudinal
        actions = [(lateral_label, longitudinal_label)] + pick_remaining_actions(
            lateral_label, available_lateral_actions, available_longitudinal_actions
        )
        response = instantiate_response(actions)
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


def split_fine_tuning_samples(sample_path: str, train_size: int = 2000):
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

    train_samples = [
        sample for i, sample in enumerate(all_samples) if i not in val_indices
    ]

    train_path = "finetuning_files/train.jsonl"
    val_path = "finetuning_files/val.jsonl"
    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)


if __name__ == "__main__":
    generate_conversations("finetuning_files/labels.csv", "finetuning_files/conversations.jsonl")
    split_fine_tuning_samples("finetuning_files/conversations.jsonl")
