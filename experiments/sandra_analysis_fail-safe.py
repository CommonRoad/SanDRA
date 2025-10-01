import itertools
from pathlib import Path
import pandas as pd

# === CONFIGURATION ===
model_name = "gpt-4o"
scenario_settings = [(5, 3.0)]
seeds = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 4213, 2572]
base_dir = Path("/home/liny/Documents/commonroad/results-sandra/new-results-updated")

# Cartesian product of configs
configs = itertools.product(
    [True],   # use_rules_in_prompt
    [False],  # use_rules_in_reach
    ["set-based"],  # prediction type ["set-based"]
    scenario_settings,
    seeds,
)

# Aggregated counters
total_verified3 = 0
total_steps = 0

# === PROCESS CONFIGS ===
for use_rules_in_prompt, use_rules_in_reach, prediction_type, (lanes_count, vehicle_density), seed in configs:
    # Determine expected folder pattern
    if prediction_type == "set-based":
        expected_pattern = (
            f"results-True-{model_name}-{lanes_count}-{vehicle_density}-{seed}"
            f"-spot-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"
        )
    else:
        expected_pattern = (
            f"results-True-{model_name}-{lanes_count}-{vehicle_density}-{seed}"
            f"-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"
        )

    seed_dir = base_dir / str(seed)
    if not seed_dir.exists():
        continue

    for run_folder in seed_dir.iterdir():
        print(run_folder)
        if not run_folder.is_dir() or not run_folder.name.startswith("run-"):
            continue

        result_folder = run_folder / expected_pattern
        if not result_folder.exists():
            continue

        csv_file = result_folder / "evaluation.csv"
        if not csv_file.exists():
            continue

        try:
            df = pd.read_csv(csv_file)
            if "verified-id" not in df.columns:
                continue

            verified_values = df["verified-id"].dropna().astype(int)
            total_verified3 += (verified_values == 3).sum()
            total_steps += len(verified_values)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

# === COMPUTE AND PRINT RATIO ===
if total_steps > 0:
    avg_verified3_ratio = total_verified3 / total_steps
    print(f"Average ratio of verified-id == 3 across all steps: {avg_verified3_ratio:.4f}")
else:
    print("No steps found to compute verified-id ratio.")
