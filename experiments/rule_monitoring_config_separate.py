import os
import numpy as np
import itertools
import re
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator

# Configuration settings
scenario_settings = [
    (4, 2.0),  # setting 1
    (4, 3.0),  # setting 2
    (5, 3.0),  # setting 3
]

seeds = [
    5838, 2421, 7294, 9650, 4176,
    6382, 8765, 1348, 4213, 2572
]

# Cartesian product of configs
configs = itertools.product(
    [True, False],  # use_rules_in_prompt
    [True, False],  # use_rules_in_reach
    ["set-based", "most-likely"],  # prediction type
    scenario_settings
)

# Choose one specific configuration to iterate over seeds
selected_config = (True, True, "set-based", (4, 2.0))
use_rules_in_prompt, use_rules_in_reach, prediction_type, (lanes_count, vehicles_density) = selected_config

# Set use_sonia based on prediction type
use_sonia = (prediction_type == "set-based")

# Model name (adjust as needed)
model_name = "gpt-4o"

SKIP_FIRST_STEP = False
rules = ["R_G1", "R_G2", "R_G3"]

# Initialize stats for each run
run_stats = {}  # {run_id: {rule: stats}}

# Base directory where seed folders are located
base_dir = "/home/liny/Documents/sandra_rule_results/new-results (1)/new-results"  # Change this to your actual base directory

# Iterate over all seeds for the selected configuration
for seed in seeds:
    seed_dir = os.path.join(base_dir, str(seed))

    if not os.path.exists(seed_dir):
        print(f"⚠️ Seed directory not found: {seed_dir}")
        continue

    # Find all run directories (run-1, run-2, etc.)
    run_dirs = []
    for item in os.listdir(seed_dir):
        if item.startswith("run-") and os.path.isdir(os.path.join(seed_dir, item)):
            run_dirs.append(os.path.join(seed_dir, item))

    if not run_dirs:
        print(f"⚠️ No run directories found for seed {seed}")
        continue

    # Process each run directory
    for run_dir in run_dirs:
        # Extract run ID from directory name
        run_match = re.search(r'run-(\d+)', run_dir)
        run_id = f"seed_{seed}_run_{run_match.group(1)}" if run_match else f"seed_{seed}_run_unknown"

        if run_id not in run_stats:
            run_stats[run_id] = {
                rule: {"total": 0, "violated": 0, "steps_total": 0, "steps_violated": 0}
                for rule in rules
            }

        # Find the results folder inside the run directory
        results_folders = []
        for item in os.listdir(run_dir):
            # Build the expected pattern for the results folder
            if use_sonia:
                expected_pattern = f"results-True-{model_name}-{lanes_count}-{vehicles_density}-{seed}-spot-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"
            else:
                expected_pattern = f"results-True-{model_name}-{lanes_count}-{vehicles_density}-{seed}-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"
            # results-True-gpt-4o-4-2.0-1348-rule_prompt-False-reach-False
            # Check if the folder starts with the expected pattern
            if item.startswith(expected_pattern) and os.path.isdir(os.path.join(run_dir, item)):
                results_folders.append(os.path.join(run_dir, item))

        if not results_folders:
            print(f"⚠️ No results folder found in {run_dir} with pattern starting with: {expected_pattern}")
            continue

        # Process each results folder
        for results_folder in results_folders:
            print(f"\n=== Processing: {results_folder} (seed: {seed}, run: {run_id}) ===")

            for file_name in os.listdir(results_folder):
                if not file_name.endswith(".xml"):
                    continue

                scenario_path = os.path.join(results_folder, file_name)
                print(f"Processing XML: {scenario_path}")

                try:
                    # Open scenario with lanelet assignment
                    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
                    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

                    # Create world
                    world = World.create_from_scenario(scenario)

                    # Find ego vehicle
                    ego_vehicle = None
                    for vehicle in world.vehicles:
                        if np.array_equal(vehicle.state_list_cr[0].position, planning_problem.initial_state.position):
                            ego_vehicle = vehicle
                            break

                    if ego_vehicle is None:
                        print(f"⚠️ No matching ego vehicle found in {file_name}, skipping...")
                        continue

                    # Evaluate all rules for this ego
                    for rule in rules:
                        rule_evaluator = RuleEvaluator.create_from_config(world, ego_vehicle, rule)
                        robustness_array = rule_evaluator.evaluate()[:-1]

                        run_stats[run_id][rule]["total"] += 1

                        if robustness_array.shape[0] > 0:
                            run_stats[run_id][rule]["steps_total"] += robustness_array.shape[0]

                            if SKIP_FIRST_STEP:
                                neg_steps = np.sum(robustness_array[1:] < 0)
                                if robustness_array[0] < 0:
                                    # ignore first step if negative
                                    pass
                            else:
                                neg_steps = np.sum(robustness_array < 0)

                            run_stats[run_id][rule]["steps_violated"] += neg_steps

                            if neg_steps > 0:
                                run_stats[run_id][rule]["violated"] += 1

                except Exception as e:
                    print(f"❌ Error processing {file_name}: {e}")

# Calculate averages across all runs
avg_stats = {
    rule: {"total": 0, "violated": 0, "steps_total": 0, "steps_violated": 0,
           "scenario_ratio": 0.0, "step_ratio": 0.0}
    for rule in rules
}

# Sum across all runs
for run_id in run_stats:
    for rule in rules:
        for stat in ["total", "violated", "steps_total", "steps_violated"]:
            avg_stats[rule][stat] += run_stats[run_id][rule][stat]

# Calculate averages
num_runs = len(run_stats)
if num_runs > 0:
    for rule in rules:
        if avg_stats[rule]["total"] > 0:
            avg_stats[rule]["scenario_ratio"] = avg_stats[rule]["violated"] / avg_stats[rule]["total"]
        if avg_stats[rule]["steps_total"] > 0:
            avg_stats[rule]["step_ratio"] = avg_stats[rule]["steps_violated"] / avg_stats[rule]["steps_total"]

# Print results per run
print(f"\n{'=' * 80}")
print(f"Detailed results per run for config: {selected_config}")
print(f"{'=' * 80}")

for run_id in sorted(run_stats.keys()):
    print(f"\n--- {run_id} ---")
    for rule in rules:
        stats = run_stats[run_id][rule]
        if stats["total"] > 0:
            scenario_ratio = stats["violated"] / stats["total"]
            print(f"{rule}: total={stats['total']}, violated={stats['violated']}, violation ratio={scenario_ratio:.2f}")

            if stats["steps_total"] > 0:
                step_ratio = stats["steps_violated"] / stats["steps_total"]
                print(
                    f"   time steps: total={stats['steps_total']}, violated={stats['steps_violated']}, violation ratio={step_ratio:.2f}")

# Print average results
print(f"\n{'=' * 80}")
print(f"Average results across {num_runs} runs for config: {selected_config}")
print(f"{'=' * 80}")

for rule in rules:
    if avg_stats[rule]["total"] > 0:
        print(f"{rule}:")
        print(f"  Scenarios: total={avg_stats[rule]['total']}, violated={avg_stats[rule]['violated']}, "
              f"violation ratio={avg_stats[rule]['scenario_ratio']:.2f}")

        if avg_stats[rule]["steps_total"] > 0:
            print(
                f"  Time steps: total={avg_stats[rule]['steps_total']}, violated={avg_stats[rule]['steps_violated']}, "
                f"violation ratio={avg_stats[rule]['step_ratio']:.2f}")
    else:
        print(f"{rule}: no cases evaluated.")

# Print standard deviation if you have multiple runs
if num_runs > 1:
    print(f"\n{'=' * 80}")
    print(f"Standard deviation across {num_runs} runs")
    print(f"{'=' * 80}")

    # Calculate standard deviation for scenario violation ratios
    scenario_ratios = {}
    step_ratios = {}

    for rule in rules:
        scenario_ratios[rule] = []
        step_ratios[rule] = []

    for run_id in run_stats:
        for rule in rules:
            stats = run_stats[run_id][rule]
            if stats["total"] > 0:
                scenario_ratios[rule].append(stats["violated"] / stats["total"])
            if stats["steps_total"] > 0:
                step_ratios[rule].append(stats["steps_violated"] / stats["steps_total"])

    for rule in rules:
        if len(scenario_ratios[rule]) > 1:
            std_dev = np.std(scenario_ratios[rule])
            print(f"{rule} scenario ratio std dev: {std_dev:.3f}")
        if len(step_ratios[rule]) > 1:
            std_dev = np.std(step_ratios[rule])
            print(f"{rule} step ratio std dev: {std_dev:.3f}")