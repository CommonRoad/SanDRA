import os
import numpy as np
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

# All combinations to test
config_combinations = [
    (True, True, "set-based"),
    (False, True, "set-based"),
    # (True, False, "set-based"),
    # (False, False, "set-based"),
    # (True, True, "most-likely"),
    # (False, True, "most-likely"),
    # (True, False, "most-likely"),
    # (False, False, "most-likely")
]

# Model name (adjust as needed)
model_name = "gpt-4o"

SKIP_FIRST_STEP = False
rules = ["R_G1", "R_G2", "R_G3"]

# Base directory where seed folders are located
base_dir = "/home/liny/Documents/commonroad/results-sandra/new-results-updated"

# Dictionary to store all results
all_results = {}

# Iterate through all configuration combinations
for selected_config in config_combinations:
    use_rules_in_prompt, use_rules_in_reach, prediction_type = selected_config

    # Initialize stats for this configuration
    config_stats = {
        rule: {
            "total_scenarios": 0,
            "violated_scenarios": 0,
            "total_steps": 0,
            "violated_steps": 0,
            "scenario_ratio": 0.0,
            "step_ratio": 0.0
        }
        for rule in rules
    }

    # Initialize stats for each scenario setting
    scenario_stats = {
        scenario_setting: {
            rule: {
                "total_scenarios": 0,
                "violated_scenarios": 0,
                "total_steps": 0,
                "violated_steps": 0,
                "scenario_ratio": 0.0,
                "step_ratio": 0.0
            }
            for rule in rules
        }
        for scenario_setting in scenario_settings
    }

    # Track all individual results for calculating standard deviation
    scenario_ratios = {rule: [] for rule in rules}
    step_ratios = {rule: [] for rule in rules}

    print(f"\n{'=' * 80}")
    print(f"Processing configuration: {selected_config}")
    print(f"{'=' * 80}")

    # Iterate over all scenario settings
    for scenario_setting in scenario_settings:
        lanes_count, vehicles_density = scenario_setting

        # Set use_sonia based on prediction type
        use_sonia = (prediction_type == "set-based")

        print(f"\nProcessing scenario setting: ({lanes_count}, {vehicles_density})")

        # Iterate over all seeds for the current scenario setting
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
                run_id = run_match.group(1) if run_match else "unknown"

                # Find the results folder inside the run directory
                results_folders = []
                for item in os.listdir(run_dir):
                    # Build the expected pattern for the results folder
                    if use_sonia:
                        expected_pattern = f"results-True-{model_name}-{lanes_count}-{vehicles_density}-{seed}-spot-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"
                    else:
                        expected_pattern = f"results-True-{model_name}-{lanes_count}-{vehicles_density}-{seed}-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"

                    # Check if the folder starts with the expected pattern
                    if item.startswith(expected_pattern) and os.path.isdir(os.path.join(run_dir, item)):
                        results_folders.append(os.path.join(run_dir, item))

                if not results_folders:
                    print(f"⚠️ No results folder found in {run_dir} with pattern starting with: {expected_pattern}")
                    continue

                # Initialize stats for this specific run
                run_stats = {
                    rule: {"total": 0, "violated": 0, "steps_total": 0, "steps_violated": 0}
                    for rule in rules
                }

                # Process each results folder
                for results_folder in results_folders:
                    print(f"Processing: {results_folder} (seed: {seed}, run: {run_id})")

                    for file_name in os.listdir(results_folder):
                        if not file_name.endswith(".xml"):
                            continue

                        scenario_path = os.path.join(results_folder, file_name)

                        try:
                            # Open scenario with lanelet assignment
                            scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open(
                                lanelet_assignment=True)
                            planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

                            # Create world
                            world = World.create_from_scenario(scenario)

                            # Find ego vehicle
                            ego_vehicle = None
                            for vehicle in world.vehicles:
                                if np.array_equal(vehicle.state_list_cr[0].position,
                                                  planning_problem.initial_state.position):
                                    ego_vehicle = vehicle
                                    break

                            if ego_vehicle is None:
                                print(f"⚠️ No matching ego vehicle found in {file_name}, skipping...")
                                continue

                            # Evaluate all rules for this ego
                            for rule in rules:
                                rule_evaluator = RuleEvaluator.create_from_config(world, ego_vehicle, rule)
                                robustness_array = rule_evaluator.evaluate()[:-1]

                                run_stats[rule]["total"] += 1
                                config_stats[rule]["total_scenarios"] += 1
                                scenario_stats[scenario_setting][rule]["total_scenarios"] += 1

                                if robustness_array.shape[0] > 0:
                                    run_stats[rule]["steps_total"] += robustness_array.shape[0]
                                    config_stats[rule]["total_steps"] += robustness_array.shape[0]
                                    scenario_stats[scenario_setting][rule]["total_steps"] += robustness_array.shape[0]

                                    if SKIP_FIRST_STEP:
                                        neg_steps = np.sum(robustness_array[1:] < 0)
                                        if robustness_array[0] < 0:
                                            # ignore first step if negative
                                            pass
                                    else:
                                        neg_steps = np.sum(robustness_array < 0)

                                    run_stats[rule]["steps_violated"] += neg_steps
                                    config_stats[rule]["violated_steps"] += neg_steps
                                    scenario_stats[scenario_setting][rule]["violated_steps"] += neg_steps

                                    if neg_steps > 0:
                                        run_stats[rule]["violated"] += 1
                                        config_stats[rule]["violated_scenarios"] += 1
                                        scenario_stats[scenario_setting][rule]["violated_scenarios"] += 1

                        except Exception as e:
                            print(f"❌ Error processing {file_name}: {e}")

                # Calculate ratios for this run and add to lists for std dev calculation
                for rule in rules:
                    if run_stats[rule]["total"] > 0:
                        scenario_ratio = run_stats[rule]["violated"] / run_stats[rule]["total"]
                        scenario_ratios[rule].append(scenario_ratio)

                    if run_stats[rule]["steps_total"] > 0:
                        step_ratio = run_stats[rule]["steps_violated"] / run_stats[rule]["steps_total"]
                        step_ratios[rule].append(step_ratio)

        # Calculate ratios for this scenario setting
        for rule in rules:
            if scenario_stats[scenario_setting][rule]["total_scenarios"] > 0:
                scenario_stats[scenario_setting][rule]["scenario_ratio"] = (
                        scenario_stats[scenario_setting][rule]["violated_scenarios"] /
                        scenario_stats[scenario_setting][rule]["total_scenarios"]
                )

            if scenario_stats[scenario_setting][rule]["total_steps"] > 0:
                scenario_stats[scenario_setting][rule]["step_ratio"] = (
                        scenario_stats[scenario_setting][rule]["violated_steps"] /
                        scenario_stats[scenario_setting][rule]["total_steps"]
                )

    # Calculate average ratios for this configuration
    for rule in rules:
        if config_stats[rule]["total_scenarios"] > 0:
            config_stats[rule]["scenario_ratio"] = config_stats[rule]["violated_scenarios"] / config_stats[rule][
                "total_scenarios"]

        if config_stats[rule]["total_steps"] > 0:
            config_stats[rule]["step_ratio"] = config_stats[rule]["violated_steps"] / config_stats[rule]["total_steps"]

        # Calculate standard deviations
        if scenario_ratios[rule]:
            config_stats[rule]["scenario_std"] = np.std(scenario_ratios[rule])
        else:
            config_stats[rule]["scenario_std"] = 0.0

        if step_ratios[rule]:
            config_stats[rule]["step_std"] = np.std(step_ratios[rule])
        else:
            config_stats[rule]["step_std"] = 0.0

    # Store results for this configuration
    all_results[selected_config] = {
        'config_stats': config_stats,
        'scenario_stats': scenario_stats
    }

    # Print results for this configuration
    print(f"\n{'=' * 80}")
    print(f"RESULTS FOR CONFIGURATION: {selected_config}")
    print(f"{'=' * 80}")

    total_runs = len(seeds) * len(scenario_settings) * 3  # 3 runs per seed-scenario combination
    print(f"Total runs processed: {total_runs}")

    # Print overall configuration results
    print(f"\nOVERALL RESULTS:")
    for rule in rules:
        if config_stats[rule]["total_scenarios"] > 0:
            print(f"\n{rule}:")
            print(f"  Scenarios: total={config_stats[rule]['total_scenarios']}, "
                  f"violated={config_stats[rule]['violated_scenarios']}, "
                  f"violation ratio={config_stats[rule]['scenario_ratio']:.3f} ± {config_stats[rule]['scenario_std']:.3f}")

            if config_stats[rule]["total_steps"] > 0:
                print(f"  Time steps: total={config_stats[rule]['total_steps']}, "
                      f"violated={config_stats[rule]['violated_steps']}, "
                      f"violation ratio={config_stats[rule]['step_ratio']:.3f} ± {config_stats[rule]['step_std']:.3f}")
        else:
            print(f"\n{rule}: no cases evaluated.")

    # Print results for each scenario setting
    print(f"\n{'=' * 50}")
    print(f"RESULTS BY SCENARIO SETTING:")
    print(f"{'=' * 50}")

    for scenario_setting in scenario_settings:
        lanes_count, vehicles_density = scenario_setting
        print(f"\nScenario Setting ({lanes_count}, {vehicles_density}):")

        for rule in rules:
            if scenario_stats[scenario_setting][rule]["total_scenarios"] > 0:
                print(f"  {rule}:")
                print(f"    Scenarios: total={scenario_stats[scenario_setting][rule]['total_scenarios']}, "
                      f"violated={scenario_stats[scenario_setting][rule]['violated_scenarios']}, "
                      f"violation ratio={scenario_stats[scenario_setting][rule]['scenario_ratio']:.3f}")

                if scenario_stats[scenario_setting][rule]["total_steps"] > 0:
                    print(f"    Time steps: total={scenario_stats[scenario_setting][rule]['total_steps']}, "
                          f"violated={scenario_stats[scenario_setting][rule]['violated_steps']}, "
                          f"violation ratio={scenario_stats[scenario_setting][rule]['step_ratio']:.3f}")
            else:
                print(f"  {rule}: no cases evaluated.")

# Print summary of all configurations
print(f"\n{'=' * 80}")
print(f"SUMMARY OF ALL CONFIGURATIONS")
print(f"{'=' * 80}")

for i, config in enumerate(config_combinations):
    config_data = all_results[config]
    config_stats = config_data['config_stats']

    print(f"\nConfiguration {i + 1}: {config}")
    print(f"{'-' * 50}")

    for rule in rules:
        if config_stats[rule]["total_scenarios"] > 0:
            print(f"  {rule}:")
            print(
                f"    Scenario violation ratio = {config_stats[rule]['scenario_ratio']:.3f} ± {config_stats[rule]['scenario_std']:.3f}")
            print(
                f"    Step violation ratio = {config_stats[rule]['step_ratio']:.3f} ± {config_stats[rule]['step_std']:.3f}")
        else:
            print(f"  {rule}: no cases evaluated.")

# Print detailed results by scenario setting for all configurations
print(f"\n{'=' * 80}")
print(f"DETAILED RESULTS BY SCENARIO SETTING FOR ALL CONFIGURATIONS")
print(f"{'=' * 80}")

for i, config in enumerate(config_combinations):
    config_data = all_results[config]
    scenario_stats = config_data['scenario_stats']

    print(f"\nConfiguration {i + 1}: {config}")
    print(f"{'-' * 50}")

    for scenario_setting in scenario_settings:
        lanes_count, vehicles_density = scenario_setting
        print(f"\n  Scenario Setting ({lanes_count}, {vehicles_density}):")

        for rule in rules:
            if scenario_stats[scenario_setting][rule]["total_scenarios"] > 0:
                print(f"    {rule}:")
                print(
                    f"      Scenario violation ratio = {scenario_stats[scenario_setting][rule]['scenario_ratio']:.3f}")
                print(f"      Step violation ratio = {scenario_stats[scenario_setting][rule]['step_ratio']:.3f}")
            else:
                print(f"    {rule}: no cases evaluated.")

print(f"\n{'=' * 80}")
print(f"All configurations processed:")
for i, config in enumerate(config_combinations, 1):
    print(f"Config {i}: {config}")
print(f"{'=' * 80}")