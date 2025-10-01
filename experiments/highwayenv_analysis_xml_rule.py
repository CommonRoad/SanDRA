import itertools
import numpy as np
from pathlib import Path

import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from sandra.utility.general import extract_ego_vehicle
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator

# === CONFIGURATION ===
model_name = "gpt-4o"
scenario_settings = [(4, 2.0)]
seeds = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 4213, 2572]
base_dir = Path("/home/liny/Documents/commonroad/results-sandra/new-results-updated")
rules = ["R_G1" , "R_G2", "R_G3"]
SKIP_FIRST_STEP = True

# Cartesian product of configs
configs = itertools.product(
    [True],   # use_rules_in_prompt
    [True],  # use_rules_in_reach
    ["not-set-based"],  # prediction type
    scenario_settings,
    seeds,
)

# Aggregated results
results = []
all_distances_global = []
finished_count_global = 0
total_runs_global = 0

# Initialize aggregated rule stats
aggregated_stats = {rule: {"total_runs": 0, "total_steps": 0, "compliant_steps": 0, "violated_runs": 0} for rule in rules}
ensemble_stats = {"total_runs": 0, "total_steps": 0, "compliant_steps": 0, "violated_runs": 0}

# === PROCESS CONFIGS ===
for use_rules_in_prompt, use_rules_in_reach, prediction_type, (lanes_count, vehicle_density), seed in configs:
    expected_pattern = (
        f"results-True-{model_name}-{lanes_count}-{vehicle_density}-{seed}"
        f"-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"
    ) if not prediction_type == "set-based" else (
        f"results-True-{model_name}-{lanes_count}-{vehicle_density}-{seed}"
        f"-spot-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"
    )

    print(f"\nLooking for folder: {expected_pattern}")

    all_distances = []
    finished_count = 0
    total_runs = 0

    seed_dir = base_dir / str(seed)
    if not seed_dir.exists():
        continue

    for run_folder in seed_dir.iterdir():
        if not run_folder.is_dir() or not run_folder.name.startswith("run-"):
            continue

        result_folder = run_folder / expected_pattern
        if not result_folder.exists():
            continue

        print(f"Processing folder: {result_folder}")

        for file_path in result_folder.glob("*.xml"):
            total_runs += 1
            total_runs_global += 1
            try:
                scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
                planning_problems = list(planning_problem_set.planning_problem_dict.values())
                if not planning_problems:
                    continue
                planning_problem = planning_problems[0]
                ego_vehicle_traj = extract_ego_vehicle(scenario, planning_problem)

                # --- Trajectory distances ---
                trajectory = ego_vehicle_traj.prediction.trajectory
                if trajectory and trajectory.final_state.time_step in [30, 31]:
                    finished_count += 1
                    finished_count_global += 1
                    x_positions = [s.position[0] for s in trajectory.state_list]
                    longitudinal_distance = x_positions[-1] - x_positions[0]
                    all_distances.append(longitudinal_distance)
                    all_distances_global.append(longitudinal_distance)
                    print(f"  {file_path.name}: finished, distance = {longitudinal_distance:.2f} m")
                else:
                    print(f"  {file_path.name}: did not finish")

                # --- Rule evaluation ---
                world = World.create_from_scenario(scenario)
                ego_vehicle = None
                for vehicle in world.vehicles:
                    if np.array_equal(vehicle.state_list_cr[0].position, planning_problem.initial_state.position):
                        ego_vehicle = vehicle
                        break
                if ego_vehicle is None:
                    continue

                all_rules_compliant_arrays = []
                for rule in rules:
                    evaluator = RuleEvaluator.create_from_config(world, ego_vehicle, rule)
                    robustness_array = evaluator.evaluate()[1:]
                    aggregated_stats[rule]["total_runs"] += 1
                    robustness_to_check = robustness_array[1:] if SKIP_FIRST_STEP else robustness_array
                    aggregated_stats[rule]["total_steps"] += robustness_to_check.size
                    if rule == "R_G2":
                        csv_file = result_folder / "evaluation.csv"
                        df = None
                        if csv_file.exists():
                            try:
                                df = pd.read_csv(csv_file)
                                df["iteration-id"] = df["iteration-id"].astype(int)  # ensure type matches
                            except Exception as e:
                                print(f"Error reading {csv_file}: {e}")

                        compliant_steps = 0
                        idx = 1
                        for rob in robustness_to_check:
                            if df is not None:
                                row = df.loc[df["iteration-id"] == str(idx), "verified-id"]
                                if not row.empty:
                                    verified_value = int(row.values[0])
                                    if rob > 0 or (rob < 0 and verified_value == 3):
                                        compliant_steps += 1
                            else:
                                # fallback if CSV missing
                                if rob >= 0:
                                    compliant_steps += 1
                            idx += 1

                        aggregated_stats[rule]["compliant_steps"] += compliant_steps
                        if np.any(robustness_to_check < 0):
                            aggregated_stats[rule]["violated_runs"] += 1

                        all_rules_compliant_arrays.append(robustness_to_check >= 0)
                    else:
                        compliant_steps = np.sum(robustness_to_check >= 0)
                        aggregated_stats[rule]["compliant_steps"] += compliant_steps
                        if np.any(robustness_to_check < 0):
                            aggregated_stats[rule]["violated_runs"] += 1
                        all_rules_compliant_arrays.append(robustness_to_check >= 0)

                if all_rules_compliant_arrays:
                    ensemble_compliant = np.logical_and.reduce(all_rules_compliant_arrays)
                    ensemble_stats["total_runs"] += 1
                    ensemble_stats["total_steps"] += ensemble_compliant.size
                    ensemble_stats["compliant_steps"] += np.sum(ensemble_compliant)
                    if np.any(~ensemble_compliant):
                        ensemble_stats["violated_runs"] += 1

            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")

    # Store per-config results
    avg_distance = np.mean(all_distances) if all_distances else 0.0
    finished_rate = (finished_count / total_runs * 100) if total_runs > 0 else 0.0
    results.append({
        "use_rules_in_prompt": use_rules_in_prompt,
        "use_rules_in_reach": use_rules_in_reach,
        "prediction_type": prediction_type,
        "lanes_count": lanes_count,
        "vehicle_density": vehicle_density,
        "seed": seed,
        "avg_distance": avg_distance,
        "finished_count": finished_count,
        "total_runs": total_runs,
        "finished_rate": finished_rate,
    })

# === PER-CONFIG SUMMARY ===
print("\n=== Summary of Results Per Config ===")
for res in results:
    print(
        f"Config: lanes={res['lanes_count']} density={res['vehicle_density']} "
        f"seed={res['seed']} prompt={res['use_rules_in_prompt']} reach={res['use_rules_in_reach']} "
        f"pred={res['prediction_type']} -> "
        f"avg_dist={res['avg_distance']:.2f} m, finished={res['finished_count']}/{res['total_runs']} "
        f"({res['finished_rate']:.1f}%)"
    )

# === OVERALL DISTANCE AVERAGES ===
if all_distances_global:
    print(f"\nAverage longitudinal traveled distance (all finished runs): {np.mean(all_distances_global):.2f} m")
else:
    print("\nNo finished distances computed.")

if total_runs_global > 0:
    print(f"Finished runs (all configs): {finished_count_global}/{total_runs_global} "
          f"({finished_count_global / total_runs_global * 100:.1f}%)")
else:
    print("No runs processed.")

# === AGGREGATED RULE COMPLIANCE ===
print("\n=== Aggregated Rule Compliance (Per Rule) ===")
for rule, stats in aggregated_stats.items():
    avg_compliant = stats["compliant_steps"] / stats["total_runs"] if stats["total_runs"] > 0 else 0
    avg_total = stats["total_steps"] / stats["total_runs"] if stats["total_runs"] > 0 else 0
    print(f"{rule}: Avg compliant steps = {avg_compliant:.2f}, Avg total steps = {avg_total:.2f}, "
          f"Runs violated = {stats['violated_runs']}, Total runs = {stats['total_runs']}")

# === ENSEMBLED RULE COMPLIANCE ===
avg_compliant = ensemble_stats["compliant_steps"] / ensemble_stats["total_runs"] if ensemble_stats["total_runs"] > 0 else 0
avg_total = ensemble_stats["total_steps"] / ensemble_stats["total_runs"] if ensemble_stats["total_runs"] > 0 else 0
print("\n=== Ensemble Rule Compliance (All Rules) ===")
print(f"Average compliant steps = {avg_compliant:.2f}, "
      f"Average total steps = {avg_total:.2f}, "
      f"Runs violated = {ensemble_stats['violated_runs']}, "
      f"Total runs = {ensemble_stats['total_runs']}")
