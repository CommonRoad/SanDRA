import itertools
import numpy as np
from pathlib import Path
from commonroad.common.file_reader import CommonRoadFileReader
from sandra.utility.vehicle import extract_ego_vehicle

# === CONFIGURATION ===
model_name = "gpt-4o"
use_sonia = False
scenario_settings = [(5, 3.0)]
seeds = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 4213, 2572]
base_dir = Path("/home/liny/Documents/commonroad/results-sandra/new-results-updated")

# Cartesian product of configs
configs = itertools.product(
    [True],   # use_rules_in_prompt
    [True],  # use_rules_in_reach
    ["not-set-based"],  # prediction type
    scenario_settings,
    seeds,
)

results = []
all_distances_global = []
finished_count_global = 0
total_runs_global = 0

# === PROCESS CONFIGS ===
for use_rules_in_prompt, use_rules_in_reach, prediction_type, (lanes_count, vehicle_density), seed in configs:
    # Determine expected folder name
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
                scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
                planning_problems = list(planning_problem_set.planning_problem_dict.values())
                if not planning_problems:
                    continue
                planning_problem = planning_problems[0]
                ego_vehicle = extract_ego_vehicle(scenario, planning_problem)

                trajectory = ego_vehicle.prediction.trajectory
                print(trajectory.final_state.time_step)
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
            except Exception as e:
                print(f"  {file_path.name}: error ({e})")

    # Store results per config
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

# === OVERALL AVERAGES ===
if all_distances_global:
    avg_distance_global = np.mean(all_distances_global)
    print(f"\nAverage longitudinal traveled distance (all finished runs): {avg_distance_global:.2f} m")
else:
    print("\nNo finished distances computed.")

if total_runs_global > 0:
    finished_rate_global = finished_count_global / total_runs_global * 100
    print(f"Finished runs (all configs): {finished_count_global}/{total_runs_global} ({finished_rate_global:.1f}%)")
else:
    print("No runs processed.")
