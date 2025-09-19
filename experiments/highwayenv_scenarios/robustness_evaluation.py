import numpy as np
import pandas as pd
from pathlib import Path
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem
from scenariogeneration import Scenario

from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator

def extract_ego_vehicle(
    scenario: Scenario, planning_problem: PlanningProblem
):
    ego_vehicle = None
    for vehicle in scenario.dynamic_obstacles:
        diff: np.ndarray = (
            vehicle.initial_state.position - planning_problem.initial_state.position
        )
        if np.linalg.norm(diff) < 0.1:
            ego_vehicle = vehicle
    return ego_vehicle

# === CONFIGURATION ===
rules = ["R_G1", "R_G2", "R_G3"]
SKIP_FIRST_STEP = True
folder = Path.cwd()   # gets the current working directory

# Collect all CSVs in folder
xml_files = list(folder.glob("*.xml"))
if not xml_files:
    raise FileNotFoundError(f"No xml files found in {folder}")

for xml_file in xml_files:
    print(f"\n=== Processing {xml_file.name} ===")

    # match scenario XML
    scenario_file = folder / (xml_file.stem + ".xml")
    if not scenario_file.exists():
        print(f"  ⚠️ No scenario XML for {xml_file.stem}, skipping.")
        continue

    try:
        # load scenario + ego vehicle
        scenario, pp_set = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        planning_problems = list(pp_set.planning_problem_dict.values())
        if not planning_problems:
            print(f"  ⚠️ No planning problems in {scenario_file.name}")
            continue
        planning_problem = planning_problems[0]
        ego_vehicle_traj = extract_ego_vehicle(scenario, planning_problem)
        print(ego_vehicle_traj.obstacle_id)
        world = World.create_from_scenario(scenario)
        ego_vehicle = None

        for vehicle in world.vehicles:
            if np.array_equal(vehicle.state_list_cr[0].position,
                              planning_problem.initial_state.position):
                ego_vehicle = vehicle
                break
        if ego_vehicle is None:
            print(f"  ⚠️ No ego vehicle found for {scenario_file.name}")
            continue

        # evaluate rules
        robustness_dict = {}
        for rule in rules:
            evaluator = RuleEvaluator.create_from_config(world, ego_vehicle, rule)
            robustness_array = evaluator.evaluate()
            if SKIP_FIRST_STEP:
                robustness_array = robustness_array[1:]
            robustness_dict[rule] = robustness_array

        # compute min robustness across rules (per timestep)
        stacked = np.vstack([robustness_dict[r] for r in rules])
        min_across_rules = np.min(stacked, axis=0)

        # print results
        for rule in rules:
            print(f"{rule} robustness: {robustness_dict[rule]}")
        print(f"Min robustness across all rules: {min_across_rules}")

    except Exception as e:
        print(f"  ❌ Error processing {scenario_file.name}: {e}")
