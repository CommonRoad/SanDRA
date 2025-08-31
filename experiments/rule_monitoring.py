import os
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator

# Folder containing scenarios
scenario_folder = "../scenarios_monitoring_5-3.0_reach_rule_True"
scenario_folder = "../scenarios_monitoring_4-3.0_reach_rule_True"
scenario_folder = "../scenarios_monitoring_4-3.0_reach_rule_False"
SKIP_FIRST_STEP = False

# Rules to evaluate
rules = ["R_G1", "R_G2", "R_G3"]

# Stats per rule
stats = {
    rule: {"total": 0, "violated": 0, "steps_total": 0, "steps_violated": 0}
    for rule in rules
}

for file_name in os.listdir(scenario_folder):
    if not file_name.endswith(".xml"):
        continue

    scenario_path = os.path.join(scenario_folder, file_name)
    print(f"Processing: {scenario_path}")

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

            stats[rule]["total"] += 1

            if robustness_array.shape[0] > 0:
                # Step counts
                stats[rule]["steps_total"] += robustness_array.shape[0]

                if SKIP_FIRST_STEP:
                    neg_steps = np.sum(robustness_array[1:] < 0)
                    if robustness_array[0] < 0:
                        # ignore first step if negative
                        pass
                else:
                    neg_steps = np.sum(robustness_array < 0)

                stats[rule]["steps_violated"] += neg_steps

                if neg_steps > 0:
                    stats[rule]["violated"] += 1

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")

# Final results
print("\n=== Summary per rule ===")
for rule in rules:
    total = stats[rule]["total"]
    violated = stats[rule]["violated"]
    steps_total = stats[rule]["steps_total"]
    steps_violated = stats[rule]["steps_violated"]

    if total > 0:
        scenario_ratio = violated / total
        print(f"{rule}: total={total}, violated={violated}, violation ratio={scenario_ratio:.2f}")

        if steps_total > 0:
            step_ratio = steps_violated / steps_total
            print(f"   time steps: total={steps_total}, violated={steps_violated}, violation ratio={step_ratio:.2f}")
    else:
        print(f"{rule}: no cases evaluated.")
