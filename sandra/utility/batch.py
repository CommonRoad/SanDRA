import csv
import os
from typing import List, Tuple, Set

from commonroad.common.file_reader import CommonRoadFileReader
from tqdm import tqdm

from sandra.common.config import SanDRAConfiguration
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.labeler import TrajectoryLabeler, ReachSetLabeler
from sandra.utility.vehicle import extract_ego_vehicle


def load_scenarios_recursively(scenario_folder: str) -> List[Tuple[str, str]]:
    """
    Recursively search for XML scenario files and return their IDs and directories.

    Args:
        scenario_folder (str): Root folder to search.

    Returns:
        List[Tuple[str, str]]: List of (scenario_id, directory) tuples.
    """
    scenario_ids = []
    if not os.path.isdir(scenario_folder):
        raise ValueError(f"Provided path '{scenario_folder}' is not a valid directory.")

    for root, dirs, files in os.walk(scenario_folder):
        for file in files:
            if file.endswith(".xml"):
                scenario_id = os.path.splitext(file)[0]
                scenario_ids.append((scenario_id, root))

    return scenario_ids

def batch_labelling(scenario_folder: str, config: SanDRAConfiguration):
    """
    Batch-process all scenario XML files in the given folder and label their actions.
    Results are saved to a single CSV file in the folder.
    """
    scenario_entries = load_scenarios_recursively(scenario_folder)

    if not scenario_entries:
        print("No scenarios found to process.")
        return

    # Define output CSV path
    csv_path = os.path.join(scenario_folder, "batch_labelling_results.csv")

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["ScenarioID", "TrajectoryLabels", "ReachSetLabels"])

        for scenario_id, file_dir in tqdm(
            scenario_entries, desc="Scenarios processed", colour="red"
        ):
            scenario_path = os.path.join(file_dir, scenario_id + ".xml")
            print(f"\nProcessing scenario '{scenario_id}' in {file_dir}")
            try:
                # Read scenario
                scenario, planning_problem_set = CommonRoadFileReader(str(scenario_path)).open(
                    lanelet_assignment=True
                )
                planning_problem = next(
                    iter(planning_problem_set.planning_problem_dict.values())
                )
                ego_vehicle = extract_ego_vehicle(scenario, planning_problem)

                # Build ego lane network
                road_network = RoadNetwork.from_lanelet_network_and_position(
                    scenario.lanelet_network,
                    planning_problem.initial_state.position,
                    consider_reversed=True,
                )
                ego_lane_network = EgoLaneNetwork.from_route_planner(
                    scenario.lanelet_network,
                    planning_problem,
                    road_network,
                )

                # Labelling with trajectories
                traj_labeler = TrajectoryLabeler(config, scenario)
                traj_actions = traj_labeler.label(ego_vehicle, ego_lane_network)

                # Labelling with reachable sets' areas
                if scenario.obstacle_by_id(ego_vehicle.obstacle_id):
                    scenario.remove_obstacle(scenario.obstacle_by_id(ego_vehicle.obstacle_id))
                reach_labeler = ReachSetLabeler(config, scenario, scenario_folder=scenario_folder)
                reach_actions = reach_labeler.label(ego_vehicle, ego_lane_network)

                # Write this scenario's row to CSV
                _write_labels_row(
                    writer,
                    scenario_id,
                    traj_actions,
                    reach_actions
                )
            except Exception as e:
                print(f"Failed to label '{scenario_id}': {e}")

    print(f"\n✅ All labels saved to: {csv_path}")


def _write_labels_row(
    writer: csv.writer,
    scenario_id: str,
    traj_actions: List[Set],
    reach_actions: List[Set]
):
    """
    Write one row to the CSV: ScenarioID, TrajectoryLabels, ReachSetLabels.
    Each label list is joined by '; '.
    """
    def serialize_actions(actions: List[Set]) -> str:
        return "; ".join(
            ",".join(sorted(action.value for action in action_set))
            for action_set in actions
        )

    traj_str = serialize_actions(traj_actions)
    reach_str = serialize_actions(reach_actions)

    writer.writerow([scenario_id, traj_str, reach_str])

if __name__ == '__main__':
    scenarios_path = "/home/liny/Documents/commonroad/highd_scenarios/"
    config = SanDRAConfiguration()
    config.h = 20
    batch_labelling(scenarios_path, config)