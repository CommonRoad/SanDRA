import csv
import os
from typing import List, Tuple

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


def batch_labelling(
    scenario_folder: str,
    config: SanDRAConfiguration,
    evaluate_trajectory_labels: bool = True,
    evaluate_reachset_labels: bool = True,
):
    scenario_entries = load_scenarios_recursively(scenario_folder)

    if not scenario_entries:
        print("No scenarios found to process.")
        return

    csv_path = os.path.join(scenario_folder, "batch_labelling_results.csv")

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Prepare header dynamically
        headers = ["ScenarioID", "EgoID"]

        if evaluate_trajectory_labels:
            headers.extend(["Trajectory_Longitudinal", "Trajectory_Lateral"])

        if evaluate_reachset_labels:
            # We don't know max horizon upfront, so we use a placeholder and adjust later
            max_reach_steps = config.m
            for i in range(1, max_reach_steps + 1):
                headers.append(f"ReachSet_Longitudinal_{i}")
                headers.append(f"ReachSet_Lateral_{i}")

        writer.writerow(headers)

        for scenario_id, file_dir in tqdm(
            scenario_entries, desc="Scenarios processed", colour="red"
        ):
            scenario_path = os.path.join(file_dir, scenario_id + ".xml")
            print(f"\nProcessing scenario '{scenario_id}' in {file_dir}")
            try:
                scenario, planning_problem_set = CommonRoadFileReader(
                    str(scenario_path)
                ).open(lanelet_assignment=True)
                planning_problem = next(
                    iter(planning_problem_set.planning_problem_dict.values())
                )
                ego_vehicle = extract_ego_vehicle(scenario, planning_problem)

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

                traj_long = []
                traj_lat = []
                reach_long = []
                reach_lat = []

                if evaluate_trajectory_labels:
                    traj_labeler = TrajectoryLabeler(config, scenario)
                    traj_actions = traj_labeler.label(ego_vehicle, ego_lane_network)
                    traj_long, traj_lat = _split_long_lat(traj_actions)

                if evaluate_reachset_labels:
                    if scenario.obstacle_by_id(ego_vehicle.obstacle_id):
                        scenario.remove_obstacle(
                            scenario.obstacle_by_id(ego_vehicle.obstacle_id)
                        )
                    reach_labeler = ReachSetLabeler(
                        config, scenario, scenario_folder=scenario_folder
                    )
                    reach_actions = reach_labeler.label(ego_vehicle, ego_lane_network)
                    reach_long, reach_lat = _split_long_lat(reach_actions)

                _write_labels_row(
                    writer,
                    scenario_id,
                    ego_vehicle.obstacle_id,
                    traj_long,
                    traj_lat,
                    reach_long,
                    reach_lat,
                    evaluate_trajectory_labels,
                    evaluate_reachset_labels,
                    max_reach_steps=config.h,
                )

            except Exception as e:
                print(f"Failed to label '{scenario_id}': {e}")

    print(f"\n✅ All labels saved to: {csv_path}")


def _split_long_lat(actions: List[List]) -> Tuple[List[str], List[str]]:
    """
    Split [Longitudinal, Lateral] pairs into two lists of string labels.
    """
    long_labels = []
    lat_labels = []
    for action_pair in actions:
        long_labels.append(action_pair[0].value)
        lat_labels.append(action_pair[1].value)
    return long_labels, lat_labels


def _serialize_list(labels: List[str]) -> str:
    """
    Join labels by semicolons (only used for trajectory labels).
    """
    return "; ".join(labels)


def _write_labels_row(
    writer: csv.writer,
    scenario_id: str,
    ego_id: int,
    traj_long: List[str],
    traj_lat: List[str],
    reach_long: List[str],
    reach_lat: List[str],
    eval_traj: bool,
    eval_reach: bool,
    max_reach_steps: int,
):
    """
    Write a row with separate columns for each reachset time step.
    """
    row = [scenario_id, ego_id]

    if eval_traj:
        row.extend(
            [
                _serialize_list(traj_long),
                _serialize_list(traj_lat),
            ]
        )

    if eval_reach:
        # Fill reach labels step by step
        for i in range(max_reach_steps):
            # If there are fewer labels than max_reach_steps, fill empty
            long_label = reach_long[i] if i < len(reach_long) else ""
            lat_label = reach_lat[i] if i < len(reach_lat) else ""
            row.extend([long_label, lat_label])

    writer.writerow(row)


if __name__ == "__main__":
    scenarios_path = "/home/liny/Documents/commonroad/highd_scenarios/"
    config = SanDRAConfiguration()
    config.h = 20
    batch_labelling(
        scenarios_path,
        config,
        evaluate_trajectory_labels=True,
        evaluate_reachset_labels=True,
    )
