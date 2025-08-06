import csv
import os
import re
from datetime import datetime
from typing import List, Tuple

from commonroad.common.file_reader import CommonRoadFileReader
from tqdm import tqdm

from config.sandra import SanDRAConfiguration
from sandra.utility.road_network import RoadNetwork, EgoLaneNetwork
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.commonroad.reach import ReachVerifier
from sandra.decider import Decider
from sandra.labeler import TrajectoryLabeler
from sandra.utility.vehicle import extract_ego_vehicle
from sandra.verifier import VerificationStatus

import matplotlib

print(matplotlib.get_backend())
matplotlib.use("Agg")


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


def create_csv(writer, sandra_config: SanDRAConfiguration):
    headers = ["ScenarioID", "EgoID"]
    headers.append("Prompt")

    for i in range(1, sandra_config.k + 1):
        headers.append(f"{sandra_config.model_name}_Longitudinal_{i}")
        headers.append(f"{sandra_config.model_name}_Lateral_{i}")

    headers.extend(["Trajectory_Longitudinal", "Trajectory_Lateral"])
    headers.extend(["Safe_Top1", "Safe_TopK"])
    headers.extend(["MONA_safe"])
    headers.extend(["Match_Top1", "Match_TopK"])
    headers.append("Inference_Duration")
    headers.append("Reach_Duration")
    writer.writerow(headers)


def write_labels_row(
    writer: csv.writer,
    scenario_id: str,
    ego_id: int,
    prompt: str,
    ranking_long: List[str],
    ranking_lat: List[str],
    traj_long: List[str],
    traj_lat: List[str],
    llm1_safe: int,
    llmk_safe: int,
    highd_safe: int,
    match_top1: int,
    match_topk: int,
    max_steps: int,
    inf_duration: float,
    reach_duration: float
):
    row = [scenario_id, ego_id]
    row.append(prompt)
    for i in range(max_steps):
        long_label = ranking_long[i] if i < len(ranking_long) else ""
        lat_label = ranking_lat[i] if i < len(ranking_lat) else ""
        row.extend([long_label, lat_label])
    row.extend(
        [
            "; ".join(traj_long),
            "; ".join(traj_lat),
        ]
    )
    row.append(llm1_safe)
    row.append(llmk_safe)
    row.append(highd_safe)
    row.append(match_top1)
    row.append(match_topk)
    row.append(str(inf_duration))
    row.append(str(reach_duration))
    writer.writerow(row)


def run(
    scenario_folder: str,
    sandra_config: SanDRAConfiguration,
    role: str = None,
    nr_scenarios: int = None,
):
    scenario_entries = load_scenarios_recursively(scenario_folder)

    if not scenario_entries:
        print("No scenarios found to process.")
        return

    if role:
        safe_role = re.sub(r"[^a-zA-Z0-9_]+", "", role.replace(" ", "_").lower())
        filename = f"batch_labelling_results_{sandra_config.model_name}_{safe_role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        filename = (
            f"batch_labelling_results_{sandra_config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    csv_path = os.path.join(scenario_folder, filename)

    total_scenarios = 0
    top1_hits = 0
    topk_hits = 0
    llm1_safe = 0
    llmk_safe = 0
    highd_safe = 0

    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        create_csv(writer, sandra_config)

        nr = 0
        for i, (scenario_id, file_dir) in enumerate(
            tqdm(scenario_entries, desc="Scenarios processed", colour="red")
        ):
            if nr_scenarios is not None and nr >= nr_scenarios:
                break
            scenario_path = os.path.join(file_dir, scenario_id + ".xml")
            print(f"\nProcessing scenario '{scenario_id}' in {file_dir}")
            try:
                # initialize all necessary objects
                scenario, planning_problem_set = CommonRoadFileReader(
                    str(scenario_path)
                ).open(lanelet_assignment=True)

                planning_problem = next(
                    iter(planning_problem_set.planning_problem_dict.values())
                )
                describer = CommonRoadDescriber(
                    scenario, planning_problem, 0, sandra_config, role=role
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
                verifier = ReachVerifier(
                    scenario,
                    planning_problem,
                    sandra_config,
                    ego_lane_network,
                    scenario_folder=scenario_folder,
                )
                decider = Decider(
                    sandra_config,
                    describer,
                    verifier,
                )

                # obtain a verified action from the LLM
                ranking, verification_idx, inf_duration, reach_duration = decider.decide(include_metadata=True)
                ranking_long, ranking_lat = zip(*ranking)

                traj_labeler = TrajectoryLabeler(sandra_config, scenario)
                traj_actions = traj_labeler.label(ego_vehicle, ego_lane_network)
                traj_long, traj_lat = zip(*traj_actions)

                llm1_verified = verification_idx == 0
                llmk_verified = verification_idx < len(ranking)
                llm1_safe += int(llm1_verified)
                llmk_safe += int(llmk_verified)
                status = verifier.verify(traj_actions[0])
                highd_verified = status == VerificationStatus.SAFE
                if status == VerificationStatus.UNSAFE:
                    continue
                if highd_verified:
                    highd_safe += highd_verified

                match_top1 = tuple(traj_actions[0]) == tuple(ranking[0])
                match_topk = tuple(traj_actions[0]) in [tuple(r) for r in ranking]
                top1_hits += match_top1
                topk_hits += match_topk
                total_scenarios += 1

                write_labels_row(
                    writer,
                    scenario_id,
                    ego_vehicle.obstacle_id,
                    describer.system_prompt() + describer.user_prompt(),
                    [x.value for x in ranking_long],
                    [x.value for x in ranking_lat],
                    [x.value for x in traj_long],
                    [x.value for x in traj_lat],
                    llm1_verified,
                    llmk_verified,
                    highd_verified,
                    match_top1,
                    match_topk,
                    sandra_config.k,
                    inf_duration,
                    reach_duration
                )
                nr += 1
            except Exception as e:
                print(f"Failed to label '{scenario_id}': {e}")

    if total_scenarios > 0:
        ratio_top1 = top1_hits / total_scenarios
        ratio_topk = topk_hits / total_scenarios

        ratio_safe1 = llm1_safe / total_scenarios
        ratio_safek = llmk_safe / total_scenarios
        ratio_highd = highd_safe / total_scenarios

        print("\n📊 Matching Statistics:")
        print(
            f"  Match Top-1 Accuracy: {ratio_top1:.2%} ({top1_hits}/{total_scenarios})"
        )
        print(
            f"  Match Top-K Accuracy: {ratio_topk:.2%} ({topk_hits}/{total_scenarios})"
        )

        print("\n📊 Safe Statistics:")
        print(
            f"  Safe with 1 action: {ratio_safe1:.2%} ({llm1_safe}/{total_scenarios})"
        )
        print(
            f"  Safe with k actions: {ratio_safek:.2%} ({llmk_safe}/{total_scenarios})"
        )
        print(
            f"  MONA safely labeled: {ratio_highd:.2%} ({highd_safe}/{total_scenarios})"
        )
    else:
        print("\nNo scenarios were evaluated for matching.")

    print(f"\n✅ All labels saved to: {csv_path}")


if __name__ == "__main__":
    scenarios_path = "/home/sebastian/Documents/Uni/Sandra/mona_scenarios/"
    config = SanDRAConfiguration()
    config.h = 25
    config.k = 3
    run(
        scenarios_path,
        config,
    )