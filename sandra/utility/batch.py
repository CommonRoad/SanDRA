import csv
import os
import re
from datetime import datetime
from typing import List, Tuple

from commonroad.common.file_reader import CommonRoadFileReader
from tqdm import tqdm

from sandra.common.config import SanDRAConfiguration
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.commonroad.reach import ReachVerifier
from sandra.decider import Decider
from sandra.labeler import TrajectoryLabeler, ReachSetLabeler
from sandra.llm import get_structured_response
from sandra.utility.vehicle import extract_ego_vehicle
from sandra.verifier import VerificationStatus


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
    role: str = None,
    evaluate_prompt: bool = True,
    evaluate_llm: bool = False,
    evaluate_safety: bool = False,
    evaluate_trajectory_labels: bool = True,
    evaluate_reachset_labels: bool = True,
    nr_scenarios: int = None
):
    scenario_entries = load_scenarios_recursively(scenario_folder)

    if not scenario_entries:
        print("No scenarios found to process.")
        return

    if role:
        safe_role = re.sub(r"[^a-zA-Z0-9_]+", "", role.replace(" ", "_").lower())
        filename = f"batch_labelling_results_{safe_role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        filename = f"batch_labelling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(scenario_folder, filename)

    total_scenarios = 0
    top1_hits = 0
    topk_hits = 0

    llm1_safe = 0
    llmk_safe = 0
    highd_safe = 0

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        headers = ["ScenarioID", "EgoID"]

        if evaluate_prompt:
            headers.append("Prompt")

        if evaluate_llm:
            for i in range(1, config.k + 1):
                headers.append(f"{config.model_name}_Longitudinal_{i}")
                headers.append(f"{config.model_name}_Lateral_{i}")

        if evaluate_trajectory_labels:
            headers.extend(["Trajectory_Longitudinal", "Trajectory_Lateral"])

        if evaluate_reachset_labels:
            for i in range(1, config.k + 1):
                headers.append(f"ReachSet_Longitudinal_{i}")
                headers.append(f"ReachSet_Lateral_{i}")

        if evaluate_llm and evaluate_safety:
            headers.extend(["Safe_Top1", "Safe_TopK"])
            if evaluate_trajectory_labels:
                headers.extend(["highD_safe"])

        if evaluate_llm and evaluate_trajectory_labels:
            headers.extend(["Match_Top1", "Match_TopK"])

        writer.writerow(headers)

        for i, (scenario_id, file_dir) in enumerate(
                tqdm(scenario_entries, desc="Scenarios processed", colour="red")
        ):
            if nr_scenarios is not None and i >= nr_scenarios:
                break
            scenario_path = os.path.join(file_dir, scenario_id + ".xml")
            print(f"\nProcessing scenario '{scenario_id}' in {file_dir}")
            try:
                scenario, planning_problem_set = CommonRoadFileReader(
                    str(scenario_path)
                ).open(lanelet_assignment=True)

                planning_problem = next(
                    iter(planning_problem_set.planning_problem_dict.values())
                )

                prompt = None
                ranking_long = []
                ranking_lat = []
                ranking = []

                if evaluate_prompt:
                    describer = CommonRoadDescriber(
                        scenario,
                        planning_problem,
                        0,
                        config,
                        role=role
                    )
                    decider = Decider(config, describer)
                    system_prompt = decider.describer.system_prompt()
                    user_prompt = decider.describer.user_prompt()
                    prompt = system_prompt + user_prompt

                    if evaluate_llm:
                        schema = decider.describer.schema()
                        structured_response = get_structured_response(
                            user_prompt, system_prompt, schema, decider.config
                        )
                        ranking = decider._parse_action_ranking(structured_response)
                        ranking = [list(action_pair) for action_pair in ranking]
                        ranking_long, ranking_lat = _split_long_lat(ranking)

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
                else:
                    traj_actions = []

                if evaluate_reachset_labels:
                    if scenario.obstacle_by_id(ego_vehicle.obstacle_id):
                        scenario.remove_obstacle(
                            scenario.obstacle_by_id(ego_vehicle.obstacle_id)
                        )
                    reach_labeler = ReachSetLabeler(
                        config, scenario, planning_problem, scenario_folder=scenario_folder
                    )
                    reach_actions = reach_labeler.label(ego_vehicle, ego_lane_network)
                    reach_long, reach_lat = _split_long_lat(reach_actions)

                llm1_verified = None
                llmk_verified = None
                highd_verified = None

                if evaluate_prompt and evaluate_llm and evaluate_safety:
                    reach_ver = ReachVerifier(scenario, planning_problem, config, ego_lane_network,
                                              scenario_folder=scenario_folder)
                    status = reach_ver.verify(
                        ranking[0], visualization=False
                    )
                    llm1_verified = status == VerificationStatus.SAFE
                    if llm1_verified == True:
                        llmk_verified = True
                    else:
                        for action_pair in ranking[1:]:
                            status = reach_ver.verify(
                                action_pair, visualization=False
                            )
                            llmk_verified = status == VerificationStatus.SAFE
                            if llmk_verified == True:
                                break

                    llm1_safe += llm1_verified
                    llmk_safe += llmk_verified

                    if evaluate_trajectory_labels:
                        status = reach_ver.verify(
                            traj_actions[0], visualization=False
                        )
                        highd_verified = status == VerificationStatus.SAFE

                    highd_safe += highd_verified

                match_top1 = None
                match_topk = None
                if evaluate_llm and evaluate_trajectory_labels and ranking and traj_actions:
                    # Compare tuple equality
                    match_top1 = tuple(traj_actions[0]) == tuple(ranking[0])

                    # Check if ground truth action appears in any rank
                    match_topk = tuple(traj_actions[0]) in [tuple(r) for r in ranking]

                    top1_hits += match_top1
                    topk_hits += match_topk

                total_scenarios += 1

                _write_labels_row(
                    writer,
                    scenario_id,
                    ego_vehicle.obstacle_id,
                    prompt,
                    ranking_long,
                    ranking_lat,
                    traj_long,
                    traj_lat,
                    reach_long,
                    reach_lat,
                    llm1_verified,
                    llmk_verified,
                    highd_verified,
                    match_top1,
                    match_topk,
                    evaluate_prompt,
                    evaluate_llm,
                    evaluate_safety,
                    evaluate_trajectory_labels,
                    evaluate_reachset_labels,
                    config.k,
                )

            except Exception as e:
                print(f"Failed to label '{scenario_id}': {e}")

    if total_scenarios > 0:
        ratio_top1 = top1_hits / total_scenarios
        ratio_topk = topk_hits / total_scenarios

        ratio_safe1 = llm1_safe / total_scenarios
        ratio_safek = llmk_safe / total_scenarios
        ratio_highd = highd_safe / total_scenarios

        print("\n📊 Matching Statistics:")
        print(f"  Match Top-1 Accuracy: {ratio_top1:.2%} ({top1_hits}/{total_scenarios})")
        print(f"  Match Top-K Accuracy: {ratio_topk:.2%} ({topk_hits}/{total_scenarios})")

        print("\n📊 Safe Statistics:")
        print(f"  Safe with 1 action: {ratio_safe1:.2%} ({llm1_safe}/{total_scenarios})")
        print(f"  Safe with k actions: {ratio_safek:.2%} ({llmk_safe}/{total_scenarios})")
        print(f"  highD safely labeled: {ratio_highd:.2%} ({highd_safe}/{total_scenarios})")
    else:
        print("\nNo scenarios were evaluated for matching.")

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
    prompt: str,
    ranking_long: List[str],
    ranking_lat: List[str],
    traj_long: List[str],
    traj_lat: List[str],
    reach_long: List[str],
    reach_lat: List[str],
    llm1_safe: int,
    llmk_safe: int,
    highd_safe: int,
    match_top1: int,
    match_topk: int,
    eval_prompt: bool,
    eval_llm: bool,
    eval_safety: bool,
    eval_traj: bool,
    eval_reach: bool,
    max_steps: int,
):
    row = [scenario_id, ego_id]

    if eval_prompt:
        row.append(prompt)

    if eval_llm:
        for i in range(max_steps):
            long_label = ranking_long[i] if i < len(ranking_long) else ""
            lat_label = ranking_lat[i] if i < len(ranking_lat) else ""
            row.extend([long_label, lat_label])

    if eval_traj:
        row.extend([
            "; ".join(traj_long),
            "; ".join(traj_lat),
        ])

    if eval_reach:
        for i in range(max_steps):
            long_label = reach_long[i] if i < len(reach_long) else ""
            lat_label = reach_lat[i] if i < len(reach_lat) else ""
            row.extend([long_label, lat_label])

    if eval_llm and eval_safety:
        row.append(llm1_safe)
        row.append(llmk_safe)
        if eval_traj:
            row.append(highd_safe)

    if eval_llm and eval_traj:
        row.append(match_top1)
        row.append(match_topk)

    writer.writerow(row)


if __name__ == "__main__":
    scenarios_path = "/home/liny/Documents/commonroad/highD-sandra-0.04/"
    config = SanDRAConfiguration()
    config.h = 25
    config.k = 3
    batch_labelling(
        scenarios_path,
        config,
        # role="Drive cautiously", # aggressively
        evaluate_prompt=True,
        evaluate_llm=False,
        evaluate_safety=False,
        evaluate_trajectory_labels=True,
        evaluate_reachset_labels=False,
        nr_scenarios=10000
    )
