import pandas as pd
from pathlib import Path
from tqdm import tqdm
from commonroad.common.file_reader import CommonRoadFileReader

from sandra.utility.vehicle import extract_ego_vehicle
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.labeler import TrajectoryLabeler
from sandra.common.config import SanDRAConfiguration
from sandra.actions import LongitudinalAction, LateralAction

def _split_long_lat(actions):
    long_labels, lat_labels = [], []
    for pair in actions:
        long_labels.append(pair[0].value)
        lat_labels.append(pair[1].value)
    return long_labels, lat_labels

def add_trajectory_labels_and_match_to_csv(existing_csv_path, scenario_folder, config, output_csv_path):
    df = pd.read_csv(existing_csv_path)

    traj_long_all = []
    traj_lat_all = []
    match_top1_all = []
    match_topk_all = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Adding Trajectory Labels + Match"):
        scenario_id = row["ScenarioID"]
        scenario_path = Path(scenario_folder) / (scenario_id + ".xml")

        try:
            scenario, planning_problem_set = CommonRoadFileReader(str(scenario_path)).open()
            planning_problem = next(iter(planning_problem_set.planning_problem_dict.values()))
            ego_vehicle = extract_ego_vehicle(scenario, planning_problem)

            road_network = RoadNetwork.from_lanelet_network_and_position(
                scenario.lanelet_network,
                planning_problem.initial_state.position,
                consider_reversed=True,
            )
            ego_lane_network = EgoLaneNetwork.from_route_planner(
                scenario.lanelet_network, planning_problem, road_network
            )

            traj_labeler = TrajectoryLabeler(config, scenario)
            traj_actions = traj_labeler.label(ego_vehicle, ego_lane_network)
            traj_long, traj_lat = _split_long_lat(traj_actions)
            traj_long_all.append("; ".join(traj_long))
            traj_lat_all.append("; ".join(traj_lat))

            # === Match computation ===
            match_top1 = None
            match_topk = None

            ranking = []
            for i in range(config.k):
                long_col = f"{config.model_name}_Longitudinal_{i+1}"
                lat_col = f"{config.model_name}_Lateral_{i+1}"
                if long_col in row and lat_col in row:
                    long_str = row[long_col]
                    lat_str = row[lat_col]
                    if pd.notna(long_str) and pd.notna(lat_str) and long_str and lat_str:
                        try:
                            long_enum = LongitudinalAction(long_str)
                            lat_enum = LateralAction(lat_str)
                            ranking.append((long_enum, lat_enum))
                        except ValueError:
                            pass

            if ranking:
                top1 = ranking[0]
                match_top1 = (traj_actions[0][0] == top1[0]) and (traj_actions[0][1] == top1[1])
                match_topk = any(
                    (traj_actions[0][0] == pair[0]) and (traj_actions[0][1] == pair[1])
                    for pair in ranking
                )

            match_top1_all.append(int(match_top1) if match_top1 is not None else "")
            match_topk_all.append(int(match_topk) if match_topk is not None else "")

        except Exception as e:
            print(f"⚠️ Failed for {scenario_id}: {e}")
            traj_long_all.append("")
            traj_lat_all.append("")
            match_top1_all.append("")
            match_topk_all.append("")

    df["Trajectory_Longitudinal"] = traj_long_all
    df["Trajectory_Lateral"] = traj_lat_all
    df["Match_Top1"] = match_top1_all
    df["Match_TopK"] = match_topk_all

    df.to_csv(output_csv_path, index=False)
    print(f"✅ Updated CSV saved to: {output_csv_path}")

# === Example usage ===
existing_csv = "/home/liny/Documents/commonroad/mona-updated-fixed-selected-ruled/batch_labelling_results_ft:gpt-4o-2024-08-06:tum::BsuinSqR_20250724_212748.csv"
scenario_dir = "/home/liny/Documents/commonroad/mona-updated-fixed-selected-ruled/"

output_csv = existing_csv.replace(".csv", "_with_trajectory_and_match.csv")

config = SanDRAConfiguration()
config.h = 25
config.k = 3
config.model_name = "ft:gpt-4o-2024-08-06:tum::BsuinSqR"

add_trajectory_labels_and_match_to_csv(existing_csv, scenario_dir, config, output_csv)
