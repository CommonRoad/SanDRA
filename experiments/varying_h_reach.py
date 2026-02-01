import os
import pandas as pd
import matplotlib
from enum import Enum

from sandra.verifier import VerificationStatus
from sandra.common.config import SanDRAConfiguration
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.commonroad.reach import ReachVerifier
from sandra.utility.general import extract_scenario_and_planning_problem
from sandra.actions import LongitudinalAction, LateralAction

# -------------------------
# Matplotlib backend
# -------------------------
print(matplotlib.get_backend())
matplotlib.use("TkAgg")
matplotlib.use("Agg")  # <- prevents GUI pop-up
# -------------------------
# Paths
# -------------------------
BATCH_FILE = "./data/rule_prompt/batch_labelling_results_gpt-4o_20250912_102453-rule_True.csv"
SCENARIO_DIR = "/home/liny/Documents/commonroad/mona-updated/"

# -------------------------
# Columns to read (Top-1/2/3)
# -------------------------
LONG_COLS = [
    "gpt-4o_Longitudinal_1",
    "gpt-4o_Longitudinal_2",
    "gpt-4o_Longitudinal_3",
]

LAT_COLS = [
    "gpt-4o_Lateral_1",
    "gpt-4o_Lateral_2",
    "gpt-4o_Lateral_3",
]

# -------------------------
# Helpers
# -------------------------
def map_longitudinal(action_str):
    try:
        return LongitudinalAction(action_str)
    except ValueError:
        return LongitudinalAction.UNKNOWN


def map_lateral(action_str):
    try:
        return LateralAction(action_str)
    except ValueError:
        return LateralAction.UNKNOWN


def find_scenario_xml(scenario_id):
    """
    Map ScenarioID -> XML filename
    Example: DEU_MONAEast-2_24310_T-24335.xml
    """
    filename = f"{scenario_id}.xml"
    path = os.path.join(SCENARIO_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scenario XML not found: {path}")
    return path


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    df = pd.read_csv(BATCH_FILE)

    safe_top1_cnt = 0
    safe_top3_cnt = 0
    total_cnt = 0

    for _, row in df.iterrows():
        scenario_id = row["ScenarioID"]

        xml_path = f"/home/liny/Documents/commonroad/mona-updated/{scenario_id}.xml"
        # ---- Load scenario ----
        scenario, planning_problem = extract_scenario_and_planning_problem(xml_path)
        config = SanDRAConfiguration()
        config.h = 35

        describer = CommonRoadDescriber(
            scenario,
            planning_problem,
            timestep=0,
            config=config,
            role="Don't change the lanes too often.",
            scenario_type="highway",
        )

        road_network = RoadNetwork.from_lanelet_network_and_position(
            scenario.lanelet_network,
            planning_problem.initial_state.position,
            consider_reversed=True,
            consider_incoming=True,
        )

        ego_lane_network = EgoLaneNetwork.from_route_planner(
            scenario.lanelet_network,
            planning_problem,
            road_network,
        )

        try:
            verifier = ReachVerifier(
                scenario,
                planning_problem,
                config,
                ego_lane_network=ego_lane_network,
                highenv=False,
                scenario_folder=SCENARIO_DIR,
            )
        except Exception as e:
            print(e)
            continue

        # ---- Build Top-3 action list ----
        action_pairs = []

        for i in range(3):
            long_action = map_longitudinal(row[LONG_COLS[i]])
            lat_action = map_lateral(row[LAT_COLS[i]])
            action_pairs.append((long_action, lat_action))

        # ---- Verification ----
        is_safe_top1 = False
        is_safe_top3 = False

        for idx, (long, lat) in enumerate(action_pairs):
            print('!!!', idx, long, lat)
            try:
                status = verifier.verify(actions=[long, lat])
            except Exception as e:
                print(e)
                continue

            if status == VerificationStatus.SAFE:
                if idx == 0:
                    is_safe_top1 = True
                is_safe_top3 = True
                break  # stop once SAFE found

        safe_top1_cnt += int(is_safe_top1)
        safe_top3_cnt += int(is_safe_top3)
        total_cnt += 1

    # -------------------------
    # Results
    # -------------------------
    print("=" * 50)
    print(f"Total scenarios evaluated: {total_cnt}")
    print(f"Safe@1: {safe_top1_cnt / total_cnt * 100:.2f}%")
    print(f"Safe@3: {safe_top3_cnt / total_cnt * 100:.2f}%")
