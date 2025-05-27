import os.path

from config import SaLaRAConfiguration, PROJECT_ROOT
from src.decider import Decider

from src.utils import extract_scenario_and_planning_problem, plot_scenario, plot_lanelet, extract_ego_vehicle, find_lanelet_id_from_state
from src.lanelet_network import EgoCenteredLaneletNetwork


def main():
    config = SaLaRAConfiguration()
    scenario_path = ""
    save_path = scenario_path
    decider = Decider(scenario_path, config, save_path=save_path)


if __name__ == "__main__":
    scenario_paths = [
        "DEU_AachenAseag-1_80_T-99.xml",
        "DEU_AachenBendplatz-1_80_T-19.xml",
        "DEU_AachenFrankenburg-1_2120_T-39.xml",
        "DEU_AachenHeckstrasse-1_30520_T-539.xml",
        "DEU_LocationALower-11_10_T-1.xml"
    ]

    scenario_folder = os.path.join(PROJECT_ROOT, "scenarios")

    scenario, planning_problem = extract_scenario_and_planning_problem(os.path.join(scenario_folder, scenario_paths[-1]))
    ego_lane_id = find_lanelet_id_from_state(planning_problem.initial_state, scenario.lanelet_network)
    lanelet_network = EgoCenteredLaneletNetwork(scenario.lanelet_network, ego_lane_id)
    print()

        # lanelets = scenario.lanelet_network.lanelets
        # for lanelet in lanelets:
        #     if lanelet.successor is not None and len(lanelet.successor) > 1:
        #         plot_lanelet(lanelet, scenario.lanelet_network)
