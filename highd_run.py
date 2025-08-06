"""
Standalone script to create HighEnvDecider and run it.
"""

import os

from commonroad.common.file_reader import CommonRoadFileReader

from config.sandra import SanDRAConfiguration
from sandra.utility.road_network import RoadNetwork, EgoLaneNetwork
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.commonroad.reach import ReachVerifier
from sandra.decider import Decider
import matplotlib

print(matplotlib.get_backend())
matplotlib.use("TkAgg")


def main():
    config = SanDRAConfiguration()
    path_to_scenarios = "/home/sebastian/Documents/Uni/Sandra/mona_scenarios/"
    path_to_results = f"results-highD-{config.model_name}/"
    results = [x[:-4] for x in os.listdir(path_to_results) if x.endswith(".csv")]

    for filename in os.listdir(path_to_scenarios):
        if not filename.endswith(".xml"):
            continue
        if filename[:-4] in results:
            continue
        scenario, planning_problem_set = CommonRoadFileReader(
            path_to_scenarios + filename
        ).open(lanelet_assignment=True)
        planning_problem = next(
            iter(planning_problem_set.planning_problem_dict.values())
        )
        describer = CommonRoadDescriber(
            scenario,
            planning_problem,
            0,
            config,
            role="Don't change the lanes too often. ",
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
                highenv=True,
                scenario_folder=path_to_scenarios,
            )
        except FileNotFoundError:
            scenario.scenario_id = filename[:-4]
            verifier = ReachVerifier(
                scenario,
                planning_problem,
                config,
                ego_lane_network=ego_lane_network,
                highenv=True,
                scenario_folder=path_to_scenarios,
            )
        decider = Decider(
            config,
            describer,
            verifier=verifier,
            save_path=f"results-highD-{config.model_name}/{scenario.scenario_id}.csv",
        )
        decider.decide()


if __name__ == "__main__":
    main()
