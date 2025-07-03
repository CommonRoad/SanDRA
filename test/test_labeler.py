"""
Unit tests of labelling functions
"""

import unittest

from commonroad.common.file_reader import CommonRoadFileReader

from sandra.actions import LongitudinalAction, LateralAction
from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.common.road_network import RoadNetwork, Lane, EgoLaneNetwork
from sandra.labeler import HighDLabeler
from sandra.utility.vehicle import extract_ego_vehicle


class TestLabeler(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        name_scenario = "DEU_LocationALower-11_10_T-1"
        path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + ".xml"
        self.scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
            lanelet_assignment=True
        )
        self.planning_problem = list(
            planning_problem_set.planning_problem_dict.values()
        )[0]

        self.config = SanDRAConfiguration()

        self.ego_vehicle = extract_ego_vehicle(self.scenario, self.planning_problem)
        print(f"Extracted ego vehicle: {self.ego_vehicle.obstacle_id}")

    def test_highd_label(self):
        road_network = RoadNetwork.from_lanelet_network_and_position(
            self.scenario.lanelet_network,
            self.planning_problem.initial_state.position,
            consider_reversed=True,
        )

        ego_lane_network = EgoLaneNetwork.from_route_planner(
            self.scenario.lanelet_network,
            self.planning_problem,
            road_network,
        )

        self.labeler = HighDLabeler(self.config, self.scenario)

        actions = self.labeler.label(self.ego_vehicle, ego_lane_network)

        assert set(actions) == {
            LateralAction.FOLLOW_LANE,
            LongitudinalAction.ACCELERATE,
        }
