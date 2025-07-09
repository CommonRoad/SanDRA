"""
Unit tests of the verifier module using reachability analysis
"""

import unittest


from commonroad.common.file_reader import CommonRoadFileReader

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.commonroad.reach import ReachVerifier, VerificationStatus
from sandra.commonroad.plan import ReactivePlanner
from sandra.actions import LongitudinalAction, LateralAction

import matplotlib
print(matplotlib.get_backend())
matplotlib.use('TkAgg')


class TestReachVerifier(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        name_scenario = "DEU_Gar-1_1_T-1"
        path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + ".xml"
        self.scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
            lanelet_assignment=True
        )
        self.planning_problem = list(
            planning_problem_set.planning_problem_dict.values()
        )[0]

        self.config = SanDRAConfiguration()

        self.reach_ver = ReachVerifier(self.scenario, self.planning_problem, self.config)

    def _build_ego_lane_network(self, scenario, planning_problem):
        """Helper to build EgoLaneNetwork"""
        road_network = RoadNetwork.from_lanelet_network_and_position(
            scenario.lanelet_network,
            planning_problem.initial_state.position,
            consider_reversed=True,
        )
        return EgoLaneNetwork.from_route_planner(
            scenario.lanelet_network,
            planning_problem,
            road_network,
        )

    def test_action_ltl(self):
        # acceleration
        acc = self.reach_ver.parse_action(LongitudinalAction.ACCELERATE)
        assert acc == ""

        ego_road_network = self._build_ego_lane_network(
            self.scenario, self.planning_problem
        )
        self.reach_ver.reset(ego_lane_network=ego_road_network)
        # right lane change (left impossible)
        right = self.reach_ver.parse_action(LateralAction.CHANGE_RIGHT)
        assert right == self.reach_ver.parse_action(LateralAction.CHANGE_RIGHT)

    def test_verification(self):
        status = self.reach_ver.verify([LongitudinalAction.STOP])
        assert status == VerificationStatus.SAFE

    def test_reactive_planning(self):
        self.reach_ver.verify([LongitudinalAction.STOP])

        planner = ReactivePlanner(self.config, self.scenario, self.planning_problem)
        planner.reset(self.reach_ver.reach_config.planning.CLCS)
        driving_corridor = self.reach_ver.reach_interface.extract_driving_corridors(
            to_goal_region=False
        )[0]
        planner.plan(driving_corridor)

        planner.visualize(
            driving_corridor=driving_corridor,
            reach_interface=self.reach_ver.reach_interface,
        )
