"""
Unit tests of the verifier module using reachability analysis
"""

import unittest


from commonroad.common.file_reader import CommonRoadFileReader

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.commonroad.reach import ReachVerifier, VerificationStatus
from sandra.commonroad.plan import ReactivePlanner
from sandra.actions import LongitudinalAction, LateralAction


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
        self.config.h = 20

        self.reach_ver = ReachVerifier(self.scenario, self.config)

    def test_action_ltl(self):
        # acceleration
        acc = self.reach_ver.parse_action(LongitudinalAction.ACCELERATE)
        assert acc == ""

        # left lane change
        left = self.reach_ver.parse_action(LateralAction.CHANGE_LEFT)
        assert left == self.reach_ver.parse_action(LateralAction.CHANGE_LEFT)

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

    def test_verification_sonia(self):
        status = self.reach_ver.verify_sonia([LongitudinalAction.STOP], visualization=True)
        assert status == VerificationStatus.SAFE

