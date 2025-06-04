"""
Unit tests of the verifier module using reachability analysis
"""

import unittest


from commonroad.common.file_reader import CommonRoadFileReader

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.verifier.reach import ReachVerifier, VerificationStatus
from sandra.actions import LongitudinalAction, LateralAction


class TestReachVerifier(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        name_scenario = "DEU_Gar-1_1_T-1"
        path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + '.xml'
        self.scenario, _ = CommonRoadFileReader(path_scenario).open(
            lanelet_assignment=True
        )
        self.config = SanDRAConfiguration()

        self.reach_ver = ReachVerifier(self.scenario,
                                       self.config)

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

