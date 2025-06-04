"""
Unit tests of the road network
"""

import unittest

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.common.road_network import RoadNetwork, Lane

class TestReachVerifier(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        name_scenario = "DEU_AachenHeckstrasse-1_30520_T-539"
        path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + '.xml'
        self.scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
            lanelet_assignment=True
        )
        self.planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

        self.config = SanDRAConfiguration()

    def test_lane_construction(self):

        ego_lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position(
            [self.planning_problem.initial_state.position]
        )[0][0]

        ego_lane = Lane(0)

        for _ in range(2):
            if ego_lanelet_id:
                ego_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
                ego_lane.add_lanelet(ego_lanelet)
                ego_lanelet_id = ego_lanelet.successor[0]

        assert ego_lane.contained_ids == [128, 155]

    def test_road_network(self):

        road_network_1 = RoadNetwork.from_lanelet_network_and_position(
            self.scenario.lanelet_network,
            self.planning_problem.initial_state.position,
            consider_reversed=True
        )

        assert len(road_network_1.lanes) == 4

        road_network_2 = RoadNetwork.from_lanelet_network_and_position(
            self.scenario.lanelet_network,
            np.asarray([19.80, -10.09]), # on lanelet 103
            consider_reversed=True
        )

        assert len(road_network_2.lanes) == 5