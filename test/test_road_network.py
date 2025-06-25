"""
Unit tests of the road network
"""

import unittest

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.common.road_network import RoadNetwork, Lane, EgoLaneNetwork
from sandra.utility.visualization import plot_road_network


class TestReachVerifier(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        name_scenario = "DEU_AachenHeckstrasse-1_30520_T-539"
        path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + ".xml"
        self.scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
            lanelet_assignment=True
        )
        self.planning_problem = list(
            planning_problem_set.planning_problem_dict.values()
        )[0]

        self.config = SanDRAConfiguration()

    def test_highD(self):
        name_scenario = "DEU_LocationALower-11_10_T-1"
        path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + ".xml"
        self.scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
            lanelet_assignment=True
        )
        self.planning_problem = list(
            planning_problem_set.planning_problem_dict.values()
        )[0]

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

        plot_road_network(road_network, ego_lane_network)

        assert len(road_network.lanes) == 2

    def test_ind_bendplatz(self):
        name_scenario = "DEU_AachenBendplatz-1_80_T-19"
        path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + ".xml"
        self.scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
            lanelet_assignment=True
        )
        self.planning_problem = list(
            planning_problem_set.planning_problem_dict.values()
        )[0]

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

        plot_road_network(road_network, ego_lane_network)

        assert len(road_network.lanes) == 6

    def test_lane_construction(self):

        ego_lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position(
            [self.planning_problem.initial_state.position]
        )[0][0]

        ego_lane = Lane(0)

        for _ in range(2):
            if ego_lanelet_id:
                ego_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(
                    ego_lanelet_id
                )
                ego_lane.add_lanelet(ego_lanelet)
                ego_lanelet_id = ego_lanelet.successor[0]

        assert ego_lane.contained_ids == [128, 155]

    def test_road_network(self):

        road_network_1 = RoadNetwork.from_lanelet_network_and_position(
            self.scenario.lanelet_network,
            self.planning_problem.initial_state.position,
            consider_reversed=True,
        )

        plot_road_network(road_network_1)

        assert len(road_network_1.lanes) == 4

        road_network_2 = RoadNetwork.from_lanelet_network_and_position(
            self.scenario.lanelet_network,
            np.asarray([19.80, -10.09]),  # on lanelet 103
            consider_reversed=True,
        )

        assert len(road_network_2.lanes) == 5

    def test_ego_lane_network(self):
        road_network_1 = RoadNetwork.from_lanelet_network_and_position(
            self.scenario.lanelet_network,
            self.planning_problem.initial_state.position,
            consider_reversed=True,
        )

        ego_lane_network_1 = EgoLaneNetwork.from_route_planner(
            self.scenario.lanelet_network,
            self.planning_problem,
            road_network_1,
        )

        plot_road_network(road_network_1, ego_lane_network_1)

        assert ego_lane_network_1.lane_right_adjacent is None
        assert ego_lane_network_1.lane_left_adjacent is None
        assert isinstance(ego_lane_network_1.lane.center_vertices, np.ndarray)

        self.planning_problem.initial_state.position = np.asarray([19.80, -10.09])
        self.planning_problem.goal.state_list[0].position.center = np.asarray(
            [51.80, -38.22]
        )

        road_network_2 = RoadNetwork.from_lanelet_network_and_position(
            self.scenario.lanelet_network,
            self.planning_problem.initial_state.position,  # on lanelet 103
            consider_reversed=True,
        )

        ego_lane_network_2 = EgoLaneNetwork.from_route_planner(
            self.scenario.lanelet_network,
            self.planning_problem,
            road_network_2,
        )

        plot_road_network(road_network_2, ego_lane_network_2)

        assert (
            ego_lane_network_2.lane_right_adjacent is not None
            and len(ego_lane_network_2.lane_right_adjacent) == 2
        )
        assert ego_lane_network_2.lane_left_adjacent is None
