"""
Unit tests of labelling functions
"""

import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader

from sandra.actions import LongitudinalAction, LateralAction
from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.labeler import TrajectoryLabeler, ReachSetLabeler
from sandra.utility.vehicle import extract_ego_vehicle


class TestLabeler(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SanDRAConfiguration()
        self.config.h = 20  # for highd scenarios

    def _load_scenario_and_ego_vehicle(self, scenario_name: str):
        """Helper to load scenario and ego vehicle"""
        path_scenario = Path(PROJECT_ROOT) / "scenarios" / f"{scenario_name}.xml"
        scenario, planning_problem_set = CommonRoadFileReader(str(path_scenario)).open(
            lanelet_assignment=True
        )
        planning_problem = next(
            iter(planning_problem_set.planning_problem_dict.values())
        )
        ego_vehicle = extract_ego_vehicle(scenario, planning_problem)
        print(f"Obtained ego vehicle: {ego_vehicle.obstacle_id}")
        return scenario, planning_problem, ego_vehicle

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

    def test_traj_label_follow_lane_accelerate(self):
        scenario_name = "DEU_LocationALower-11_10_T-1"
        scenario, planning_problem, ego_vehicle = self._load_scenario_and_ego_vehicle(
            scenario_name
        )

        ego_lane_network = self._build_ego_lane_network(scenario, planning_problem)

        labeler = TrajectoryLabeler(self.config, scenario)
        actions = labeler.label(ego_vehicle, ego_lane_network)

        self.assertSetEqual(
            set(actions[0]),
            {LateralAction.FOLLOW_LANE, LongitudinalAction.ACCELERATE},
        )

    def test_traj_label_change_left_accelerate(self):
        scenario_name = "DEU_LocationALower-36_263_T-15"
        scenario, planning_problem, ego_vehicle = self._load_scenario_and_ego_vehicle(
            scenario_name
        )

        ego_lane_network = self._build_ego_lane_network(scenario, planning_problem)

        labeler = TrajectoryLabeler(self.config, scenario)
        actions = labeler.label(ego_vehicle, ego_lane_network)

        self.assertSetEqual(
            set(actions[0]),
            {LateralAction.CHANGE_LEFT, LongitudinalAction.ACCELERATE},
        )

    def test_reach_label(self):
        scenario_name = "DEU_LocationALower-11_10_T-1"

        scenario, planning_problem, ego_vehicle = self._load_scenario_and_ego_vehicle(
            scenario_name
        )

        ego_lane_network = self._build_ego_lane_network(scenario, planning_problem)

        labeler = ReachSetLabeler(self.config, scenario, planning_problem)
        actions = labeler.label(ego_vehicle, ego_lane_network)

        expected = [
            {LateralAction.CHANGE_RIGHT, LongitudinalAction.DECELERATE},
            {LateralAction.CHANGE_RIGHT, LongitudinalAction.ACCELERATE},
            {LateralAction.FOLLOW_LANE, LongitudinalAction.DECELERATE},
        ]

        assert set(frozenset(a) for a in actions) == set(
            frozenset(a) for a in expected
        ), "Actions do not match expected"


if __name__ == "__main__":
    unittest.main()
