"""
Unit tests of the verifier module using reachability analysis
"""
import math
import unittest

import gymnasium
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import VehicleType
from commonroad.common.util import AngleInterval, Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import CustomState
from commonroad.scenario_definition.protobuf_format.generated_scripts.planning_problem_pb2 import PlanningProblem
from gymnasium.wrappers import RecordVideo
from matplotlib import pyplot as plt
from vehiclemodels.vehicle_parameters import VehicleParameters

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.commonroad.reach import ReachVerifier, VerificationStatus
from sandra.commonroad.plan import ReactivePlanner
from sandra.actions import LongitudinalAction, LateralAction

from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping

from sandra.highenv.highenv_scenario import HighwayEnvScenario


class TestReachVerifier(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.config = SanDRAConfiguration()

        # get vehicle parameters from CommonRoad vehicle models given cr_vehicle_id
        id_type_vehicle: int = 2
        vehicle_parameters: VehicleParameters = VehicleParameterMapping.from_vehicle_type(VehicleType(id_type_vehicle))
        delta_min: float = vehicle_parameters.steering.min
        delta_max: float = vehicle_parameters.steering.max
        # a_max: float = self.reach_ver.reach_config.vehicle.ego.a_max
        # v_max: float = self.reach_ver.reach_config.vehicle.ego.v_max
        env_config = {
            "highway-v0": {
                "observation": {
                    "type": "OccupancyGrid",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20]
                    },
                    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                    "grid_step": [5, 5],
                    "absolute": False
                },
                "action": {
                    "type": "ContinuousAction",
                    "acceleration_range": (-8.0, 8.0),
                    "steering_range": (delta_min, delta_max),
                    "speed_range": (0.0, 30.0),
                },
                "lanes_count": 4,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "duration": 30,
                "vehicles_density": 2.0,
                "show_trajectories": True,
                "render_agent": True,
                "scaling": 5,
                "initial_lane_id": None,
                "ego_spacing": 4,
            }
        }

        self.scenario = HighwayEnvScenario(env_config)
        dt = 0.2
        num_steps = 25
        self.cr_scenario, _, self.cr_planning_problem = self.scenario.commonroad_representation

        road_network = RoadNetwork.from_lanelet_network_and_position(
            self.cr_scenario.lanelet_network,
            self.cr_planning_problem.initial_state.position,
            consider_reversed=True,
            consider_incoming=True,
        )

        ego_lane_network = EgoLaneNetwork.from_route_planner(
            self.cr_scenario.lanelet_network,
            self.cr_planning_problem,
            road_network,
        )
        self.reach_ver = ReachVerifier(self.cr_scenario, self.config, ego_lane_network=ego_lane_network)

    def test_reactive_planning(self):
        self.reach_ver.verify([LongitudinalAction.DECELERATE, LateralAction.CHANGE_LEFT])

        planner = ReactivePlanner(self.config, self.cr_scenario, self.cr_planning_problem)
        planner.reset(self.reach_ver.reach_config.planning.CLCS)
        driving_corridor = self.reach_ver.reach_interface.extract_driving_corridors(
            to_goal_region=False
        )[0]
        planner.plan(driving_corridor)
        ego_prediction = planner.ego_vehicle.prediction.trajectory.state_list[1:]
        for i in range(self.config.h):
            ego_state = ego_prediction[i]
            action = ego_state.steering_angle, ego_state.acceleration
            _ = self.scenario.step(action)
        self.scenario._env.close()
        planner.visualize(
            driving_corridor=driving_corridor,
            reach_interface=self.reach_ver.reach_interface,
        )
