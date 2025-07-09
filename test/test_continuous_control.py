"""
Unit tests to verify that control inputs into highway-env still work
"""
import unittest

from vehiclemodels.vehicle_parameters import VehicleParameters
from commonroad.common.solution import VehicleType
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping

from sandra.actions import LateralAction
from sandra.commonroad.plan import ReactivePlanner
from sandra.commonroad.reach import ReachVerifier
from sandra.common.config import SanDRAConfiguration
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.highenv.highenv_scenario import HighwayEnvScenario


class TestReachVerifier(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.config = SanDRAConfiguration()

        # get vehicle parameters from CommonRoad vehicle models given cr_vehicle_id
        id_type_vehicle: int = 2
        vehicle_parameters: VehicleParameters = VehicleParameterMapping.from_vehicle_type(VehicleType(id_type_vehicle))
        self.delta_min = vehicle_parameters.steering.min
        self.delta_max = vehicle_parameters.steering.max
        self.a_max = 8.0
        self.a_min = -self.a_max
        self.v_min = 0.0
        self.v_max = 30.0
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
                    "acceleration_range": (self.a_min, self.a_max),
                    "steering_range": (self.delta_min, self.delta_max),
                    "speed_range": (self.v_min, self.v_max),
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
                "simulation_frequency": 15,
                "policy_frequency": 15,
            }
        }

        self.scenario = HighwayEnvScenario(env_config, seed=4213)
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
        self.reach_ver = ReachVerifier(self.cr_scenario, self.config, ego_lane_network=ego_lane_network,
                                  initial_state=self.cr_planning_problem.initial_state)

    def test_reactive_planning(self):
        simulation_length = 60
        replanning_frequency = 5
        current_ego_prediction = None

        def normalize(v, a, b):
            normalized_v = (v - a) / (b - a)
            return 2 * normalized_v - 1

        def update():
            self.scenario = HighwayEnvScenario(self.scenario._env, seed=4213)
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
            self.reach_ver = ReachVerifier(self.cr_scenario, self.config, ego_lane_network=ego_lane_network,
                                      initial_state=self.cr_planning_problem.initial_state)

        for i in range(simulation_length):
            if i % replanning_frequency == 0:
                if i > 0:
                    update()
                self.reach_ver.verify([LateralAction.CHANGE_LEFT])
                planner = ReactivePlanner(self.config, self.cr_scenario, self.cr_planning_problem)
                planner.reset(self.reach_ver.reach_config.planning.CLCS)
                driving_corridor = self.reach_ver.reach_interface.extract_driving_corridors(
                    to_goal_region=False
                )[0]
                planner.plan(driving_corridor)
                current_ego_prediction = planner.ego_vehicle.prediction.trajectory.state_list[1:]

            ego_state = current_ego_prediction[i % replanning_frequency]
            action_first = -normalize(ego_state.steering_angle, self.delta_min, self.delta_max)
            action_second = normalize(ego_state.acceleration, self.a_min, self.a_max)
            action = action_second, action_first
            _ = self.scenario.step(action)
        self.scenario._env.close()