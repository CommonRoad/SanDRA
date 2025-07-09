import math
import random
from typing import cast, Optional

import gymnasium
import numpy as np
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from highway_env.envs.common.observation import TimeToCollisionObservation
from matplotlib import pyplot as plt

from sandra.actions import LateralAction, LongitudinalAction
from sandra.common.config import SanDRAConfiguration
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.commonroad.plan import ReactivePlanner
from sandra.commonroad.reach import ReachVerifier
from sandra.decider import Decider
from sandra.highenv.describer import HighEnvDescriber
from sandra.highenv.highenv_scenario import HighwayEnvScenario
from sandra.llm import get_structured_response
from sandra.utility.vehicle import get_input_bounds
from sandra.verifier import VerificationStatus


class HighEnvDecider(Decider):
    def __init__(
        self,
        env_config: dict,
        seed: int,
        config: SanDRAConfiguration,
    ):
        super().__init__(config, None, None)
        self.lateral_action_to_id: dict[LateralAction, int] = {
            LateralAction.CHANGE_LEFT: 0,
            LateralAction.FOLLOW_LANE: 1,
            LateralAction.CHANGE_RIGHT: 2,
        }
        self.longitudinal_action_to_id: dict[LongitudinalAction, int] = {
            LongitudinalAction.KEEP: 1,
            LongitudinalAction.ACCELERATE: 3,
            LongitudinalAction.DECELERATE: 4,
        }
        self.seed = seed
        self.update(env_config)

    def update(self, env_config: Optional[dict]):
        if env_config is None:
            self.scenario = HighwayEnvScenario(self.scenario._env, self.seed, dt=self.config.dt)
        else:
            self.scenario = HighwayEnvScenario(env_config, self.seed, dt=self.config.dt)
        cr_scenario, _, cr_planning_problem = self.scenario.commonroad_representation
        self.describer = CommonRoadDescriber(
            cr_scenario,
            cr_planning_problem,
            0,
            self.config,
            scenario_type="highway",
        )
        road_network = RoadNetwork.from_lanelet_network_and_position(
            cr_scenario.lanelet_network,
            cr_planning_problem.initial_state.position,
            consider_reversed=True,
            consider_incoming=True,
        )
        ego_lane_network = EgoLaneNetwork.from_route_planner(
            cr_scenario.lanelet_network,
            cr_planning_problem,
            road_network,
        )
        self.verifier = ReachVerifier(cr_scenario, self.config, ego_lane_network=ego_lane_network,
                                      initial_state=cr_planning_problem.initial_state)
        return cr_scenario, cr_planning_problem

    def run(self):
        def normalize(v, a, b):
            normalized_v = (v - a) / (b - a)
            return 2 * normalized_v - 1

        input_bounds = get_input_bounds()
        simulation_length = 60
        replanning_frequency = 5
        current_ego_prediction = None
        cr_scenario, cr_planning_problem = None, None
        for i in range(simulation_length):
            if i % replanning_frequency == 0:
                cr_scenario, cr_planning_problem = self.update(None)
                user_prompt = self.describer.user_prompt()
                system_prompt = self.describer.system_prompt()
                schema = self.describer.schema()
                structured_response = get_structured_response(
                    user_prompt, system_prompt, schema, self.config, save_dir=self.save_path
                )
                ranking = self._parse_action_ranking(structured_response)
                ranking = []
                found_viable_action = False
                for action in ranking:
                    if self.verifier.verify(list(action), safe_distance=True) == VerificationStatus.SAFE:
                        try:
                            planner = ReactivePlanner(self.config, cr_scenario, cr_planning_problem)
                            planner.reset(self.verifier.reach_config.planning.CLCS)
                            driving_corridor = self.verifier.reach_interface.extract_driving_corridors(
                                to_goal_region=False
                            )[0]
                            planner.plan(driving_corridor)
                            current_ego_prediction = planner.ego_vehicle.prediction.trajectory.state_list[1:]
                            found_viable_action = True
                            break
                        except Exception as e:
                            print(f"Planning failed: {e}")

                if not found_viable_action:
                    if self.verifier.verify([None], safe_distance=True) == VerificationStatus.SAFE:
                        try:
                            planner = ReactivePlanner(self.config, cr_scenario, cr_planning_problem)
                            planner.reset(self.verifier.reach_config.planning.CLCS)
                            driving_corridor = self.verifier.reach_interface.extract_driving_corridors(
                                to_goal_region=False
                            )[0]
                            planner.plan(driving_corridor)
                            current_ego_prediction = planner.ego_vehicle.prediction.trajectory.state_list[1:]
                        except Exception as e:
                            raise RuntimeError(f"Planning failed: {e}")
                    else:
                        raise RuntimeError("Verification failed")

            ego_state = current_ego_prediction[i % replanning_frequency]
            action_first = -normalize(ego_state.steering_angle, input_bounds["delta_min"], input_bounds["delta_max"])
            action_second = normalize(ego_state.acceleration, input_bounds["a_min"], input_bounds["a_max"])
            action = action_second, action_first
            _ = self.scenario.step(action)
        self.scenario._env.close()

    @staticmethod
    def configure(seeds: list[int] = None) -> "HighEnvDecider":
        if seeds is None:
            seeds = [
                5838,
                2421,
                7294,
                9650,
                4176,
                6382,
                8765,
                1348,
                4213,
                2572,
                5678,
                8587,
                512,
                7523,
                6321,
                5214,
                31,
            ]
        input_bounds = get_input_bounds()
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
                    "acceleration_range": (input_bounds["a_min"], input_bounds["a_max"]),
                    "steering_range": (input_bounds["delta_min"], input_bounds["delta_max"]),
                    "speed_range": (input_bounds["v_min"], input_bounds["v_max"]),
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
        seed = random.choice(seeds)
        return HighEnvDecider(env_config, seed, SanDRAConfiguration())


if __name__ == "__main__":
    decider = HighEnvDecider.configure([4213])
    decider.run()
