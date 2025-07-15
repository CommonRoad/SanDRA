import math
import os
import random
import sys
from typing import cast, Optional, List, Union

import gymnasium
import numpy as np
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from highway_env.envs.common.observation import TimeToCollisionObservation
from matplotlib import pyplot as plt

# todo
from highway_env.vehicle.behavior import IDMVehicle
IDMVehicle.LANE_CHANGE_DELAY = 2.0

from highway_env.vehicle.controller import ControlledVehicle
ControlledVehicle.TAU_LATERAL = 2.0

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
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class HighEnvDecider(Decider):
    def __init__(
        self,
        env_config: dict,
        seed: int,
        config: SanDRAConfiguration,
        save_path: str = None,
    ):
        super().__init__(config, None, None, save_path=save_path)
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

        # Initialize the past_action list
        self.past_actions: list = []

    def record_action(self, actions: List[Union[LongitudinalAction, LateralAction, None]], action_nr: int = 5) -> None:
        """Record a new action, maintaining at most 5 recent actions."""
        self.past_actions.append(actions)
        if len(self.past_actions) > action_nr:
            self.past_actions.pop(0)

    def update(self, env_config: Optional[dict]):
        if env_config is None:
            self.scenario = HighwayEnvScenario(self.scenario._env, self.seed, dt=self.config.dt, horizon=self.config.h, use_sonia=self.config.use_sonia)
        else:
            self.scenario = HighwayEnvScenario(env_config, self.seed, dt=self.config.dt, horizon=self.config.h, use_sonia=self.config.use_sonia)
        self.scenario.time_step = self.time_step
        self.time_step += 1
        cr_scenario, _, cr_planning_problem = self.scenario.commonroad_representation
        self.describer = CommonRoadDescriber(
            cr_scenario,
            cr_planning_problem,
            0,
            self.config,
            role="Don't change the lanes too often. ",
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
        with suppress_stdout():
            self.verifier = ReachVerifier(
                cr_scenario,
                cr_planning_problem,
                self.config,
                ego_lane_network=ego_lane_network,
                highenv=True
            )
        return cr_scenario, cr_planning_problem

    def run(self):
        if self.config.highway_env.action_input:
            done = truncated = False
            # plt.imshow(self.scenario._env.render())
            # plt.show()
            while not (done or truncated):
                self.time_step += 1
                if self.time_step > self.config.highway_env.policy_frequency * self.config.highway_env.duration + 1:
                    break
                if self.scenario:
                    self.update(self.scenario._env)
                else:
                    self.update(None)
                longitudinal_action, lateral_action = self.decide(self.past_actions)

                # record the actions
                self.record_action([longitudinal_action, lateral_action])

                if lateral_action in [
                    LateralAction.CHANGE_RIGHT,
                    LateralAction.CHANGE_LEFT,
                ]:
                    action = self.lateral_action_to_id[lateral_action]
                else:
                    action = self.longitudinal_action_to_id[longitudinal_action]
                obs, reward, done, truncated, info = self.scenario._env.step(action)
                # self.describer.update_with_observation(obs)
                self.scenario._env.render()
                # plt.imshow(self.scenario._env.render())
                # plt.show()
            self.scenario._env.close()
        else:
            def normalize(v, a, b):
                normalized_v = (v - a) / (b - a)
                return 2 * normalized_v - 1

            input_bounds = get_input_bounds()
            replanning_frequency = 5
            simulation_length = 30 * 5
            current_ego_prediction = None
            for i in range(simulation_length):
                print(f"STEP ID: {self.scenario._env.step_id}")
                if i % replanning_frequency == 0:
                    cr_scenario, cr_planning_problem = self.update(None)
                    user_prompt = self.describer.user_prompt()
                    system_prompt = self.describer.system_prompt()
                    schema = self.describer.schema()
                    structured_response = get_structured_response(
                        user_prompt, system_prompt, schema, self.config, save_dir=self.save_path
                    )
                    ranking = self._parse_action_ranking(structured_response)
                    found_viable_action = False
                    for action in ranking:
                        if self.verifier.verify(list(action), safe_distance=True) == VerificationStatus.SAFE:
                            try:
                                with suppress_stdout():
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
                        if self.verifier.verify([None], safe_distance=False) == VerificationStatus.SAFE:
                            try:
                                # Prevent a bug in the planner where it deletes slip_angle attribute a second time
                                cr_planning_problem.initial_state.slip_angle = 0
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
                print(f" Actions {action_first}, {action_second}")
                action = action_second, action_first
                _ = self.scenario.step(action)
            self.scenario._env.close()

    @staticmethod
    def configure(config: SanDRAConfiguration = None, save_path = None) -> "HighEnvDecider":
        if config is None:
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
            config = SanDRAConfiguration()
        else:
            seeds = config.highway_env.seeds
        input_bounds = get_input_bounds()

        if config.highway_env.action_input:
            action_dict = {
                    "type": "DiscreteMetaAction",
                    "target_speeds": np.linspace(5, 32, 9),
                }
        else:
            action_dict = {
                    "type": "ContinuousAction",
                    "acceleration_range": (input_bounds["a_min"], input_bounds["a_max"]),
                    "steering_range": (input_bounds["delta_min"], input_bounds["delta_max"]),
                    "speed_range": (input_bounds["v_min"], input_bounds["v_max"]),
                }
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
                        "vy": [-4, 4]
                    },
                    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                    "grid_step": [5, 5],
                    "absolute": False
                },
                "action": action_dict,
                "lanes_count": config.highway_env.lanes_count,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "duration": config.highway_env.duration,
                "vehicles_density": config.highway_env.vehicles_density,
                "show_trajectories": True,
                "render_agent": True,
                "scaling": 5,
                "initial_lane_id": None,
                "ego_spacing": 4,
                "simulation_frequency": config.highway_env.simulation_frequency,
                "policy_frequency": config.highway_env.policy_frequency,
            }
        }
        seed = random.choice(seeds)
        return HighEnvDecider(env_config, seed, config, save_path=save_path)


if __name__ == "__main__":
    config = SanDRAConfiguration()
    decider = HighEnvDecider.configure(config)
    decider.run()
