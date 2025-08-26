import math
import os
import random
import sys
from pathlib import Path
from typing import cast, Optional, List, Union

import gymnasium
import numpy as np
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario, Tag
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
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
from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
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
        # entire commonroad scenario
        self.cr_scenario_whole, self.cr_ego_whole, self.planning_problem = self.update(env_config)
        for obs in self.cr_scenario_whole.dynamic_obstacles:
            obs.prediction = None
        self.cr_ego_whole.prediction = None

        # Initialize the past_action list
        self.past_actions: list = []

        # record the initial position
        self.initial_position = self.planning_problem.initial_state.position

    def record_action(
        self,
        actions: List[Union[LongitudinalAction, LateralAction, None]],
        action_nr: int = 5,
    ) -> None:
        """Record a new action, maintaining at most 5 recent actions."""
        self.past_actions.append(actions)
        if len(self.past_actions) > action_nr:
            self.past_actions.pop(0)

    def update(self, env_config: Optional[dict]):
        if env_config is None:
            self.scenario = HighwayEnvScenario(
                self.scenario._env,
                self.seed,
                dt=self.config.dt,
                horizon=self.config.h,
                use_sonia=self.config.use_sonia,
                maximum_lanelet_length=self.config.highway_env.maximum_lanelet_length,
                video_folder=f"run-{self.config.model_name}-{self.config.highway_env.lanes_count}-{self.config.highway_env.vehicles_density}"
            )
        else:
            self.scenario = HighwayEnvScenario(
                env_config,
                self.seed,
                dt=self.config.dt,
                horizon=self.config.h,
                use_sonia=self.config.use_sonia,
                maximum_lanelet_length=self.config.highway_env.maximum_lanelet_length,
                video_folder=f"run-{self.config.model_name}-{self.config.highway_env.lanes_count}-{self.config.highway_env.vehicles_density}"
            )
        self.scenario.time_step = self.time_step
        cr_scenario, cr_ego_vehicle, cr_planning_problem = self.scenario.commonroad_representation
        self.describer = CommonRoadDescriber(
            cr_scenario,
            cr_planning_problem,
            0,
            self.config,
            role="Don't change the lanes too often. ",
            scenario_type="highway",
            highway_env=True,
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
                highenv=True,
            )
        return cr_scenario, cr_ego_vehicle, cr_planning_problem

    def record(self, scenario: Scenario, ego_vehicle: DynamicObstacle) -> None:
        """Append the current states of obstacles and ego to their trajectories."""

        def append_from_initial(cr_obs, new_initial, time_step):
            """Append the given initial state as the next trajectory state."""
            new_state = CustomState(
                position=new_initial.position,
                velocity=new_initial.velocity,
                orientation=new_initial.orientation,
                acceleration=new_initial.acceleration,
                time_step=time_step,
            )
            if cr_obs.prediction is None:
                trajectory = Trajectory(1, [new_state])
                cr_obs.prediction = TrajectoryPrediction(trajectory, cr_obs.obstacle_shape)
            else:
                cr_obs.prediction.trajectory.append_state(new_state)

        # Update dynamic obstacles
        for obs in scenario.dynamic_obstacles:
            cr_obs = self.cr_scenario_whole.obstacle_by_id(obs.obstacle_id)
            if cr_obs is not None:
                append_from_initial(cr_obs, obs.initial_state, self.time_step + 1)

        # Update ego vehicle
        append_from_initial(self.cr_ego_whole, ego_vehicle.initial_state, self.time_step + 1)

    def run(self):
        if self.config.highway_env.action_input:
            done = truncated = False
            svg_save_folder = self.config.highway_env.get_save_folder(
                self.config.model_name, self.seed, self.config.use_sonia
            )
            os.makedirs(svg_save_folder, exist_ok=True)
            while not (done or truncated):
                print(f"***** Simulation frame {self.time_step}: ...")
                if (
                    self.time_step
                    > self.config.highway_env.duration  # self.config.highway_env.policy_frequency *
                ):
                    break
                if self.scenario:
                    cr_scenario, cr_ego, _ = self.update(self.scenario._env)
                else:
                    cr_scenario, cr_ego, _ = self.update(None)

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

                self.record(cr_scenario, cr_ego)

                # self.describer.update_with_observation(obs)
                # Render frame
                frame = self.scenario._env.render()  # RGB NumPy array
                if self.config.highway_env.save_frame:
                    # Save as SVG
                    fig, ax = plt.subplots()
                    ax.imshow(frame)
                    ax.axis("off")
                    ax.set_aspect("equal")
                    fig.savefig(
                        os.path.join(
                            svg_save_folder, f"frame_{self.time_step:04d}.svg"
                        ),
                        format="svg",
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    plt.close(fig)
                if done:
                    print(
                        "[red]Simulation crash after running steps: [/red] ",
                        self.time_step,
                    )
                    break
                if truncated:
                    print(
                        "[red]The agent reaches the terminal state: [/red]",
                        self.time_step,
                    )
                    break
                # only add the time step after the simulation
                self.time_step += 1
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
                    cr_scenario, cr_ego, cr_planning_problem = self.update(None)
                    user_prompt = self.describer.user_prompt()
                    system_prompt = self.describer.system_prompt()
                    schema = self.describer.schema()
                    structured_response = get_structured_response(
                        user_prompt,
                        system_prompt,
                        schema,
                        self.config,
                        save_dir=self.save_path,
                    )
                    ranking = self._parse_action_ranking(structured_response)
                    found_viable_action = False
                    for action in ranking:
                        if (
                            self.verifier.verify(list(action))
                            == VerificationStatus.SAFE
                        ):
                            try:
                                with suppress_stdout():
                                    planner = ReactivePlanner(
                                        self.config, cr_scenario, cr_planning_problem
                                    )
                                    planner.reset(
                                        self.verifier.reach_config.planning.CLCS
                                    )
                                    driving_corridor = self.verifier.reach_interface.extract_driving_corridors(
                                        to_goal_region=False
                                    )[
                                        0
                                    ]
                                    planner.plan(driving_corridor)
                                current_ego_prediction = planner.ego_vehicle.prediction.trajectory.state_list[
                                    1:
                                ]
                                found_viable_action = True
                                break
                            except Exception as e:
                                print(f"Planning failed: {e}")

                    if not found_viable_action:
                        if (
                            self.verifier.verify([None])
                            == VerificationStatus.SAFE
                        ):
                            try:
                                # Prevent a bug in the planner where it deletes slip_angle attribute a second time
                                cr_planning_problem.initial_state.slip_angle = 0
                                planner = ReactivePlanner(
                                    self.config, cr_scenario, cr_planning_problem
                                )
                                planner.reset(self.verifier.reach_config.planning.CLCS)
                                driving_corridor = self.verifier.reach_interface.extract_driving_corridors(
                                    to_goal_region=False
                                )[
                                    0
                                ]
                                planner.plan(driving_corridor)
                                current_ego_prediction = planner.ego_vehicle.prediction.trajectory.state_list[
                                    1:
                                ]
                            except Exception as e:
                                raise RuntimeError(f"Planning failed: {e}")
                        else:
                            raise RuntimeError("Verification failed")

                ego_state = current_ego_prediction[i % replanning_frequency]
                action_first = -normalize(
                    ego_state.steering_angle,
                    input_bounds["delta_min"],
                    input_bounds["delta_max"],
                )
                action_second = normalize(
                    ego_state.acceleration, input_bounds["a_min"], input_bounds["a_max"]
                )
                print(f" Actions {action_first}, {action_second}")
                action = action_second, action_first
                _ = self.scenario.step(action)
                self.scenario._env.close()

        # store the travelled distance
        _, _, planning_problem = self.update(None)
        travelled_distance = (
            planning_problem.initial_state.position[0] - self.initial_position[0]
        )
        new_row = {"iteration-id": "Travelled", "Lateral1": travelled_distance}
        self.save_iteration(new_row)

        # save the scenario for monitoring
        self.save_scenario_whole()


    def save_scenario_whole(self):
        author = "Sandra Lin"
        affiliation = "Technical University of Munich, Germany"
        source = ""
        tags = {Tag.HIGHWAY}
        self.cr_scenario_whole.add_objects(self.cr_ego_whole)

        planning_problem_set = PlanningProblemSet([self.planning_problem])
        fw = CommonRoadFileWriter(
            self.cr_scenario_whole, planning_problem_set, author, affiliation, source, tags
        )
        self.cr_scenario_whole.scenario_id = str(self.cr_scenario_whole.scenario_id) + "001"
        # build path
        path_dir = (
                PROJECT_ROOT
                + f"/scenarios_monitoring_{self.config.highway_env.lanes_count}-"
                  f"{self.config.highway_env.vehicles_density}_reach_rule_{self.config.use_rules_in_reach}/"
        )
        path_scenario = os.path.join(path_dir, str(self.cr_scenario_whole.scenario_id) + ".xml")

        # ensure directory exists
        Path(path_dir).mkdir(parents=True, exist_ok=True)

        fw.write_to_file(path_scenario, OverwriteExistingFile.ALWAYS)

    @staticmethod
    def configure(
        config: SanDRAConfiguration = None, save_path=None
    ) -> "HighEnvDecider":
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
                "steering_range": (
                    input_bounds["delta_min"],
                    input_bounds["delta_max"],
                ),
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
                        "vy": [-4, 4],
                    },
                    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                    "grid_step": [5, 5],
                    "absolute": False,
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
