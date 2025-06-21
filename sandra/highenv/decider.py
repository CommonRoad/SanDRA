import random
from typing import cast

import gymnasium
import numpy as np
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from highway_env.envs.common.observation import TimeToCollisionObservation
from matplotlib import pyplot as plt

from sandra.actions import LateralAction, LongitudinalAction
from sandra.common.config import SanDRAConfiguration
from sandra.decider import Decider
from sandra.highenv.describer import HighEnvDescriber


class HighEnvDecider(Decider):
    def __init__(
        self,
        env_wrapper: Env,
        observation: TimeToCollisionObservation,
        config: SanDRAConfiguration,
    ):
        describer = HighEnvDescriber(env_wrapper.unwrapped, observation, config, 0)
        super().__init__(config, describer, None)
        self.env_wrapper = env_wrapper
        self.describer = cast(HighEnvDescriber, describer)
        self.lateral_action_to_id: dict[LateralAction, int] = {
            LateralAction.CHANGE_LEFT: 0,
            LateralAction.KEEP: 1,
            LateralAction.CHANGE_RIGHT: 2,
        }
        self.longitudinal_action_to_id: dict[LongitudinalAction, int] = {
            LongitudinalAction.KEEP: 1,
            LongitudinalAction.ACCELERATE: 3,
            LongitudinalAction.DECELERATE: 4,
        }

    def run(self):
        done = truncated = False
        plt.imshow(self.env_wrapper.render())
        plt.show()
        while not (done or truncated):
            longitudinal_action, lateral_action = self.decide()
            if lateral_action in [
                LateralAction.CHANGE_RIGHT,
                LateralAction.CHANGE_LEFT,
            ]:
                action = self.lateral_action_to_id[lateral_action]
            else:
                action = self.longitudinal_action_to_id[longitudinal_action]
            obs, reward, done, truncated, info = self.env_wrapper.step(action)
            self.describer.update_with_observation(obs)
            self.env_wrapper.render()
            plt.imshow(self.env_wrapper.render())
            plt.show()
        self.env_wrapper.close()

    @staticmethod
    def make(seeds: list[int] = None) -> "HighEnvDecider":
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
        env_config = {
            "highway-v0": {
                "observation": {"type": "TimeToCollision", "horizon": 10},
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": np.linspace(5, 32, 9),
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

        env = gymnasium.make(
            "highway-v0", render_mode="rgb_array", config=env_config["highway-v0"]
        )
        env = RecordVideo(
            env, video_folder="run", episode_trigger=lambda e: True
        )  # record all episodes
        env.unwrapped.set_record_video_wrapper(env)
        seed = random.choice(seeds)
        obs, _ = env.reset(seed=seed)
        return HighEnvDecider(env, obs, SanDRAConfiguration())


if __name__ == "__main__":
    decider = HighEnvDecider.make([4213])
    decider.run()
