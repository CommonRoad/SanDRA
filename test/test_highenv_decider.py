"""
Unit tests of the verifier module using reachability analysis
"""

import unittest
from sandra.highenv.decider import HighEnvDecider


class TestHighEnvDecider(unittest.TestCase):
    def setUp(self) -> None:
        self.decider = HighEnvDecider.make([4213])

    def test_user_prompt(self):
        user_prompt = self.decider.describer.user_prompt()
        assert (
            "You are driving on a highway. There are 4 lanes on your current road. Your velocity is 90.0 km/h and your acceleration is 0.0 m/s²."
            in user_prompt
        )
        assert (
            """Here is an overview over all relevant obstacles surrounding you:
    - VEHICLE 1: It is driving on your current lane. It is located in front of you, with a relative distance of 11.3 meters. Its velocity is 84.8 km/h and its acceleration is 0.0 m/s².
    - VEHICLE 2: It is driving on a lane 3 lanes to your left. It is located left of you, with a relative distance of 25.6 meters. Its velocity is 86.3 km/h and its acceleration is 0.0 m/s².
    - VEHICLE 3: It is driving on a lane 3 lanes to your left. It is located left of you, with a relative distance of 34.8 meters. Its velocity is 85.3 km/h and its acceleration is 0.0 m/s².
    - VEHICLE 4: It is driving on your current lane. It is located in front of you, with a relative distance of 42.1 meters. Its velocity is 79.7 km/h and its acceleration is 0.0 m/s².
    - VEHICLE 5: It is driving on a lane 2 lanes to your left. It is located left of you, with a relative distance of 54.2 meters. Its velocity is 85.5 km/h and its acceleration is 0.0 m/s².
    - VEHICLE 6: It is driving on a lane 3 lanes to your left. It is located left of you, with a relative distance of 64.4 meters. Its velocity is 76.5 km/h and its acceleration is 0.0 m/s²."""
            in user_prompt
        )

    def test_system_prompt(self):
        system_prompt = self.decider.describer.system_prompt()
        actions = ["keep", "decelerate", "accelerate", "straight", "left"]
        not_actions = ["stop", "right"]
        for action in actions:
            assert action in system_prompt
        for not_action in not_actions:
            assert not_action not in system_prompt

    def test_video_generation(self):
        import os
        import gymnasium
        from gymnasium.wrappers import RecordVideo

        env = gymnasium.make("highway-v0", render_mode="rgb_array")
        env = RecordVideo(
            env, video_folder="run", episode_trigger=lambda e: True
        )  # record all episodes
        # Provide the video recorder to the wrapped environment
        # so it can send it intermediate simulation frames.
        env.unwrapped.set_record_video_wrapper(env)
        # Record a video as usual
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            env.render()
        env.close()
        assert os.path.exists("run"), "Run folder was not created"
        assert os.path.exists(
            "run/rl-video-episode-0.mp4"
        ), "Video file was not created"
        assert os.path.exists(
            "run/rl-video-episode-0.meta.json"
        ), "Meta file was not created"

    def test_run(self):
        self.decider.run()
        assert True
