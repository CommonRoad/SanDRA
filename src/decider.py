import copy
import os
from typing import Optional

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.prediction.prediction import TrajectoryPrediction

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad_reach.data_structure.reach.driving_corridor import DrivingCorridor

from src.actions import Action
from src.describer import Describer
from src.verifier import is_drivable
from src.llm import get_structured_response
from config import SaLaRAConfiguration


class Decider:
    def __init__(
        self,
        scenario_path,
        timestep: int,
        config: SaLaRAConfiguration,
        role_prompt: Optional[str] = None,
        goal_prompt: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        self.timestep = timestep
        self.config: SaLaRAConfiguration = config
        self.action_ranking = None
        self.scenario_path = scenario_path
        self.scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open(
            True
        )
        self.planning_problem = copy.deepcopy(
            list(planning_problem_set.planning_problem_dict.values())[0]
        )
        self.describer = Describer(self.scenario, self.planning_problem, timestep, config, role=role_prompt, goal=goal_prompt)
        self.save_path = save_path

    def _parse_action_ranking(self, ranking: dict) -> list[Action]:
        pass

    def run(self) -> DrivingCorridor:
        user_prompt = self.describer.user_prompt()
        system_prompt = self.describer.system_prompt()
        schema = self.describer.schema()
        ranking = get_structured_response(user_prompt, system_prompt, schema, self.config, save_dir=self.save_path)
        ranking = self._parse_action_ranking(ranking)

        print("Ranking:")
        for i, action in enumerate(ranking):
            print(f"{i + 1}. {action}")

        dc = None
        for action in ranking:
            if dc := is_drivable(self.scenario_path, action):
                print(f"Successfully verified {action}.")
                break
            print(f"Failed to verify {action}.")

        if dc is None:
            dc = is_drivable(self.scenario_path, None)
            print("Executing fail-safe trajectory instead.")
            assert (
                dc is not None
            ), f"There does not exist any driving corridor for {self.scenario.scenario_id}"
        return dc
