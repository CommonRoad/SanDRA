import copy
from typing import Optional, Any

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_reach.data_structure.reach.driving_corridor import DrivingCorridor

from sandra.actions import Action
from sandra.describer import Describer
from sandra.verifier_old import is_drivable
from sandra.llm import get_structured_response
from sandra.common.config import SanDRAConfiguration


class Decider:
    def __init__(
        self,
        scenario_path,
        timestep: int,
        config: SanDRAConfiguration,
        role_prompt: Optional[str] = None,
        goal_prompt: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        self.timestep = timestep
        self.config: SanDRAConfiguration = config
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

    def _parse_action_ranking(self, llm_response: dict[str, Any]) -> list[Action]:
        action_ranking = []
        for action in llm_response["action_ranking"]:
            action_ranking.append((action["longitudinal_action"], action["lateral_action"]))
        return action_ranking

    def run(self) -> DrivingCorridor:
        user_prompt = self.describer.user_prompt()
        system_prompt = self.describer.system_prompt()
        schema = self.describer.schema()
        structured_response = get_structured_response(user_prompt, system_prompt, schema, self.config, save_dir=self.save_path)
        ranking = self._parse_action_ranking(structured_response)

        print("Ranking:")
        for i, (longitudinal, lateral) in enumerate(ranking):
            print(f"{i + 1}. ({longitudinal}, {lateral})")

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
