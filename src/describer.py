from typing import Optional

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario

from config import SaLaRAConfiguration
from src.actions import get_all_actions


class Describer:
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, config: SaLaRAConfiguration, role: Optional[str], goal: Optional[str]):
        self.config = config
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.role = "" if role is None else role
        self.goal = "" if goal is None else goal
        self.actions = get_all_actions()

    def _find_lane_id(self, state) -> int:
        try:
            return self.scenario.lanelet_network.find_most_likely_lanelet_by_state(
                [state]
            )[0]
        except IndexError:
            return -1

    def schema(self) -> dict:
        return {
            "additionalProperties": False,
            "properties": {
                "reasoning": {"$ref": "#/$defs/Reasoning"},
                "action_ranking": {
                    "items": {
                        "enum": self.actions,
                        "type": "string"
                    },
                    "title": "Action Ranking",
                    "description": "Rank all available actions from best to worst",
                    "type": "array"
                }
            },
            "required": ["reasoning", "action_ranking"],
            "title": "Response",
            "type": "object",
            "$defs": {
                "Reasoning": {
                    "additionalProperties": False,
                    "properties": {
                        "observations": {
                            "items": {"type": "string"},
                            "title": "Observations",
                            "type": "array"
                        },
                        "decision": {
                            "title": "Decision",
                            "type": "string"
                        }
                    },
                    "required": ["observations", "decision"],
                    "title": "Reasoning",
                    "type": "object"
                }
            }
        }

    def user_prompt(self) -> str:
        # TODO
        pass

    def system_prompt(self) -> str:
        return f"""You are driving a car and need to decide what to do next.
{self.role}
{self.goal}
Considering the current traffic, what would you do in this kind of situation?
First observe the environment and formulate your decision in natural language. Then rank the following {len(self.actions)} actions from best to worst:
{self.actions}
Keep these things in mind:
1) You are currently driving in Germany and have to adhere to German traffic rules.
2) The best action is at index 0 in the array.
"""
