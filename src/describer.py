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

    def _extract_scenario_information(
        self, expected_num_ts: int = 20, min_num_ts: int = 10
    ) -> tuple[int, DynamicObstacle, Optional[DynamicObstacle]]:
        # remove ego vehicle from obstacles
        ego_vehicle: Optional[DynamicObstacle] = None
        ego_count = 0
        for vehicle in self.scenario.dynamic_obstacles:
            diff: np.ndarray = (
                vehicle.initial_state.position
                - self.planning_problem.initial_state.position
            )
            if np.linalg.norm(diff) < 0.1:
                ego_vehicle = vehicle
                ego_count += 1

        assert (
            ego_count == 1
        ), f"There is not exactly one ego vehicle in scenario {self.scenario.scenario_id}, instead got: {ego_count}"
        assert hasattr(ego_vehicle, "prediction") and isinstance(
            ego_vehicle.prediction, TrajectoryPrediction
        ), "Ego vehicle does not have a predicted movement!"

        if (
            num_time_steps := len(ego_vehicle.prediction.trajectory.state_list)
        ) != expected_num_ts:
            print(
                f"WARNING: len of ego prediction is actually {num_time_steps} instead of {expected_num_ts}"
            )
            assert (
                num_time_steps >= min_num_ts
            ), f"Length of ego prediction should be at least {min_num_ts} but got {num_time_steps}"

        ego_vehicle: DynamicObstacle = ego_vehicle
        ego_lanelet_id = self._find_lane_id(ego_vehicle.initial_state)

        # Try to find the nearest vehicle in front
        ego_orientation = np.array(
            [
                np.cos(self.planning_problem.initial_state.orientation),
                np.sin(self.planning_problem.initial_state.orientation),
            ]
        )
        follow_vehicle_data: Optional[tuple[DynamicObstacle, float]] = None
        for obstacle in self.scenario.dynamic_obstacles:
            if ego_lanelet_id == self._find_lane_id(obstacle.initial_state):
                vehicle_dir = (
                    obstacle.initial_state.position
                    - self.planning_problem.initial_state.position
                )
                dist = float(np.linalg.norm(vehicle_dir))
                vehicle_proj = np.dot(ego_orientation, vehicle_dir / dist)
                angle = np.arccos(vehicle_proj)
                cross_product = np.cross(ego_orientation, vehicle_dir / dist)
                if cross_product < 0:
                    angle = -angle
                threshold = np.pi / 8
                if -threshold <= angle <= threshold:
                    if follow_vehicle_data:
                        if follow_vehicle_data[1] > dist:
                            follow_vehicle_data = obstacle, dist
        if follow_vehicle_data:
            follow_vehicle = follow_vehicle_data[0]
        else:
            follow_vehicle = None
        return num_time_steps, ego_vehicle, follow_vehicle

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
