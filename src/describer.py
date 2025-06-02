from typing import Optional
import numpy as np

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficSignIDGermany
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.measure import TTC

from config import SaLaRAConfiguration
from src.actions import get_all_actions
from src.lanelet_network import EgoCenteredLaneletNetwork
from src.utils import find_lanelet_id_from_state, extract_ego_vehicle, calculate_relative_orientation


class Describer:
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, timestep: int, config: SaLaRAConfiguration, role: Optional[str] = None, goal: Optional[str] = None, scenario_type: Optional[str] = None, describe_ttc=True):
        self.lanelet_network = None
        self.ego_direction = None
        self.ego_state = None
        self.config = config
        self.timestep = timestep
        self.scenario = scenario
        self.ego_vehicle = extract_ego_vehicle(scenario, planning_problem)
        self.update(timestep=timestep)
        self.role = "" if role is None else role
        self.goal = "" if goal is None else goal
        self.scenario_type = "" if scenario_type is None else f"You are currently in an {scenario_type} scenario."
        self.actions = get_all_actions()

        self.describe_ttc = describe_ttc
        if describe_ttc:
            config = CriMeConfiguration()
            config.update(ego_id=self.ego_vehicle.obstacle_id, sce=scenario)
            self.ttc_evaluator = TTC(config)

    def update(self, timestep=None):
        if timestep is not None:
            self.timestep = timestep
        else:
            self.timestep = self.timestep + 1
        self.ego_state = self.ego_vehicle.prediction.trajectory.state_list[timestep]
        self.ego_direction: np.ndarray = np.array(
            [
                np.cos(self.ego_state.orientation),
                np.sin(self.ego_state.orientation),
            ]
        )
        self.lanelet_network = EgoCenteredLaneletNetwork(self.scenario.lanelet_network, find_lanelet_id_from_state(self.ego_state, self.scenario.lanelet_network))

    @staticmethod
    def velocity_descr(v: float, to_km=True) -> str:
        if to_km:
            v *= 3.6
            return f"{v:.1f} km/h"
        return f"{v:.1f} m/s"

    @staticmethod
    def acceleration_descr(a: float, to_km=False) -> str:
        if to_km:
            a *= 12960
            return f"{a:.1f} km/h²"
        return f"{a:.1f} m/s²"

    @staticmethod
    def angle_description(theta: float) -> str:
        if abs(0 - theta) < np.pi / 4:
            return "in front of"
        elif abs(np.pi/2 - theta) < np.pi / 4:
            return "left of"
        elif abs(np.pi - theta) < np.pi / 4:
            return "behind"
        else:
            return "right of"

    def distance_description(self, obstacle_position: np.ndarray) -> str:
        dist = np.linalg.norm(obstacle_position - self.ego_state.position)
        if dist > 0:
            return f"{dist:.1f} meters in front of"
        else:
            return f"{dist:.1f} meters behind"

    def ttc_description(self, obstacle_id: int) -> Optional[str]:
        if not self.ttc_evaluator or not self.describe_ttc:
            return None
        return f"{self.ttc_evaluator.compute(obstacle_id, self.timestep)} sec"

    def _describe_traffic_signs(self) -> str:
        max_speed = None
        for traffic_sign in self.scenario.lanelet_network.traffic_signs:
            for traffic_sign_element in traffic_sign.traffic_sign_elements:
                if traffic_sign_element.traffic_sign_element_id == TrafficSignIDGermany.MAX_SPEED:
                    max_speed = float(traffic_sign_element.additional_values[0])

        # TODO: Add support for more traffic rules
        traffic_signs_description = "These are all the traffic rules that currently apply to you:"
        initial_len = len(traffic_signs_description)
        if max_speed is not None:
            traffic_signs_description += f"\nThe maximum speed is {self.velocity_descr(max_speed)}."

        if len(traffic_signs_description) > initial_len:
            return traffic_signs_description
        return ""

    def _describe_traffic_lights(self) -> str:
        # TODO: Implement this
        return ""

    def _describe_vehicle(self, vehicle: DynamicObstacle) -> Optional[str]:
        vehicle_state = vehicle.prediction.trajectory.state_list[self.timestep]
        vehicle_lanelet_id = find_lanelet_id_from_state(vehicle_state, self.scenario.lanelet_network)
        implicit_lanelet_description = self.lanelet_network.describe_lanelet(vehicle_lanelet_id)
        if not implicit_lanelet_description:
            return None
        vehicle_description = f"It is driving {implicit_lanelet_description}. "
        relative_vehicle_direction = vehicle_state.position - self.ego_state.position
        angle = calculate_relative_orientation(self.ego_direction, relative_vehicle_direction)
        vehicle_description += f"It is located {self.angle_description(angle)} you. "
        vehicle_description += f"It is {self.distance_description(vehicle_state.position)} you. "
        vehicle_description += f"Its velocity is {self.velocity_descr(vehicle_state.velocity)} "
        vehicle_description += f"and its acceleration is {self.acceleration_descr(vehicle_state.acceleration)}."
        if (ttc := self.ttc_description(vehicle.obstacle_id)) is not None:
            vehicle_description += f" The time-to-collision is {ttc}."
        return vehicle_description

    def _get_relevant_obstacles(self) -> list[DynamicObstacle]:
        # TODO: Filter out obstacles outside a certain radius
        return self.scenario.dynamic_obstacles

    def _describe_obstacles(self) -> str:
        # TODO: Add support for static obstacles
        obstacle_description = "Here is an overview over all relevant obstacles surrounding you:\n"
        initial_len = len(obstacle_description)
        indent = "    "
        for obstacle in self._get_relevant_obstacles():
            if obstacle.obstacle_type in [
                ObstacleType.CAR,
                ObstacleType.BUS,
                ObstacleType.BICYCLE,
                ObstacleType.TRUCK
            ]:
                try:
                    temp = self._describe_vehicle(obstacle)
                    if temp is None:
                        continue
                    obstacle_description += f"{indent}- {obstacle.obstacle_type.value} {obstacle.obstacle_id}: "
                    obstacle_description += temp + "\n"
                except IndexError:
                    print(f"WARNING: Skipped {obstacle.obstacle_id} due to mysterious IndexError.")
            elif obstacle.obstacle_type == ObstacleType.PEDESTRIAN:
                print(f"WARNING: Skipped {obstacle.obstacle_id} because it is a pedestrian.")
            else:
                raise ValueError(f"Unexpected obstacle type: {obstacle.obstacle_type}")
        if len(obstacle_description) == initial_len:
            return "There are no obstacles surrounding you."
        return obstacle_description

    def _describe_ego_state(self) -> str:
        ego_description = self.lanelet_network.describe()
        ego_description += f"Your velocity is {self.velocity_descr(self.ego_state.velocity)}"
        ego_description += f" and your acceleration is {self.acceleration_descr(self.ego_state.acceleration)}."
        return ego_description

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
        return f"""Here is an overview over your environment:
{self.scenario_type}
{self._describe_traffic_signs()}
{self._describe_ego_state()}
{self._describe_traffic_lights()}
{self._describe_obstacles()}
"""

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
