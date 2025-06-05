from typing import Optional, Literal, Any
from openai import BaseModel
import numpy as np

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficSignIDGermany
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.measure import TTC

from sandra.common.config import SanDRAConfiguration
from sandra.lanelet_network import EgoCenteredLaneletNetwork
from sandra.utility import (
    find_lanelet_id_from_state,
    extract_ego_vehicle,
    calculate_relative_orientation,
)


class Thoughts(BaseModel):
    observation: list[str]
    conclusion: str
    model_config = {"extra": "forbid"}


class Action(BaseModel):
    lateral_action: Literal["placeholder"]
    longitudinal_action: Literal["placeholder"]
    model_config = {"extra": "forbid"}


class HighLevelDrivingDecision(BaseModel):
    thoughts: Thoughts
    action_ranking: list[Action]
    model_config = {"extra": "forbid"}


class Describer:
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        timestep: int,
        config: SanDRAConfiguration,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        scenario_type: Optional[str] = None,
        describe_ttc=True,
    ):
        self.lanelet_network: EgoCenteredLaneletNetwork = None
        self.ego_direction = None
        self.ego_state = None
        self.config = config
        self.timestep = timestep
        self.scenario = scenario
        self.ego_vehicle = extract_ego_vehicle(scenario, planning_problem)
        self.update(timestep=timestep)

        self.role = "" if role is None else role
        self.goal = "" if goal is None else goal
        self.scenario_type = (
            ""
            if scenario_type is None
            else f"You are currently in an {scenario_type} scenario."
        )

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
        self.lanelet_network = EgoCenteredLaneletNetwork(
            self.scenario.lanelet_network,
            find_lanelet_id_from_state(self.ego_state, self.scenario.lanelet_network),
        )

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
        elif abs(np.pi / 2 - theta) < np.pi / 4:
            return "left of"
        elif abs(np.pi - theta) < np.pi / 4:
            return "behind"
        else:
            return "right of"

    def distance_description(self, obstacle_position: np.ndarray) -> str:
        dist = np.linalg.norm(obstacle_position - self.ego_state.position)
        return f"{dist:.1f} meters"

    def ttc_description(self, obstacle_id: int) -> Optional[str]:
        if not self.ttc_evaluator or not self.describe_ttc:
            return None
        return f"{self.ttc_evaluator.compute(obstacle_id, self.timestep)} sec"

    def _describe_traffic_signs(self) -> str:
        max_speed = None
        for traffic_sign in self.scenario.lanelet_network.traffic_signs:
            for traffic_sign_element in traffic_sign.traffic_sign_elements:
                if (
                    traffic_sign_element.traffic_sign_element_id
                    == TrafficSignIDGermany.MAX_SPEED
                ):
                    max_speed = float(traffic_sign_element.additional_values[0])

        # TODO: Add support for more traffic rules
        traffic_signs_description = (
            "These are all the traffic rules that currently apply to you:"
        )
        initial_len = len(traffic_signs_description)
        if max_speed is not None:
            traffic_signs_description += (
                f"\nThe maximum speed is {self.velocity_descr(max_speed)}."
            )

        if len(traffic_signs_description) > initial_len:
            return traffic_signs_description
        return ""

    def _describe_traffic_lights(self) -> str:
        # TODO: Implement this
        return ""

    def _describe_vehicle(self, vehicle: DynamicObstacle) -> Optional[str]:
        vehicle_state = vehicle.prediction.trajectory.state_list[self.timestep]
        vehicle_lanelet_id = find_lanelet_id_from_state(
            vehicle_state, self.scenario.lanelet_network
        )
        if vehicle_lanelet_id < 0:
            return None
        implicit_lanelet_description = self.lanelet_network.describe_lanelet(
            vehicle_lanelet_id
        )
        if not implicit_lanelet_description:
            return None
        vehicle_description = f"It is driving {implicit_lanelet_description}. "
        relative_vehicle_direction = vehicle_state.position - self.ego_state.position
        angle = calculate_relative_orientation(
            self.ego_direction, relative_vehicle_direction
        )
        vehicle_description += f"It is located {self.angle_description(angle)} you, "
        vehicle_description += f"with a relative distance of {self.distance_description(vehicle_state.position)}. "
        vehicle_description += (
            f"Its velocity is {self.velocity_descr(vehicle_state.velocity)} "
        )
        vehicle_description += f"and its acceleration is {self.acceleration_descr(vehicle_state.acceleration)}."
        if (ttc := self.ttc_description(vehicle.obstacle_id)) is not None:
            vehicle_description += f" The time-to-collision is {ttc}."
        return vehicle_description

    def _get_relevant_obstacles(self) -> list[DynamicObstacle]:
        # TODO: Filter out obstacles outside a certain radius
        return self.scenario.dynamic_obstacles

    def _describe_obstacles(self) -> str:
        # TODO: Add support for static obstacles
        obstacle_description = (
            "Here is an overview over all relevant obstacles surrounding you:\n"
        )
        initial_len = len(obstacle_description)
        indent = "    "
        for obstacle in self._get_relevant_obstacles():
            if obstacle.obstacle_type in [
                ObstacleType.CAR,
                ObstacleType.BUS,
                ObstacleType.BICYCLE,
                ObstacleType.TRUCK,
            ]:
                try:
                    temp = self._describe_vehicle(obstacle)
                    if temp is None:
                        continue
                    obstacle_description += f"{indent}- {obstacle.obstacle_type.value} {obstacle.obstacle_id}: "
                    obstacle_description += temp + "\n"
                except IndexError:
                    print(
                        f"WARNING: Skipped {obstacle.obstacle_id} due to mysterious IndexError."
                    )
            elif obstacle.obstacle_type == ObstacleType.PEDESTRIAN:
                print(
                    f"WARNING: Skipped {obstacle.obstacle_id} because it is a pedestrian."
                )
            else:
                raise ValueError(f"Unexpected obstacle type: {obstacle.obstacle_type}")
        if len(obstacle_description) == initial_len:
            return "There are no obstacles surrounding you."
        return obstacle_description

    def _describe_ego_state(self) -> str:
        ego_description = self.lanelet_network.describe()
        ego_description += (
            f"Your velocity is {self.velocity_descr(self.ego_state.velocity)}"
        )
        ego_description += f" and your acceleration is {self.acceleration_descr(self.ego_state.acceleration)}."
        return ego_description

    def schema(self) -> dict[str, Any]:
        laterals = self.lanelet_network.lateral_actions()
        longitudinals = self.lanelet_network.longitudinal_actions()
        schema_dict = HighLevelDrivingDecision.model_json_schema()
        lateral_action = schema_dict["$defs"]["Action"]["properties"]["lateral_action"]
        lateral_action["enum"] = laterals
        if len(laterals) == 1:
            lateral_action["const"] = laterals[0]
        else:
            lateral_action.pop("const", None)
        longitudinal_action = schema_dict["$defs"]["Action"]["properties"][
            "longitudinal_action"
        ]
        longitudinal_action["enum"] = longitudinals
        if len(longitudinals) == 1:
            longitudinal_action["const"] = longitudinals[0]
        else:
            longitudinal_action.pop("const", None)

        return schema_dict

    def user_prompt(self) -> str:
        return f"""Here is an overview over your environment:
{self.scenario_type}
{self._describe_traffic_signs()}
{self._describe_ego_state()}
{self._describe_traffic_lights()}
{self._describe_obstacles()}
"""

    def system_prompt(self) -> str:
        laterals = self.lanelet_network.lateral_actions()
        laterals_str = "\n".join([f"    - {x}" for x in laterals])
        longitudinals = self.lanelet_network.longitudinal_actions()
        longitudinals_str = "\n".join([f"    - {x}" for x in longitudinals])
        return f"""You are driving a car and need to make a high-level driving decision.
{self.role}
{self.goal}
First observe the environment and formulate your decision in natural language. Then return a ranking of the advisable actions which consist of {len(laterals) * len(longitudinals)} combinations:
Longitudinal actions:
{longitudinals_str}
Lateral actions:
{laterals_str}
Keep these things in mind:
1) You are currently driving in Germany and have to adhere to German traffic rules.
2) The best action is at index 0 in the array.
3) You need to enumerate all combinations in your action ranking.
"""


if __name__ == "__main__":
    pass
