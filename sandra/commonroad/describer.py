from typing import Optional, Any
import numpy as np
from openai import BaseModel

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficSignIDGermany
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.measure import TTC

from sandra.actions import LateralAction, LongitudinalAction
from sandra.common.config import SanDRAConfiguration
from sandra.common.road_network import EgoLaneNetwork, RoadNetwork
from sandra.describer import DescriberBase, Thoughts, Action
from sandra.utility.vehicle import (
    find_lanelet_id_from_state,
    extract_ego_vehicle,
    calculate_relative_orientation,
)


class HighLevelDrivingDecision(BaseModel):
    thoughts: Thoughts
    best_combination: Action
    model_config = {"extra": "forbid"}


class CommonRoadDescriber(DescriberBase):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, timestep: int,
                 config: SanDRAConfiguration, role: Optional[str] = None, goal: Optional[str] = None,
                 scenario_type: Optional[str] = None, describe_ttc=True, k=5, enforce_k=False):
        self.ego_lane_network: EgoLaneNetwork = None
        self.ego_direction = None
        self.ego_state = None
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.ego_vehicle = extract_ego_vehicle(scenario, planning_problem)
        self.describe_ttc = describe_ttc
        assert 1 <= k <= 10, f"Unsupported k {k}"
        self.k = k
        self.enforce_k = enforce_k
        super().__init__(timestep, config, role, goal, scenario_type)

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
        road_network = RoadNetwork.from_lanelet_network_and_position(self.scenario.lanelet_network, self.ego_state.position)
        self.ego_lane_network = EgoLaneNetwork.from_route_planner(self.scenario.lanelet_network, self.planning_problem, road_network)

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

    def _describe_lanelet(self, lanelet_id) -> Optional[str]:
        if self.ego_lane_network.lane and self.ego_lane_network.lane.contains(lanelet_id):
            return "your current lane"
        for (direction, side), lanes in self.ego_lane_network.neighbor_dict.items():
            lane_list = [] if lanes is None else lanes
            for lane in lane_list:
                if lane.contains(lanelet_id):
                    return f"the {direction}-direction lane to your {side}"
        return None

    def _describe_vehicle(self, vehicle: DynamicObstacle) -> Optional[str]:
        vehicle_state = vehicle.prediction.trajectory.state_list[self.timestep]
        vehicle_lanelet_id = find_lanelet_id_from_state(
            vehicle_state, self.scenario.lanelet_network
        )
        if vehicle_lanelet_id < 0:
            return None
        implicit_lanelet_description = self._describe_lanelet(
            vehicle_lanelet_id
        )
        if not implicit_lanelet_description:
            return None
        vehicle_description = f"It is driving on {implicit_lanelet_description}. "
        ego_position = self.ego_state.position
        relative_vehicle_direction = vehicle_state.position - ego_position
        angle = calculate_relative_orientation(
            self.ego_direction, relative_vehicle_direction
        )
        vehicle_description += f"It is located {self.angle_description(angle)} you, "
        vehicle_description += f"with a relative distance of {self.distance_description(ego_position, vehicle_state.position)}. "
        vehicle_description += (
            f"Its velocity is {self.velocity_descr(vehicle_state.velocity)} "
        )
        vehicle_description += f"and its acceleration is {self.acceleration_descr(vehicle_state.acceleration)}."
        if (ttc := self.ttc_description(vehicle.obstacle_id)) is not None:
            vehicle_description += f" The time-to-collision is {ttc}."
        return vehicle_description

    def _get_relevant_obstacles(self, perception_radius=100) -> list[DynamicObstacle]:
        circle_center = self.ego_state.position
        return [x for x in self.scenario.dynamic_obstacles if np.linalg.norm(x.prediction.trajectory.state_list[self.timestep].position - circle_center) < perception_radius]

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
        ego_description = f"You are currently driving in a {self.scenario_type} scenario." if self.scenario_type else ""
        for (direction, side), lanes in self.ego_lane_network.neighbor_dict.items():
            if lanes:
                quantifier = f"are {direction}-direction lanes" if len(lanes) > 1 else f"is a {direction}-direction lane"
                ego_description += f" There {quantifier} to your {side}."
        ego_description += (
            f"\nYour velocity is {self.velocity_descr(self.ego_state.velocity)}"
        )
        ego_description += f" and your acceleration is {self.acceleration_descr(self.ego_state.acceleration)}."
        return ego_description

    def _describe_schema(self) -> str:
        laterals, longitudinals = self._get_available_actions()
        laterals_str = "\n".join([f"    - {x.value}" for x in laterals])
        longitudinals_str = "\n".join([f"    - {x.value}" for x in longitudinals])
        return f"""First observe the environment and formulate your decision in natural language. Then return a ranking of the top-{self.k} advisable action combinations:
Longitudinal actions:
{longitudinals_str}
Lateral actions:
{laterals_str}"""

    def _describe_reminders(self) -> list[str]:
        reminders = [
            "You are currently driving in Germany and have to adhere to German traffic rules.",
            "You need to enumerate all combinations in your action ranking."
        ]
        if not self.enforce_k:
            reminders.append("The best action is at index 0 in the array.")
        return reminders

    def _get_available_actions(self) -> tuple[list[LateralAction], list[LongitudinalAction]]:
        lateral_actions = [LateralAction.KEEP]
        if self.ego_lane_network.lane_left_adjacent or self.ego_lane_network.lane_left_reversed:
            lateral_actions.append(LateralAction.CHANGE_LEFT)
        if self.ego_lane_network.lane_right_adjacent or self.ego_lane_network.lane_right_reversed:
            lateral_actions.append(LateralAction.CHANGE_RIGHT)
        longitudinal_actions = [x for x in LongitudinalAction]
        return lateral_actions, longitudinal_actions

    def schema(self) -> dict[str, Any]:
        laterals, longitudinals = self.get_available_actions()
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

        action_dict = schema_dict["properties"]["best_combination"]
        variable_name_prefixes = [
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ]
        added_variable_names = []
        for prefix in variable_name_prefixes[:self.k-1]:
            variable_name = f"{prefix}_best_combination"
            schema_dict["properties"][variable_name] = action_dict
            added_variable_names.append(variable_name)

        schema_dict["required"] = schema_dict["required"] + added_variable_names
        return schema_dict


if __name__ == "__main__":
    pass
