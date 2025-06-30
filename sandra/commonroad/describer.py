import copy
import math
from typing import Optional, Any, Union, List, Tuple
import numpy as np
from commonroad.geometry.shape import Rectangle
from openai import BaseModel

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficSignIDGermany
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.measure import TTC
from commonroad_crime.utility.general import check_elements_state

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
        k=3,
        past_action: List[Union[LongitudinalAction, LateralAction]] = None,
        country: Optional[str] = "Germany",
    ):
        self.ego_lane_network: EgoLaneNetwork = None
        self.ego_direction = None
        self.ego_state = None
        self.ego_past_action = past_action
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.country = country
        self.ego_vehicle = extract_ego_vehicle(scenario, planning_problem)
        self.describe_ttc = describe_ttc
        assert 1 <= k <= 10, f"Unsupported k {k}"
        self.k = k

        if describe_ttc:
            crime_config = CriMeConfiguration()
            if self.ego_vehicle:
                crime_config.update(ego_id=self.ego_vehicle.obstacle_id, sce=scenario)
            else:
                check_elements_state(planning_problem.initial_state)
                self.ego_vehicle = StaticObstacle(
                    obstacle_id=planning_problem.planning_problem_id,
                    obstacle_type=ObstacleType.CAR,
                    obstacle_shape=Rectangle(length=config.length, width=config.width),
                    initial_state=planning_problem.initial_state,
                )
                crime_sce = copy.deepcopy(scenario)
                crime_sce.add_objects(self.ego_vehicle)
                crime_config.update(ego_id=self.ego_vehicle.obstacle_id, sce=crime_sce)

            self.ttc_evaluator = TTC(crime_config)
        else:
            self.ttc_evaluator = None
        super().__init__(timestep, config, role, goal, scenario_type)


    def update(self, timestep=None):
        if timestep is not None:
            self.timestep = timestep
        else:
            self.timestep = self.timestep + 1
        self.ego_state = self.planning_problem.initial_state
        self.ego_direction: np.ndarray = np.array(
            [
                np.cos(self.ego_state.orientation),
                np.sin(self.ego_state.orientation),
            ]
        )
        road_network = RoadNetwork.from_lanelet_network_and_position(
            self.scenario.lanelet_network, self.ego_state.position
        )
        self.ego_lane_network = EgoLaneNetwork.from_route_planner(
            self.scenario.lanelet_network, self.planning_problem, road_network
        )

        # clcs for crime
        if self.ttc_evaluator:
            self.ttc_evaluator.configuration.update(CLCS=self.ego_lane_network.lane.clcs)

        if not self.scenario_type and (
            self.ego_lane_network.lane_incoming_left
            or self.ego_lane_network.lane_incoming_right
        ):
            self.scenario_type = "intersection"

    def ttc_description(self, obstacle_id: int) -> Optional[str]:
        if not self.describe_ttc or not self.ttc_evaluator:
            return None

        ttc = self.ttc_evaluator.compute(obstacle_id, self.timestep)

        try:
            ttc_val = float(ttc)
            if math.isnan(ttc_val):
                return "inf sec"
            return f"{ttc_val:.1f} sec"
        except (TypeError, ValueError):
            return None

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
            f"Please adhere to the traffic regulations in {self.country}:"
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

    def _describe_lanelet(
        self, lanelet_id
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        # Check if lanelet is in ego lane
        if self.ego_lane_network.lane and self.ego_lane_network.lane.contains(
            lanelet_id
        ):
            return "the same lane", None, None

        # Check neighboring lanes (left/right and direction)
        for (direction, side), lanes in self.ego_lane_network.neighbor_dict.items():
            for lane in lanes or []:
                if lane.contains(lanelet_id):
                    return (
                        f"the {side}-adjacent lane in the {direction} direction",
                        side,
                        direction,
                    )

        # Check incoming lanes (left and right)
        incoming_checks = [
            ("the left incoming lane", self.ego_lane_network.lane_incoming_left),
            ("the right incoming lane", self.ego_lane_network.lane_incoming_right),
        ]
        for description, lane_list in incoming_checks:
            for lane in lane_list or []:
                if lane.contains(lanelet_id):
                    return description, None, "incoming"

        # If not found, return None
        return None, None, None

    def _describe_vehicle(self, vehicle: DynamicObstacle) -> Optional[str]:
        vehicle_state = vehicle.prediction.trajectory.state_list[self.timestep]
        vehicle_lanelet_id = find_lanelet_id_from_state(
            vehicle_state, self.scenario.lanelet_network
        )
        if vehicle_lanelet_id < 0:
            return None
        implicit_lanelet_description, side, direction = self._describe_lanelet(
            vehicle_lanelet_id
        )
        if not implicit_lanelet_description:
            return None
        vehicle_description = f"It is driving on {implicit_lanelet_description} "
        ego_position = self.ego_state.position
        # todo: clcs distance
        # relative_vehicle_direction = vehicle_state.position - ego_position
        # angle = calculate_relative_orientation(
        #     self.ego_direction, relative_vehicle_direction
        # )
        # vehicle_description += f"It is located {self.angle_description(angle)} you, "
        vehicle_description += f"and is {self.distance_description_clcs(ego_position, vehicle_state.position, self.ego_lane_network.lane.clcs, direction)}. "
        vehicle_description += (
            f"Its velocity is {self.velocity_descr(vehicle_state.velocity)} "
        )
        vehicle_description += f"and its acceleration is {self.acceleration_descr(vehicle_state.acceleration)}."
        if (ttc := self.ttc_description(vehicle.obstacle_id)) is not None:
            if ttc:
                vehicle_description += f" The time-to-collision is {ttc}."
        return vehicle_description

    def _get_relevant_obstacles(self, perception_radius=100) -> list[DynamicObstacle]:
        circle_center = self.ego_state.position
        return [
            x
            for x in self.scenario.dynamic_obstacles
            if np.linalg.norm(
                x.prediction.trajectory.state_list[self.timestep].position
                - circle_center
            )
            < perception_radius
        ]

    def _describe_obstacles(self) -> str:
        # TODO: Add support for static obstacles
        obstacle_description = (
            "Here is an overview of all relevant obstacles surrounding you:\n"
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
                if self.ego_vehicle and obstacle.obstacle_id == self.ego_vehicle.obstacle_id:
                    continue
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
        if self.scenario_type == "intersection":
            ego_description = "You are currently approaching an intersection."
        elif self.scenario_type == "roundabout":
            ego_description = "You are currently entering a roundabout."
        elif self.scenario_type:
            article = "an" if self.scenario_type[0].lower() in "aeiou" else "a"
            ego_description = f"You are currently driving in {article} {self.scenario_type} scenario."
        else:
            ego_description = ""

        if (
            self.ego_lane_network.lane_incoming_left
            and self.ego_lane_network.lane_incoming_right
        ):
            ego_description += " There are incoming lanes on both the left and right. "
        elif self.ego_lane_network.lane_incoming_left:
            ego_description += " There are incoming lanes on the left. "
        elif self.ego_lane_network.lane_incoming_right:
            ego_description += " There are incoming lanes on the right. "

        sides = {"left": [], "right": []}

        for (direction, side), lanes in self.ego_lane_network.neighbor_dict.items():
            if side in sides and lanes:
                sides[side].append(
                    f"a {side}-adjacent lane with the {direction} direction"
                )

        for side in ["left", "right"]:
            if sides[side]:
                for desc in sides[side]:
                    ego_description += f"There is {desc}. "
            else:
                ego_description += f"There is no {side}-adjacent lane. "

        ego_description += (
            f"\nYour velocity is {self.velocity_descr(self.ego_state.velocity)}"
        )
        ego_description += f" and your acceleration is {self.acceleration_descr(self.ego_state.acceleration)}."
        if self.ego_past_action:
            actions_str = ", ".join(action.value for action in self.ego_past_action)
            ego_description += f"Your last actions are: {actions_str}. "
        return ego_description

    def _describe_schema(self) -> str:
        laterals, longitudinals = self._get_available_actions()
        laterals_str = "\n".join([f"  - {x.value}" for x in laterals])
        longitudinals_str = "\n".join([f"  - {x.value}" for x in longitudinals])
        return (
            "First observe the environment and formulate your decision in natural language.\n"
            f"Then, return the top {self.k} advisable longitudinal–lateral action pairs, ranked from best to worst.\n"
            "Feasible longitudinal actions:\n"
            f"{longitudinals_str}\n"
            "Feasible lateral actions:\n"
            f"{laterals_str}"
        )

    def _describe_reminders(self) -> list[str]:
        reminders = [
            "You are currently driving in Germany and have to adhere to German traffic rules.",
        ]
        return reminders

    def _get_available_actions(
        self,
    ) -> tuple[list[LateralAction], list[LongitudinalAction]]:
        lateral_actions = [LateralAction.FOLLOW_LANE]
        if (
            self.ego_lane_network.lane_left_adjacent
            or self.ego_lane_network.lane_left_reversed
        ):
            lateral_actions.append(LateralAction.CHANGE_LEFT)
        if (
            self.ego_lane_network.lane_right_adjacent
            or self.ego_lane_network.lane_right_reversed
        ):
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
        for prefix in variable_name_prefixes[: self.k - 1]:
            variable_name = f"{prefix}_best_combination"
            schema_dict["properties"][variable_name] = action_dict
            added_variable_names.append(variable_name)

        schema_dict["required"] = schema_dict["required"] + added_variable_names
        return schema_dict


if __name__ == "__main__":
    pass
