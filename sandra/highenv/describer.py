from typing import Any, Optional, cast, overload

import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import TimeToCollisionObservation
from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.vehicle.behavior import IDMVehicle, AggressiveVehicle, DefensiveVehicle
from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from sandra.actions import LateralAction, LongitudinalAction
from sandra.common.config import SanDRAConfiguration
from sandra.commonroad.describer import HighLevelDrivingDecision
from sandra.describer import DescriberBase
from sandra.utility.vehicle import calculate_relative_orientation


class HighEnvDescriber(DescriberBase):

    def __init__(
        self,
        env: AbstractEnv,
        observation: TimeToCollisionObservation,
        config: SanDRAConfiguration,
        timestep: int,
        obstacle_type: type = IDMVehicle,
        switch_sides: bool = True,
    ):
        super().__init__(
            timestep,
            config,
            goal="Your goal is to maneuver through the highway as fast as possible while avoiding any collisions.",
        )
        self.ego_direction = None
        assert obstacle_type in [
            IDMVehicle,
            AggressiveVehicle,
            DefensiveVehicle,
        ], f"Obstacle type {obstacle_type} is not supported."
        self.env = env
        self.ego: MDPVehicle = cast(MDPVehicle, env.vehicle)
        self.road: Road = env.road
        self.network: RoadNetwork = self.road.network
        self.k = 5
        self.id_to_lateral_action: dict[int, LateralAction] = {
            0: LateralAction.CHANGE_LEFT,
            1: LateralAction.KEEP,
            2: LateralAction.CHANGE_RIGHT,
            3: LateralAction.KEEP,
            4: LateralAction.KEEP,
        }
        self.id_to_longitudinal_action: dict[int, LongitudinalAction] = {
            0: LongitudinalAction.KEEP,
            1: LongitudinalAction.KEEP,
            2: LongitudinalAction.KEEP,
            3: LongitudinalAction.ACCELERATE,
            4: LongitudinalAction.DECELERATE,
        }
        if switch_sides:
            self.direction_mapping: dict[str, str] = {
                "left": "right",
                "right": "left",
            }
        else:
            self.direction_mapping: dict[str, str] = {
                "left": "left",
                "right": "right",
            }
        self.update_with_observation(observation)

    def update_with_observation(self, observation: TimeToCollisionObservation):
        self.ego_direction: np.ndarray = np.array(
            [
                np.cos(self.ego.heading),
                np.sin(self.ego.heading),
            ]
        )

    def _describe_relative_lane_position(self, lane_idx: int) -> str:
        ego_lane_idx: LaneIndex = self.ego.lane_index
        ego_lane_idx: int = ego_lane_idx[-1]
        if lane_idx == ego_lane_idx:
            return "your current lane"
        diff = ego_lane_idx - lane_idx
        if diff < 0:
            if diff == -1:
                return "the lane to your left"
            else:
                return f"a lane {abs(diff)} lanes to your left"
        else:
            if diff == 1:
                return "the lane to your right"
            else:
                return f"a lane {diff} lanes to your right"

    def _describe_traffic_signs(self) -> str:
        return ""

    def _describe_traffic_lights(self) -> str:
        return ""

    def _describe_vehicle(self, vehicle) -> Optional[str]:
        lane_idx = vehicle.lane_index
        vehicle_description = (
            f"It is driving on {self._describe_relative_lane_position(lane_idx[-1])}."
        )
        relative_vehicle_direction = vehicle.position - self.ego.position
        angle = calculate_relative_orientation(
            self.ego_direction, relative_vehicle_direction
        )
        vehicle_description += f" It is located {self.angle_description(angle)} you, "
        vehicle_description += f"with a relative distance of {self.distance_description(self.ego.position, vehicle.position)}. "
        vehicle_description += f"Its velocity is {self.velocity_descr(vehicle.speed)} "
        vehicle_description += f"and its acceleration is {self.acceleration_descr(vehicle.action['acceleration'])}."
        # todo add ttc from observational matrix
        # if (ttc := self.ttc_description(vehicle.obstacle_id)) is not None:
        #     vehicle_description += f" The time-to-collision is {ttc}."
        left = self.direction_mapping["left"]
        right = self.direction_mapping["right"]
        vehicle_description = vehicle_description.replace("left", "sandra")
        vehicle_description = vehicle_description.replace("right", right)
        vehicle_description = vehicle_description.replace("sandra", left)
        return vehicle_description

    def get_relevant_obstacles(self) -> list:
        return cast(
            list,
            self.road.close_vehicles_to(
                self.ego, self.env.PERCEPTION_DISTANCE / 3.0, see_behind=True, sort=True
            ),
        )

    def _describe_obstacles(self) -> str:
        obstacles = self.get_relevant_obstacles()
        if not obstacles:
            return "There are no obstacles surrounding you."
        indentation = " " * 4
        obstacle_description = (
            "Here is an overview over all relevant obstacles surrounding you:\n"
        )
        for obstacle in obstacles:
            obstacle_id = self.road.vehicles.index(obstacle)
            obstacle_description += f"{indentation}- VEHICLE {obstacle_id}: "
            obstacle_description += self._describe_vehicle(obstacle) + "\n"
        return obstacle_description

    def _describe_ego_state(self) -> str:
        env_type = self.env.spec.id
        if env_type == "highway-v0":
            ego_description = "You are driving on a highway."
        else:
            raise NotImplementedError("Unsupported environment type.")
        ego_lane_idx: LaneIndex = self.ego.lane_index
        side_lanes = self.network.all_side_lanes(ego_lane_idx)
        ego_description += f" There are {len(side_lanes)} lanes on your current road."
        ego_description += f" Your velocity is {self.velocity_descr(self.ego.speed)}"
        ego_acceleration = self.ego.action["acceleration"]
        ego_description += (
            f" and your acceleration is {self.acceleration_descr(ego_acceleration)}" "."
        )
        return ego_description

    def _describe_schema(self) -> str:
        laterals, longitudinals = self._get_available_actions()
        laterals_str = "\n".join([f"    - {x.value}" for x in laterals])
        longitudinals_str = "\n".join([f"    - {x.value}" for x in longitudinals])
        return f"""First observe the environment and formulate your decision in natural language. Then return a ranking of the advisable actions which consist of {len(laterals) * len(longitudinals)} combinations:
Longitudinal actions:
{longitudinals_str}
Lateral actions:
{laterals_str}"""

    def _describe_reminders(self) -> list[str]:
        # return [
        #     "The best action is at index 0 in the array.",
        #     "You need to enumerate all combinations in your action ranking."
        # ]
        return []

    def _get_available_actions(
        self,
    ) -> tuple[list[LateralAction], list[LongitudinalAction]]:
        availableActions = self.env.get_available_actions()
        laterals = [self.id_to_lateral_action[x] for x in availableActions]
        longitudinals = [self.id_to_longitudinal_action[x] for x in availableActions]
        return list(set(laterals)), list(set(longitudinals))

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
