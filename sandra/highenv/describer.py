from typing import Any, Optional, cast

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.vehicle.behavior import IDMVehicle, AggressiveVehicle, DefensiveVehicle
from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from sandra.actions import LateralAction, LongitudinalAction
from sandra.common.config import SanDRAConfiguration
from sandra.describer import DescriberBase


class HighEnvDescriber(DescriberBase):

    def __init__(self, env: AbstractEnv, config: SanDRAConfiguration, timestep: int, obstacle_type: type = IDMVehicle):
        super().__init__(timestep, config)
        assert obstacle_type in [IDMVehicle, AggressiveVehicle, DefensiveVehicle], f"Obstacle type {obstacle_type} is not supported."
        self.env = env
        self.ego: MDPVehicle = cast(MDPVehicle, env.vehicle)
        self.road: Road = env.road
        self.network: RoadNetwork = self.road.network
        self.id_to_lateral_action: dict[int, LateralAction] = {
            0: LateralAction.CHANGE_LEFT,
            1: LateralAction.KEEP,
            2: LateralAction.CHANGE_RIGHT,
            3: LateralAction.KEEP,
            4: LateralAction.KEEP
        }
        self.id_to_longitudinal_action: dict[int, LongitudinalAction] = {
            0: LongitudinalAction.KEEP,
            1: LongitudinalAction.KEEP,
            2: LongitudinalAction.KEEP,
            3: LongitudinalAction.ACCELERATE,
            4: LongitudinalAction.DECELERATE
        }

    @staticmethod
    def _describe_lane_position(lane_num: int, lane_idx) -> str:
        middle_lane = lane_num // 2 + 1 if lane_num % 2 == 1 else None
        left_most_lane = 0
        right_most_lane = lane_num - 1
        left_least_lane =  lane_num // 2
        if middle_lane is not None and lane_idx == middle_lane:
            return "middle lane"
        if lane_idx == left_most_lane:
            return "left-most lane"
        if lane_idx == right_most_lane:
            return "right-most lane"
        idx_to_str = {x: f"{x + 1}th" for x in range(3, left_least_lane + 1)}
        idx_to_str[1] = "2nd"
        idx_to_str[2] = "3rd"
        if lane_idx <= left_least_lane:
            return f"{idx_to_str[lane_idx]}-from-left lane"
        lane_idx = lane_num - 1 - lane_idx
        return f"{idx_to_str[lane_idx]}-from-right lane"

    def _describe_relative_lane_position(self, lane_idx: int) -> str:
        ego_lane_idx: LaneIndex = self.ego.lane_index
        side_lanes = self.network.all_side_lanes(ego_lane_idx)
        ego_lane_idx: int = ego_lane_idx[-1]
        if lane_idx == ego_lane_idx:
            return "your current lane"
        diff = ego_lane_idx - lane_idx
        if diff < 0:
            if diff == -1:
                return "the lane to your left"
            else:
                return f"{abs(diff)} lanes to your left"
        else:
            if diff == 1:
                return "the lane to your right"
            else:
                return f"{diff} lanes to your right"

    def _describe_traffic_signs(self) -> str:
        return ""

    def _describe_traffic_lights(self) -> str:
        return ""

    def _describe_vehicle(self, vehicle: type[IDMVehicle]) -> Optional[str]:
        lane_idx = vehicle.lane_index
        # vehicle_state = vehicle.prediction.trajectory.state_list[self.timestep]
        # vehicle_lanelet_id = find_lanelet_id_from_state(
        #     vehicle_state, self.scenario.lanelet_network
        # )
        # if vehicle_lanelet_id < 0:
        #     return None
        # implicit_lanelet_description = self.lanelet_network.describe_lanelet(
        #     vehicle_lanelet_id
        # )
        # if not implicit_lanelet_description:
        #     return None
        # vehicle_description = f"It is driving {implicit_lanelet_description}. "
        # ego_position = self.ego_state.position
        # relative_vehicle_direction = vehicle_state.position - ego_position
        # angle = calculate_relative_orientation(
        #     self.ego_direction, relative_vehicle_direction
        # )
        # vehicle_description += f"It is located {self.angle_description(angle)} you, "
        # vehicle_description += f"with a relative distance of {self.distance_description(ego_position, vehicle_state.position)}. "
        # vehicle_description += (
        #     f"Its velocity is {self.velocity_descr(vehicle_state.velocity)} "
        # )
        # vehicle_description += f"and its acceleration is {self.acceleration_descr(vehicle_state.acceleration)}."
        # if (ttc := self.ttc_description(vehicle.obstacle_id)) is not None:
        #     vehicle_description += f" The time-to-collision is {ttc}."
        # return vehicle_description

    def _get_relevant_obstacles(self) -> list:
        return cast(list, self.road.close_vehicles_to(
            self.ego, self.env.PERCEPTION_DISTANCE, see_behind=True,
            sort=True
        ))

    def _describe_obstacles(self) -> str:
        pass

    def _describe_ego_state(self) -> str:
        env_type = self.env.spec.id
        if env_type == "highway-v0":
            ego_description = "You are driving on a highway."
        else:
            raise NotImplementedError("Unsupported environment type.")
        ego_lane_idx: LaneIndex = self.ego.lane_index
        side_lanes = self.network.all_side_lanes(ego_lane_idx)
        ego_description += f" There are {len(side_lanes)} lanes on your current road."
        ego_description += (
            f"Your velocity is {self.velocity_descr(self.ego.speed)}"
        )
        ego_acceleration = self.ego.action['acceleration']
        ego_description += f" and your acceleration is {self.acceleration_descr(ego_acceleration)}""."
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
        return [
            "The best action is at index 0 in the array.",
            "You need to enumerate all combinations in your action ranking."
        ]

    def _get_available_actions(self) -> tuple[list[LateralAction], list[LongitudinalAction]]:
        availableActions = self.env.get_available_actions()
        laterals = [self.id_to_lateral_action[x] for x in availableActions]
        longitudinals = [self.id_to_longitudinal_action[x] for x in availableActions]
        return list(set(laterals)), list(set(longitudinals))
