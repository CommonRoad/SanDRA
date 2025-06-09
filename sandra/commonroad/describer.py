from typing import Optional
import numpy as np

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficSignIDGermany
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.measure import TTC

from sandra.actions import LateralAction, LongitudinalAction
from sandra.common.config import SanDRAConfiguration
from sandra.describer import DescriberBase
from sandra.commonroad.lanelet_network import EgoCenteredLaneletNetwork
from sandra.utility.vehicle import (
    find_lanelet_id_from_state,
    extract_ego_vehicle,
    calculate_relative_orientation,
)


class CommonRoadDescriber(DescriberBase):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, timestep: int,
                 config: SanDRAConfiguration, role: Optional[str] = None, goal: Optional[str] = None,
                 scenario_type: Optional[str] = None, describe_ttc=True):
        super().__init__(timestep, config, role, goal, scenario_type)
        self.lanelet_network: EgoCenteredLaneletNetwork = None
        self.ego_direction = None
        self.ego_state = None
        self.scenario = scenario
        self.ego_vehicle = extract_ego_vehicle(scenario, planning_problem)
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
            "You are currently driving in Germany and have to adhere to German traffic rules.",
            "The best action is at index 0 in the array.",
            "You need to enumerate all combinations in your action ranking."
        ]

    def _get_available_actions(self) -> tuple[list[LateralAction], list[LongitudinalAction]]:
        return self.lanelet_network.lateral_actions(), self.lanelet_network.longitudinal_actions()


if __name__ == "__main__":
    pass
