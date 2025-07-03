from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario

from sandra.actions import LateralAction, LongitudinalAction
from sandra.common.config import SanDRAConfiguration
from sandra.common.road_network import EgoLaneNetwork


class LabelerBase(ABC):
    def __init__(self, config: SanDRAConfiguration, scenario: Scenario):
        self.config = config
        self.scenario = scenario

    @abstractmethod
    def label(
        self,
        obstacle: DynamicObstacle,
        obs_lane_network: EgoLaneNetwork,
    ) -> List[set[Union[LateralAction, LongitudinalAction]]]:
        """Assign a label to the trajectory of a dynamic obstacle.

        Subclasses must implement this method.
        """
        pass


class TrajectoryLabeler(LabelerBase):
    def __init__(self, config: SanDRAConfiguration, scenario: Scenario):
        super().__init__(config, scenario)

    def label(
        self, obstacle: DynamicObstacle, obs_lane_network: EgoLaneNetwork
    ) -> List[set[Union[LateralAction, LongitudinalAction]]]:

        long_label = self.longitudinal_label(obstacle)
        lat_label = self.lateral_label(obstacle, obs_lane_network)
        return [{long_label, lat_label}]

    @staticmethod
    def augment_state_acceleration(obstacle: DynamicObstacle, dt: float) -> List[float]:
        """augment the state acceleration of a dynamic obstacle of highd dataset."""
        accelerations = (
            np.diff(
                [
                    state.velocity
                    for state in [obstacle.initial_state]
                    + obstacle.prediction.trajectory.state_list
                ]
            )
            / dt
        ).tolist()
        obstacle.initial_state.acceleration = accelerations[0]
        for a, state in zip(
            accelerations[1:], obstacle.prediction.trajectory.state_list
        ):
            state.acceleration = a
        return accelerations

    def longitudinal_label(
        self, obstacle: DynamicObstacle
    ) -> Union[LongitudinalAction]:
        """label the longitudinal action of a dynamic obstacle."""
        accelerations = self.augment_state_acceleration(obstacle, self.scenario.dt)
        last_state = obstacle.prediction.trajectory.state_list[-1]
        # stopping
        # FG -> we only check the last state
        if abs(last_state.velocity) <= self.config.v_err:
            return LongitudinalAction.STOP
        # accelerating -> pick the average that is more robust than considering individual time steps
        elif np.average(accelerations) > self.config.a_lim:
            return LongitudinalAction.ACCELERATE
        # decelerating
        elif np.average(accelerations) < -self.config.a_lim:
            return LongitudinalAction.DECELERATE
        # default: idle
        elif self.config.a_lim > np.average(accelerations) > -self.config.a_lim:
            return LongitudinalAction.KEEP
        else:
            return LongitudinalAction.UNKNOWN

    def lateral_label(
        self, obstacle: DynamicObstacle, obs_lane_network: EgoLaneNetwork
    ) -> Union[LateralAction]:
        """label the lateral action of a dynamic obstacle."""
        # find a list of the most likely occupied lanelet
        obs_lanelet_list = (
            self.scenario.lanelet_network.find_most_likely_lanelet_by_state(
                [obstacle.initial_state] + obstacle.prediction.trajectory.state_list
            )
        )
        if all(
            lanelet in obs_lane_network.lane.contained_ids
            for lanelet in obs_lanelet_list
        ):
            return LateralAction.FOLLOW_LANE
        if obs_lane_network.lane_left_adjacent:
            left_lanelet_ids = {
                lanelet_id
                for left_lane in obs_lane_network.lane_left_adjacent
                for lanelet_id in left_lane.contained_ids
            }
            # FG -> check the last state
            if obs_lanelet_list[-1] in left_lanelet_ids:
                return LateralAction.CHANGE_LEFT

        if obs_lane_network.lane_left_adjacent:
            right_lanelet_ids = {
                lanelet_id
                for right_lane in obs_lane_network.lane_right_adjacent
                for lanelet_id in right_lane.contained_ids
            }
            if obs_lanelet_list[-1] in right_lanelet_ids:
                return LateralAction.CHANGE_RIGHT

        return LateralAction.UNKNOWN


class ReachSetLabeler(LabelerBase):
    def __init__(self, config: SanDRAConfiguration, scenario: Scenario):
        super().__init__(config, scenario)
