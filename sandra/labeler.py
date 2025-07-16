import itertools
import warnings
from abc import ABC, abstractmethod
from typing import Union, List, Optional

import numpy as np
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario

from sandra.actions import LateralAction, LongitudinalAction
from sandra.common.config import SanDRAConfiguration
from sandra.common.road_network import EgoLaneNetwork
from sandra.commonroad.reach import ReachVerifier

from commonroad_reach.utility import reach_operation as util_reach_operation

from sandra.verifier import VerificationStatus


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
    ) -> List[List[Union[LateralAction, LongitudinalAction]]]:

        long_label = self.longitudinal_label(obstacle)
        lat_label = self.lateral_label(obstacle, obs_lane_network)
        return [[long_label, lat_label]]

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

        if obs_lane_network.lane_right_adjacent:
            right_lanelet_ids = {
                lanelet_id
                for right_lane in obs_lane_network.lane_right_adjacent
                for lanelet_id in right_lane.contained_ids
            }
            if obs_lanelet_list[-1] in right_lanelet_ids:
                return LateralAction.CHANGE_RIGHT

        return LateralAction.UNKNOWN


class ReachSetLabeler(LabelerBase):
    def __init__(
        self,
        config: SanDRAConfiguration,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        scenario_folder: str = None,
    ):
        super().__init__(config, scenario)

        self.reach_ver = ReachVerifier(
            self.scenario,
            planning_problem,
            self.config,
            scenario_folder=scenario_folder,
        )

    def label(
        self,
        obstacle: DynamicObstacle,
        obs_lane_network: EgoLaneNetwork,
    ) -> List[List[Union[LateralAction, LongitudinalAction]]]:
        """
        label the top-m action pairs using the area of the corresponding reachable sets
        for a dynamic obstacle. Each action is a combination of one lateral and one
        longitudinal action.
        """
        # Reset the reach verifier
        self._reset_reach_verifier(obstacle, obs_lane_network)

        # Compute area per action
        action_area_dict = self._compute_action_areas(obs_lane_network)

        # Filter and select top-m actions
        top_action_sets = self._filter_and_select_actions(action_area_dict)

        return top_action_sets

    def _reset_reach_verifier(self, obstacle, obs_lane_network):
        """
        Resets the reach verifier, optionally removing the obstacle from the scenario.
        """
        existing_obstacle = self.reach_ver.reach_config.scenario.obstacle_by_id(
            obstacle.obstacle_id
        )

        if existing_obstacle is not None:
            self.reach_ver.reach_config.scenario.remove_obstacle(existing_obstacle)
            self.reach_ver.reset(
                ego_lane_network=obs_lane_network,
                scenario=self.reach_ver.reach_config.scenario,
            )
        else:
            self.reach_ver.reset(
                ego_lane_network=obs_lane_network,
            )

    def _compute_action_areas(self, obs_lane_network):
        """
        Computes the reachable area for each action pair and logs the results.
        Returns a dict: { (lat, lon) -> area }
        """
        import itertools

        # Candidate actions
        long_candidates = [
            LongitudinalAction.ACCELERATE,
            LongitudinalAction.DECELERATE,
            LongitudinalAction.KEEP,
        ]
        lat_candidates = [LateralAction.FOLLOW_LANE]
        if obs_lane_network.lane_left_adjacent:
            lat_candidates.append(LateralAction.CHANGE_LEFT)
        if obs_lane_network.lane_right_adjacent:
            lat_candidates.append(LateralAction.CHANGE_RIGHT)

        action_area_dict = {}

        for lat, lon in itertools.product(lat_candidates, long_candidates):
            status = self.reach_ver.verify([lat, lon])
            area = 0.0
            if status == VerificationStatus.SAFE:
                for (
                    _,
                    reach_set_nodes,
                ) in self.reach_ver.reach_interface.reachable_set.items():
                    area += util_reach_operation.compute_area_of_reach_nodes(
                        reach_set_nodes
                    )

            action_area_dict[(lat, lon)] = area

            print(
                f"Action pair ({lat.value}, {lon.value}): verification {status.name}, reachable area = {area:.3f}"
            )

        print("\n=== All action areas (sorted descending) ===")
        sorted_items = sorted(
            action_area_dict.items(), key=lambda item: item[1], reverse=True
        )
        for (lat, lon), area in sorted_items:
            print(f"\t- ({lat.value}, {lon.value}): area = {area:.3f}")

        return action_area_dict

    def _filter_and_select_actions(self, action_area_dict):
        """
        Filters non-zero area actions, issues a warning if fewer than m,
        and returns the top-m action sets.
        """

        # Sort descending
        sorted_actions = sorted(
            action_area_dict.items(), key=lambda item: item[1], reverse=True
        )

        # Keep only non-zero area actions
        non_zero_actions = [
            (lat, lon) for (lat, lon), area in sorted_actions if area > 0.0
        ]

        if len(non_zero_actions) < self.config.k:
            warnings.warn(
                f"Only {len(non_zero_actions)} valid action pairs with "
                f"non-zero reachable area (expected {self.config.k})."
            )

        # Take top m
        top_action_list = [
            [lon, lat] for (lat, lon) in non_zero_actions[: self.config.k]
        ]

        return top_action_list
