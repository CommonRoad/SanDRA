import copy
from typing import Optional, Union, List
import numpy as np
from commonroad.geometry.shape import ShapeGroup
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from lanelet2.traffic_rules import TrafficRules

from commonroad_reach_semantic.data_structure.config.semantic_configuration_builder import (
    SemanticConfigurationBuilder,
)
from commonroad_reach_semantic.data_structure.config.semantic_configuration import (
    SemanticConfiguration,
)
from commonroad_reach_semantic.data_structure.environment_model.semantic_model import (
    SemanticModel,
)
from commonroad_reach_semantic.data_structure.reach.semantic_reach_interface import (
    SemanticReachableSetInterface,
)
from commonroad_reach_semantic.data_structure.rule.traffic_rule_interface import (
    TrafficRuleInterface,
)
from commonroad_reach_semantic.utility import visualization as util_visual

from sandra.actions import LongitudinalAction, LateralAction
from sandra.common.config import (
    SanDRAConfiguration,
    COMMONROAD_REACH_SEMANTIC_ROOT,
    PROJECT_ROOT,
)
from sandra.utility.vehicle import extract_ego_vehicle
from sandra.common.road_network import EgoLaneNetwork, Lane
from sandra.verifier import ActionLTL, VerifierBase, VerificationStatus
from sandra.rules import InterstateRule
from commonroad_spot.spot_interface import SPOTInterface


class ReachVerifier(VerifierBase):
    """Verifier using reachability analysis"""

    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        sandra_config: SanDRAConfiguration,
        ego_lane_network: EgoLaneNetwork = None,
        verbose: bool = True,
        scenario_folder: str = None,
        highenv: bool = False,
    ):

        # basic elements
        super().__init__()
        self.verbose = verbose
        self.scenario = scenario
        self.initial_state = planning_problem.initial_state
        self.sandra_config = sandra_config
        self.ego_lane_network = ego_lane_network

        # reachability analysis configurations
        self.reach_config: Optional[SemanticConfiguration] = (
            SemanticConfigurationBuilder(
                COMMONROAD_REACH_SEMANTIC_ROOT
            ).build_configuration(str(scenario.scenario_id))
        )
        self.reach_config.traffic_rule.activated_rules = []
        if scenario_folder is None:
            scenario_folder = PROJECT_ROOT + "/scenarios/"
        self.reach_config.general.path_scenarios = scenario_folder
        self.reach_config.general.path_scenario = (
            scenario_folder + str(scenario.scenario_id) + ".xml"
        )
        self.reach_config.vehicle.ego.v_lon_min = 0.
        self.reach_config.vehicle.ego.t_react = 0.4
        self.reach_config.reachable_set.mode_computation = 8

        self.reach_config.planning.dt = scenario.dt
        if highenv:
            self.reach_config.vehicle.ego.v_lat_max = 12.
            self.reach_config.vehicle.ego.v_lat_min = -12.
            self.reach_config.vehicle.ego.a_lat_max = 4.
            self.reach_config.vehicle.ego.a_lat_min = -4.

            self.reach_config.vehicle.ego.a_lon_max = 6.
            self.reach_config.vehicle.ego.a_lon_min = -6.

            self.reach_config.vehicle.other.a_lon_min = -12.
            self.reach_config.vehicle.other.a_lon_max = 12.
        self.reach_config.planning.steps_computation = self.sandra_config.h
        self.reach_config.update()

        # remove ego vehicle if existed
        ego_vehicle = extract_ego_vehicle(scenario, planning_problem)
        if ego_vehicle:
            if ego_in_sce := self.reach_config.scenario.obstacle_by_id(
                ego_vehicle.obstacle_id
            ):
                self.reach_config.scenario.remove_obstacle(ego_in_sce)
                self.reset(
                    ego_lane_network=ego_lane_network,
                    scenario=self.reach_config.scenario,
                )
            self.reach_config.vehicle.ego.length = ego_vehicle.obstacle_shape.length
            self.reach_config.vehicle.ego.width = ego_vehicle.obstacle_shape.width
        else:
            self.reach_config.vehicle.ego.length = sandra_config.length
            self.reach_config.vehicle.ego.width = sandra_config.width

        # initialize semantic model and traffic rule interface
        self.semantic_model = SemanticModel(self.reach_config)
        rule_interface = TrafficRuleInterface(self.reach_config, self.semantic_model)
        self.reach_interface = SemanticReachableSetInterface(
            self.reach_config, self.semantic_model, rule_interface
        )

        # default params to be stored
        self._default_a_lon_max = self.reach_config.vehicle.ego.a_lon_max
        self._default_a_lon_min = self.reach_config.vehicle.ego.a_lon_min

        # vehicle id for the preceding vehicle
        self._preceding_veh_id = None
        self._other_a_max = 12. # m/s^2

    def reset(
        self,
        reach_config: SemanticConfiguration = None,
        actions: List[Union[LongitudinalAction, LateralAction]] = None,
        rules: List[Union[InterstateRule]] = None,
        ego_lane_network: EgoLaneNetwork = None,
        scenario: Scenario = None,
    ):
        """resets configs"""
        if reach_config:
            self.reach_config = reach_config

            # update the config in the reach interface
            self.reach_interface.reset(
                config=self.reach_config,
            )

        if ego_lane_network:
            self.ego_lane_network = ego_lane_network
            if scenario:
                self.reach_config.update(
                    planning_problem=self.reach_config.planning_problem,
                    scenario=scenario,
                    CLCS=self.ego_lane_network.lane.clcs,
                )

        if actions:
            ltl_list = []

            # reset the specification list within the rule interface
            for action in actions:
                if type(action) is LateralAction and EgoLaneNetwork is None:
                    AssertionError("For lateral actions, the lane network is needed!")
                action_ltl = self.parse_action(action)
                if action_ltl:
                    ltl_list.append(action_ltl)

            # including the formalized traffic rules
            if rules:
                for rule in rules:
                    rule_ltl = self.parse_traffic_rule(rule)
                    if rule_ltl:
                        ltl_list.append(rule_ltl)

            self.reach_config.traffic_rule.list_traffic_rules_activated = ltl_list

            rule_interface = TrafficRuleInterface(
                self.reach_config, self.semantic_model
            )
            for item in self.reach_config.traffic_rule.list_traffic_rules_activated:
                rule_interface._parse_traffic_rule(item, allow_abstract_rules=True)

            rule_interface.print_summary()
            self.reach_config.print_configuration_summary()

            # reset the interface
            self.reach_interface.reset(
                config=self.reach_config, rule_interface=rule_interface
            )

    def parse_action(
        self, action: Union[LongitudinalAction, LateralAction, None]
    ) -> str:
        """
        Parses the given action into an appropriate LTL formula or modifies reachability configuration.

        Returns:
            A string representing the corresponding LTL formula, or an empty string if handled by reach config.
        """
        if action == LongitudinalAction.ACCELERATE:
            self.reach_config.vehicle.ego.a_lon_min = self.sandra_config.a_lim
            self.reach_config.vehicle.ego.a_lon_max = self._default_a_lon_max
            return ""

        elif action == LongitudinalAction.DECELERATE:
            self.reach_config.vehicle.ego.a_lon_max = -self.sandra_config.a_lim
            self.reach_config.vehicle.ego.a_lon_min = self._default_a_lon_min
            return ""

        elif action == LongitudinalAction.KEEP:
            self.reach_config.vehicle.ego.a_lon_max = self.sandra_config.a_lim
            self.reach_config.vehicle.ego.a_lon_min = -self.sandra_config.a_lim
            return ""

        elif action == LateralAction.CHANGE_LEFT:
            if not self.ego_lane_network.lane_left_adjacent:
                raise AssertionError(f"No left adjacent lane for action {action}")
            clause = self._format_lane_clause(self.ego_lane_network.lane_left_adjacent)
            return ActionLTL.from_action(action).replace("InLeftAdjacentLane", clause)

        elif action == LateralAction.CHANGE_RIGHT:
            if not self.ego_lane_network.lane_right_adjacent:
                raise AssertionError(f"No right adjacent lane for action {action}")
            clause = self._format_lane_clause(self.ego_lane_network.lane_right_adjacent)
            return ActionLTL.from_action(action).replace("InRightAdjacentLane", clause)

        elif action == LateralAction.FOLLOW_LANE:
            if not self.ego_lane_network.lane:
                raise AssertionError(
                    f"No current lane assigned to ego for action {action}"
                )
            # fixme: reach semantic error -- no region for single lane
            if not self.ego_lane_network.lane_left_adjacent and not self.ego_lane_network.lane_right_adjacent:
                return "LTL true"
            clause = self._format_lane_clause([self.ego_lane_network.lane])
            return ActionLTL.from_action(action).replace("InCurrentLane", clause)

        elif action is None:
            return "LTL true"
        else:
            return ActionLTL.from_action(action)

    def parse_traffic_rule(
            self, traffic_rule: Union[InterstateRule, None]
    ) -> str:
        """
        Parses the given traffic rule into its corresponding LTL formula or applies
        modifications to the reachability configuration.
        """
        if traffic_rule == InterstateRule.RG_1:
            initial_lanelet = self.scenario.lanelet_network.find_most_likely_lanelet_by_state(
                [self.initial_state]
            )[0]

            distance_min = np.inf
            self._preceding_veh_id = None

            for obstacle in self.scenario.dynamic_obstacles:
                # Get the lanelets the obstacle occupies
                obstacle_lanelets = self.scenario.lanelet_network.find_lanelet_by_position(
                    [obstacle.initial_state.position]
                )[0]

                # Only consider if ego and obstacle share a lanelet
                if initial_lanelet not in obstacle_lanelets:
                    continue

                # Check whether obstacle is in front of ego
                if is_in_front(self.initial_state, obstacle.initial_state, threshold=0.0):
                    distance = np.linalg.norm(
                        np.array(self.initial_state.position) - np.array(obstacle.initial_state.position)
                    )
                    if distance < distance_min:
                        distance_min = distance
                        self._preceding_veh_id = obstacle.obstacle_id

            if self._preceding_veh_id is not None and not self.sandra_config.use_sonia:
                return f"LTL G (SafeDistance_V{self._preceding_veh_id})"
            # fixme: workaround for safety distance rule with set-based prediction
            elif self.sandra_config.use_rules_in_reach and self.sandra_config.use_sonia and self._preceding_veh_id is not None:
                pre_obs = self.reach_config.scenario.obstacle_by_id(self._preceding_veh_id)
                phantom_obs = self.reach_config.scenario.obstacle_by_id(85748)
                state_list = []
                for ts in range(1, self.sandra_config.h + 1):
                    min_vel = max(0.0,
                                  pre_obs.initial_state.velocity - ts * self.reach_config.scenario.dt * self._other_a_max)
                    if isinstance(pre_obs.occupancy_at_time(ts).shape, ShapeGroup):
                        min_rear_s = np.inf
                        for shape in pre_obs.occupancy_at_time(ts).shape.shapes:
                            min_rear_s = min(min_rear_s, shape.shapely_object.bounds[0])
                    else:
                        min_rear_s = pre_obs.occupancy_at_time(ts).shape.shapely_object.bounds[0]
                    min_center_s = min_rear_s + pre_obs.obstacle_shape.length / 2.0
                    state_list.append(CustomState(position=np.array([min_center_s, pre_obs.initial_state.position[1]]),
                                                  time_step=ts,
                                                  orientation=0.0, #todo
                                                  velocity=min_vel))
                phantom_obs.initial_state = pre_obs.initial_state
                phantom_obs.prediction = TrajectoryPrediction(
                    Trajectory(1, state_list), shape=pre_obs.obstacle_shape
                )

                return f"LTL G (!Behind_V{phantom_obs.obstacle_id} | SafeDistance_V{phantom_obs.obstacle_id})"
            else:
                return ""
        elif traffic_rule == InterstateRule.RG_3:
            return  "LTL G (KeepsBrakeSpeedLimit & KeepsFovSpeedLimit & KeepsTypeSpeedLimit & KeepsLaneSpeedLimit)"
        elif traffic_rule == InterstateRule.RG_2:
            self.reach_config.vehicle.ego.a_lon_min = -2.0
            return ""
        else:
            raise NotImplementedError(f"The rule {traffic_rule} is not supported.")

    def _format_lane_clause(self, lanes: List[Lane]) -> str:
        """
        Converts a list of lanes to a disjunctive clause over lanelet IDs.
        """
        lanelet_ids = []
        for lane in lanes:
            lanelet_ids.extend(lane.contained_ids)
        return " | ".join(f"InLanelet_{lid}" for lid in lanelet_ids)

    def verify(
        self,
        actions: List[Union[LongitudinalAction, LateralAction]],
        rules: List[Union[InterstateRule]] = None,
        only_in_lane: bool = False,
    ) -> VerificationStatus:
        if self.sandra_config.use_rules_in_reach and rules is None:
            rules = [InterstateRule.RG_1, InterstateRule.RG_2, InterstateRule.RG_3]

        if self.sandra_config.use_sonia:
            # self.sandra_config.a_lim = 0.11
            return self.verify_sonia(actions, rules, only_in_lane)
        else:
            return self.verify_base(actions, rules)

    def verify_base(
        self,
        actions: List[Union[LongitudinalAction, LateralAction]],
        rules: List[Union[InterstateRule]],
    ) -> VerificationStatus:
        """
        verifies the given actions (in a list)
        """
        print("[Verifier] Resetting with given actions...")
        self.reset(
            actions=actions,
            rules=rules,
        )

        print("[Verifier] Computing reachable sets...")
        try:
            # the formulas corresponding to all actions are conjunctively combined.
            self.reach_interface.compute_reachable_sets(
                step_end=self.sandra_config.h, verbose=self.verbose
            )
        except Exception as e:
            print("[Error] Exception during reachable set computation:", str(e))
            return VerificationStatus.UNSAFE

        # plot
        if self.sandra_config.visualize_reach:
            util_visual.plot_scenario_with_reachable_sets(
                self.reach_interface, save_gif=True
            )
            util_visual.plot_scenario_with_regions(self.semantic_model, "CVLN")

        # Checks whether the last time step in the horizon is reachable, i.e., whether the reachable set is empty
        if self.sandra_config.h >= len(self.reach_interface.reachable_set):
            print("[Verifier] Result: UNSAFE – Horizon index out of range.")
            return VerificationStatus.UNSAFE
        elif not self.reach_interface.reachable_set[self.sandra_config.h]:
            print("[Verifier] Result: UNSAFE – Final reachable set is empty.")
            return VerificationStatus.UNSAFE
        else:
            print("[Verifier] Result: SAFE")
            return VerificationStatus.SAFE

    def verify_sonia(
        self,
        actions: List[Union[LongitudinalAction, LateralAction]],
        rules: List[Union[InterstateRule]],
        only_in_lane: bool = False,
    ) -> VerificationStatus:
        update_dict = {
            "Vehicle": {
                0: {  # 0 means that all vehicles will be changed
                    "a_max": self._other_a_max,
                    "v_max": 30.0,
                    "compute_occ_m1": True,
                    "compute_occ_m2": True,
                    "compute_occ_m3": True,
                    "onlyInLane": only_in_lane,
                }
            },
            "EgoVehicle": {
                0: {  # ID is ignored for ego vehicle (which is created based on cr_planning problem)
                    "a_max": 6.0,
                    "length": 5.0,
                    "width": 2.0,
                }
            },
        }
        sonia_interface = SPOTInterface(
            scenario=self.reach_config.scenario,
            planning_problem=self.reach_config.planning_problem,
        )
        sonia_interface.set_logging_mode(True)
        sonia_interface.update_properties(update_dict)
        prediction_dict, scenario_time_step, occ_poly_list_dict, vel_interval_dict = (
            sonia_interface.do_occupancy_prediction(
                prediction_horizon=self.sandra_config.h + 1, update_dict=update_dict
            )
        )
        set_based_prediction_dict = sonia_interface.postprocess_results(
            scenario_time_step, prediction_dict, increment=1e-2
        )
        sonia_interface.update_scenario_with_results(
            set_based_prediction_dict, scenario_to_update=self.reach_config.scenario
        )

        return self.verify_base(
            actions=actions,
            rules=rules,
        )

    def extract_corridor(self):
        # todo: goal shape?
        return self.reach_interface.extract_driving_corridors(to_goal_region=False)[0]


def is_in_front(ego_state, obs_state, threshold=0.0):
    """
    Orientation check ensures that only obstacles in the driving direction of the ego
     are considered “in front, not just geometrically ahead along the lanelet.
     """
    # Ego position and heading
    ego_pos = np.array([ego_state.position[0], ego_state.position[1]])
    ego_heading = ego_state.orientation  # in radians
    ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)])

    # Obstacle position
    obs_pos = np.array([obs_state.position[0], obs_state.position[1]])

    # Vector from ego to obstacle
    v = obs_pos - ego_pos

    # Projection of v onto ego's direction
    projection = np.dot(v, ego_dir)

    return projection > threshold