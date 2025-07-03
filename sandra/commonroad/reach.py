from typing import Optional, Union, List
from commonroad.scenario.scenario import Scenario

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
from sandra.common.road_network import EgoLaneNetwork, Lane
from sandra.verifier import ActionLTL, VerifierBase, VerificationStatus


class ReachVerifier(VerifierBase):
    """Verifier using reachability analysis"""

    def __init__(
        self,
        scenario: Scenario,
        sandra_config: SanDRAConfiguration,
        ego_lane_network: EgoLaneNetwork = None,
        verbose=False,
    ):

        # basic elements
        super().__init__()
        self.verbose = verbose
        self.scenario = scenario
        self.sandra_config = sandra_config
        self.ego_lane_network = ego_lane_network

        # reachability analysis configurations
        self.reach_config: Optional[SemanticConfiguration] = (
            SemanticConfigurationBuilder(
                COMMONROAD_REACH_SEMANTIC_ROOT
            ).build_configuration(str(scenario.scenario_id))
        )
        self.reach_config.traffic_rule.activated_rules = []
        self.reach_config.general.path_scenario = (
            PROJECT_ROOT + "/scenarios/" + str(scenario.scenario_id) + ".xml"
        )
        self.reach_config.vehicle.ego.v_lon_min = 0
        # fix the dimension
        self.reach_config.vehicle.ego.length = sandra_config.length
        self.reach_config.vehicle.ego.width = sandra_config.width
        self.reach_config.general.path_scenarios = PROJECT_ROOT + "/scenarios/"
        self.reach_config.planning.dt = scenario.dt
        self.reach_config.planning.steps_computation = self.sandra_config.h
        self.reach_config.update()

        # initialize semantic model and traffic rule interface
        self.semantic_model = SemanticModel(self.reach_config)
        self.rule_interface = TrafficRuleInterface(
            self.reach_config, self.semantic_model
        )
        self.reach_interface = SemanticReachableSetInterface(
            self.reach_config, self.semantic_model, self.rule_interface
        )

        # default params to be stored
        self._default_a_lon_max = self.reach_config.vehicle.ego.a_lon_max
        self._default_a_lon_min = self.reach_config.vehicle.ego.a_lon_min

    def reset(
        self,
        reach_config: SemanticConfiguration = None,
        actions: List[Union[LongitudinalAction, LateralAction]] = None,
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
                    CLCS = self.ego_lane_network.lane.clcs
                )

        if actions:
            ltl_list = []
            # reset the specification list within the rule interface
            self.rule_interface.list_specifications_ltl = []
            for action in actions:
                if type(action) is LateralAction and EgoLaneNetwork is None:
                    AssertionError("For lateral actions, the lane network is needed!")
                action_ltl = self.parse_action(action)
                if action_ltl:
                    ltl_list.append(action_ltl)
            self.reach_config.traffic_rule.list_traffic_rules_activated = ltl_list
            for item in self.reach_config.traffic_rule.list_traffic_rules_activated:
                self.rule_interface._parse_traffic_rule(item, allow_abstract_rules=True)
            self.rule_interface.print_summary()

            # reset the interface
            self.reach_interface.reset(
                config=self.reach_config, rule_interface=self.rule_interface
            )

    def parse_action(self, action: Union[LongitudinalAction, LateralAction]) -> str:
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
            clause = self._format_lane_clause([self.ego_lane_network.lane])
            return ActionLTL.from_action(action).replace("InCurrentLane", clause)

        else:
            return ActionLTL.from_action(action)

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
        visualization=False,
    ) -> VerificationStatus:
        """
        verifies the given actions (in a list)
        """
        print("[Verifier] Resetting with given actions...")
        self.reset(
            actions=actions,
        )

        print("[Verifier] Computing reachable sets...")
        # the formulas corresponding to all actions are conjunctively combined.
        self.reach_interface.compute_reachable_sets(
            step_end=self.sandra_config.h, verbose=self.verbose
        )

        # plot
        if visualization:
            util_visual.plot_scenario_with_reachable_sets(
                self.reach_interface, save_gif=True
            )

        # checks whether the last time step in the horizon is reachable, i.e., whether the reachable set is empty
        if not self.reach_interface.reachable_set[self.sandra_config.h]:
            print("[Verifier] Result: UNSAFE – Final reachable set is empty.")
            return VerificationStatus.UNSAFE
        else:
            print("[Verifier] Result: SAFE")
            return VerificationStatus.SAFE

    def extract_corridor(self):
        # todo: goal shape?
        return self.reach_interface.extract_driving_corridors(to_goal_region=False)[0]
