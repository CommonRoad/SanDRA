import os.path
from typing import Optional, Union, List

import enum
from commonroad.scenario.scenario import Scenario

from commonroad_reach_semantic.data_structure.config.semantic_configuration_builder import SemanticConfigurationBuilder
from commonroad_reach_semantic.data_structure.config.semantic_configuration import SemanticConfiguration
from commonroad_reach_semantic.data_structure.environment_model.semantic_model import SemanticModel
from commonroad_reach_semantic.data_structure.model_checking.spot_interface import SpotInterface
from commonroad_reach_semantic.data_structure.reach.semantic_reach_interface import SemanticReachableSetInterface
from commonroad_reach_semantic.data_structure.rule.traffic_rule_interface import TrafficRuleInterface
from commonroad_reach_semantic.utility import visualization as util_visual

from sandra.actions import LongitudinalAction, LateralAction
from sandra.config import SanDRAConfiguration, COMMONROAD_REACH_SEMANTIC_ROOT, PROJECT_ROOT
from sandra.verifier.action2ltl import ActionLTL


class VerificationStatus(enum.Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"


class ReachVerifier:
    """Verifier using reachability analysis"""

    def __init__(self,
                 scenario: Scenario,
                 sandra_config: SanDRAConfiguration,
                 verbose=False):

        # basic elements
        self.verbose = verbose
        self.scenario = scenario
        self.sandra_config = sandra_config

        # reachability analysis configurations
        self.reach_config: Optional[SemanticConfiguration] =\
            SemanticConfigurationBuilder(
                COMMONROAD_REACH_SEMANTIC_ROOT
            ).build_configuration(
                str(scenario.scenario_id)
            )
        self.reach_config.traffic_rule.activated_rules = []
        self.reach_config.general.path_scenario =\
            PROJECT_ROOT + "/scenarios/" + str(scenario.scenario_id) + ".xml"
        self.reach_config.planning.dt = scenario.dt
        self.reach_config.planning.steps_computation = self.sandra_config.h
        self.reach_config.update()

        #initialize semantic model and traffic rule interface
        self.semantic_model = SemanticModel(self.reach_config)
        self.rule_interface = TrafficRuleInterface(self.reach_config, self.semantic_model)
        self.reach_interface = SemanticReachableSetInterface(self.reach_config,
                                                             self.semantic_model,
                                                             self.rule_interface)

    def reset(self,
              reach_config: SemanticConfiguration = None,
              actions: List[Union[LongitudinalAction, LateralAction]] = None):
        """resets configurations and actions"""
        if reach_config:
            self.reach_config = reach_config

            # update the config in the reach interface
            self.reach_interface.reset(
                config=self.reach_config,
            )

        if actions:
            ltl_list = []
            for action in actions:
                action_ltl = self.parse_action(action)
                if action_ltl:
                    ltl_list.append(action_ltl)
            self.reach_config.traffic_rule.list_traffic_rules_activated = ltl_list
            for item in self.reach_config.traffic_rule.list_traffic_rules_activated:
                self.rule_interface._parse_traffic_rule(item, allow_abstract_rules=True)
            self.rule_interface.print_summary()

            # reset the interface
            self.reach_interface.reset(
                config=self.reach_config,
                rule_interface=self.rule_interface
            )

    def parse_action(self, action: Union[LongitudinalAction, LateralAction]) -> str:
        """parses action to corresponding LTL formulas (or the correct reach configs)"""
        if action  == LongitudinalAction.ACCELERATE:
            self.reach_config.vehicle.ego.a_lon_min = self.sandra_config.a_lim
            return ""
        elif action == LongitudinalAction.DECELERATE:
            self.reach_config.vehicle.ego.a_lon_max = - self.sandra_config.a_lim
            return ""
        elif action == LongitudinalAction.KEEP:
            self.reach_config.vehicle.ego.a_lon_max = self.sandra_config.a_lim
            self.reach_config.vehicle.ego.a_lon_min = -self.sandra_config.a_lim
            return ""
        else:
            return ActionLTL.from_action(action)

    def verify(self, actions: List[Union[LongitudinalAction, LateralAction]]) -> VerificationStatus:
        """
        verifies the given actions (in a list)
        """
        self.reset(
            actions=actions,
        )

        # the formulas corresponding to all actions are conjunctively combined.
        self.reach_interface.compute_reachable_sets(
            step_end=self.sandra_config.h,
            verbose=self.verbose
        )

        # plot
        util_visual.plot_scenario_with_reachable_sets(self.reach_interface, save_gif=False)

        # checks whether the last time step in the horizon is reachable, i.e., whether the reachable set is empty
        if not self.reach_interface.reachable_set[self.sandra_config.h]:
            return VerificationStatus.UNSAFE
        else:
            return VerificationStatus.SAFE
