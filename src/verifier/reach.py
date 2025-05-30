from typing import Optional

from commonroad.scenario.scenario import Scenario

from commonroad_reach_semantic.data_structure.config.semantic_configuration_builder import SemanticConfigurationBuilder
from commonroad_reach_semantic.data_structure.config.semantic_configuration import SemanticConfiguration
from commonroad_reach_semantic.data_structure.environment_model.semantic_model import SemanticModel
from commonroad_reach_semantic.data_structure.model_checking.spot_interface import SpotInterface
from commonroad_reach_semantic.data_structure.reach.semantic_reach_interface import SemanticReachableSetInterface
from commonroad_reach_semantic.data_structure.rule.traffic_rule_interface import TrafficRuleInterface

class ReachVerifier:
    """Verifier using reachability analysis"""

    def __init__(self,
                 scenario: Scenario,
                 verbose=False):
        self.verbose = verbose

        # todo: general config with scenarios
        self.reach_config: Optional[SemanticConfiguration] =\
            SemanticConfigurationBuilder().build_configuration(str(scenario.scenario_id))


        # ==== initialize semantic model and traffic rule interface
        self.semantic_model = SemanticModel(self.reach_config)
        self.rule_interface = TrafficRuleInterface(self.reach_config, self.semantic_model)
        self.reach_interface = SemanticReachableSetInterface(self.reach_config,
                                                             self.semantic_model,
                                                             self.rule_interface)

    def update(self,
               reach_config: SemanticConfiguration = None):
        #
        if reach_config:
            self.reach_config = reach_config

            # update the config in the reach interface
            self.reach_interface.reset(
                config=self.reach_config,
            )




    def compute_reachable_sets(self):
        pass

    def verify(self, reachable_sets):
        pass