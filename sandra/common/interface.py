from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario

from sandra.common.config import SanDRAConfiguration


class SanDRAInterface:
    def __init__(
        self,
        configuration: SanDRAConfiguration,
        scenario: Scenario,
        planning_problem: PlanningProblem,
    ):
        self.configuration = configuration
        self.scenario = scenario
        self.planning_problem = planning_problem

    def reset(
        self,
        configuration: SanDRAConfiguration = None,
        scenario: Scenario = None,
        planning_problem: PlanningProblem = None,
    ):
        if configuration:
            self.configuration = configuration
        if scenario:
            self.scenario = scenario
        if planning_problem:
            self.planning_problem = planning_problem
