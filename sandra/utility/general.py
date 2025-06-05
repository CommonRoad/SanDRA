import copy

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario


def extract_scenario_and_planning_problem(
    absolute_scenario_path: str,
) -> tuple[Scenario, PlanningProblem]:
    scenario, planning_problem_set = CommonRoadFileReader(absolute_scenario_path).open(
        True
    )
    planning_problem = copy.deepcopy(
        list(planning_problem_set.planning_problem_dict.values())[0]
    )
    return scenario, planning_problem