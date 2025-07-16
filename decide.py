import unittest

from commonroad.common.file_reader import CommonRoadFileReader

from sandra.common.config import PROJECT_ROOT, SanDRAConfiguration
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.utility.visualization import draw_scenario_paper

name_scenario = "DEU_Goeppingen-37_1_T-4"
path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + ".xml"
scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
    lanelet_assignment=True
)
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

config = SanDRAConfiguration()


draw_scenario_paper(
    scenario,
    planning_problem,
    0,
    config,
    draw_planning_problem=False,
    plot_limits=config.plot_limits,
)


describer = CommonRoadDescriber(scenario, planning_problem, 0, config)

user_prompt = describer.user_prompt()

print(user_prompt)
