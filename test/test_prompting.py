import unittest

from commonroad.common.file_reader import CommonRoadFileReader

from sandra.common.config import PROJECT_ROOT, SanDRAConfiguration
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.utility.visualization import plot_scenario


class TestPrompting(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        name_scenario = "DEU_AachenBendplatz-1_80_T-19"
        path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + ".xml"
        self.scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
            lanelet_assignment=True
        )
        self.planning_problem = list(
            planning_problem_set.planning_problem_dict.values()
        )[0]

        self.config = SanDRAConfiguration()
        self.describer = CommonRoadDescriber(self.scenario, self.planning_problem, 0, self.config)
        self.user_prompt = self.describer.user_prompt()

    def test_user_prompt(self):
        plot_scenario(self.scenario, self.planning_problem)
        assert "intersection" in self.user_prompt
        assert "incoming lanes" in self.user_prompt
