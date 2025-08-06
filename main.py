import os.path

from commonroad.common.file_reader import CommonRoadFileReader

from config.sandra import SanDRAConfiguration, PROJECT_ROOT
from sandra.commonroad.reach import ReachVerifier
from sandra.decider import Decider
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.llm import get_structured_response

import matplotlib

print(matplotlib.get_backend())
matplotlib.use("Agg")


def main(scenario_path: str):
    config = SanDRAConfiguration()
    save_path = scenario_path

    scenario, planning_problem_set = CommonRoadFileReader(
        scenario_path
    ).open(lanelet_assignment=True)
    planning_problem = next(
        iter(planning_problem_set.planning_problem_dict.values())
    )
    describer = CommonRoadDescriber(
        scenario, planning_problem, 0, config, goal="Drive faster.", describe_ttc=False
    )
    verifier = ReachVerifier(scenario, planning_problem, config)
    decider = Decider(config, describer, verifier, save_path=save_path)
    print(f"-----------------SYSTEM PROMPT-------------------")
    print(decider.describer.system_prompt())
    print(f"-----------------USER PROMPT-------------------")
    print(decider.describer.user_prompt())
    user_prompt = decider.describer.user_prompt()
    system_prompt = decider.describer.system_prompt()
    schema = decider.describer.schema()
    structured_response = get_structured_response(
        user_prompt, system_prompt, schema, decider.config
    )
    ranking = decider._parse_action_ranking(structured_response)

    print("Ranking:")
    for i, (longitudinal, lateral) in enumerate(ranking):
        print(f"{i + 1}. ({longitudinal}, {lateral})")


if __name__ == "__main__":
    scenario_paths = [
        "DEU_AachenAseag-1_80_T-99.xml",
        "DEU_AachenBendplatz-1_80_T-19.xml",
        "DEU_AachenFrankenburg-1_2120_T-39.xml",
        "DEU_AachenHeckstrasse-1_30520_T-539.xml",
        "DEU_LocationALower-11_10_T-1.xml",
        "DEU_Gar-1_1_T-1.xml",
        "DEU_Goeppingen-37_1_T-4.xml",
        "DEU_MONAEast-2_14326_T-14351.xml"
    ]

    scenario_folder = os.path.join(PROJECT_ROOT, "scenarios")
    main(os.path.join(scenario_folder, scenario_paths[-1]))
