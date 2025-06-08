import os.path

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.decider import Decider
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.llm import get_structured_response
from sandra.utility.general import extract_scenario_and_planning_problem


def main(scenario_path: str):
    config = SanDRAConfiguration()
    save_path = scenario_path
    scenario, planning_problem = extract_scenario_and_planning_problem(scenario_path)
    describer = CommonRoadDescriber(
            scenario,
            planning_problem,
            0,
            config,
        )
    decider = Decider(scenario_path, config, describer)
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
    ]

    scenario_folder = os.path.join(PROJECT_ROOT, "scenarios")
    main(os.path.join(scenario_folder, scenario_paths[-2]))
