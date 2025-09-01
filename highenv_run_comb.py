"""
Standalone script to create HighEnvDecider and run it for all combinations.
"""
import os

from highway_env.vehicle.behavior import IDMVehicle
from sandra.common.config import SanDRAConfiguration
from sandra.highenv.decider import HighEnvDecider
import matplotlib
import itertools

print(matplotlib.get_backend())
matplotlib.use("TkAgg")


def main():
    seeds = [
        5838,
        2421,
        7294,
        9650,
        4176,
        6382,
        8765,
        1348,  # initial spot fail
        4213,
        2572,  # initial spot fail
    ]

    # Scenario settings: (lanes_count, vehicles_density)
    scenario_settings = [
        (4, 2.0),  # setting 1
        (4, 3.0),  # setting 2
        (5, 3.0),  # setting 3
    ]

    for seed in seeds:
        # Cartesian product of configs
        configs = itertools.product(
            [True, False],  # use_rules_in_reach
            [True, False],  # use_rules_in_prompt
            ["set-based", "most-likely"],  # prediction type
            scenario_settings
        )
        for use_rules_in_reach, use_rules_in_prompt, pred_type, (lanes, density) in configs:
            config = SanDRAConfiguration()

            # 1. rules in reachable set
            config.use_rules_in_reach = use_rules_in_reach

            # 2. rules in prompt
            config.use_rules_in_prompt = use_rules_in_prompt

            config.visualize_reach = False

            # 3. prediction type
            if pred_type == "set-based":
                config.use_sonia = True
                config.h = 8
            else:
                config.use_sonia = False
                config.h = 15

            # 4. scenario setup
            config.highway_env.lanes_count = lanes
            config.highway_env.vehicles_density = density

            # Seed
            config.highway_env.seeds = [seed]

            # Run decider
            save_path = config.highway_env.get_save_folder(
                config.model_name,
                seed,
                config.use_sonia,
                config.use_rules_in_prompt,
                config.use_rules_in_reach
            ) + "/evaluation.csv"

            print(
                f"\n=== Running seed {seed}, "
                f"reach={use_rules_in_reach}, "
                f"prompt={use_rules_in_prompt}, "
                f"pred={pred_type}, "
                f"lanes={lanes}, density={density} ==="
            )

            decider = HighEnvDecider.configure(config=config, save_path=save_path)
            decider.run()


if __name__ == "__main__":
    main()
