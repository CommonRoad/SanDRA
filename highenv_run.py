"""
Standalone script to create HighEnvDecider and run it.
"""

from highway_env.vehicle.behavior import IDMVehicle

# IDMVehicle.COMFORT_ACC_MIN = -1
# IDMVehicle.ACC_MAX = 4
# IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN = 1.0
IDMVehicle.LANE_CHANGE_DELAY = 3
# IDMVehicle.TIME_WANTED = 1.0



from sandra.config import SanDRAConfiguration
from sandra.highenv.decider import HighEnvDecider
import matplotlib

print(matplotlib.get_backend())
matplotlib.use("TkAgg")


def main():
    config = SanDRAConfiguration()

    # 1. whether we use rules in the reachable sets computation or not?
    config.use_rules_in_reach = True

    # 2. whether we use rules (natural language) in the prompt?
    config.use_rules_in_prompt = True

    config.visualize_reach = False

    # 3. whether we use set-based prediction or most-likely trajectory?
    # ----- set-based
    config.use_sonia = True
    config.h = 8 # spot

    # ----- most likely
    # config.use_sonia = False
    # config.h = 15 # most-likely

    # 4. scenario set up
    # ----- setting 1
    config.highway_env.lanes_count = 4
    config.highway_env.vehicles_density = 2.0

    # # ----- setting 2
    # config.highway_env.lanes_count = 4
    # config.highway_env.vehicles_density = 3.0
    #
    # # ----- setting 3
    # config.highway_env.lanes_count = 5
    # config.highway_env.vehicles_density = 3.0

    seeds = [
        5838,
        2421,
        7294,
        9650,
        4176,
        6382,
        8765,
        1348, # initial spot fail
        4213,
        2572, # initial spot fail
        # 5678, # collision
        # 8587, # -
        # 512,
        # 7523,
        # 6321,
        # 5214,
        # 31,
    ]

    for seed in seeds:
        config.highway_env.seeds = [seed]
        decider = HighEnvDecider.configure(
            config=config,
            save_path=config.highway_env.get_save_folder(
                config.model_name, seed, config.use_sonia, config.use_rules_in_prompt, config.use_rules_in_reach
            )
            + "/evaluation.csv",
        )
        decider.run()


if __name__ == "__main__":
    main()
