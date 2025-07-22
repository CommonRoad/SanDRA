"""
Standalone script to create HighEnvDecider and run it.
"""

from highway_env.vehicle.behavior import IDMVehicle

# IDMVehicle.COMFORT_ACC_MIN = -1
# IDMVehicle.ACC_MAX = 4
# IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN = 1.0
IDMVehicle.LANE_CHANGE_DELAY = 3
# IDMVehicle.TIME_WANTED = 1.0



from sandra.common.config import SanDRAConfiguration
from sandra.highenv.decider import HighEnvDecider
import matplotlib

print(matplotlib.get_backend())
matplotlib.use("TkAgg")


def main():
    config = SanDRAConfiguration()
    # config.use_sonia = False
    config.h = 8 # spot
    config.h = 15 # most-likely
    config.highway_env.lanes_count = 5

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
                config.model_name, seed, config.use_sonia
            )
            + "/evaluation.csv",
        )
        decider.run()


if __name__ == "__main__":
    main()
