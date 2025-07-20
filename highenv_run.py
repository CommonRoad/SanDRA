"""
Standalone script to create HighEnvDecider and run it.
"""

from sandra.common.config import SanDRAConfiguration
from sandra.highenv.decider import HighEnvDecider
import matplotlib

print(matplotlib.get_backend())
matplotlib.use("TkAgg")


def main():
    config = SanDRAConfiguration()

    seeds = [
        5838,
        2421,
        7294,
        9650,
        4176,
        6382,
        8765,
        1348,
        4213,
        2572,
        5678,
        8587,
        512,
        7523,
        6321,
        5214,
        31,
    ]

    for seed in seeds:
        config.highway_env.seeds = [seed]
        decider = HighEnvDecider.configure(
            config=config,
            save_path=config.highway_env.get_save_folder(config.model_name, seed, config.use_sonia)
            + "/evaluation.csv",
        )
        decider.run()


if __name__ == "__main__":
    main()
