"""
Standalone script to create HighEnvDecider and run it.
"""
from sandra.common.config import SanDRAConfiguration
from sandra.highenv.decider import HighEnvDecider
import matplotlib
print(matplotlib.get_backend())
matplotlib.use('TkAgg')

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
            save_path=f"results-{config.highway_env.action_input}-{config.model_name}-{config.highway_env.lanes_count}-{config.highway_env.vehicles_density}/run-{seed}.csv"
        )
        decider.run()


if __name__ == "__main__":
    main()
