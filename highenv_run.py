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

    # Create decider with seed 4213
    decider = HighEnvDecider.configure(config)

    # Run the decider (this likely starts your planning/interaction loop)
    decider.run()


if __name__ == "__main__":
    main()
