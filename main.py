from config import SaLaRAConfiguration
from src.decider import Decider

if __name__ == "__main__":
    config = SaLaRAConfiguration()
    scenario_path = ""
    save_path = scenario_path
    decider = Decider(scenario_path, config, save_path=save_path)
