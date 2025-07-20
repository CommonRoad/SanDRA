import os
from typing import List
from dataclasses import dataclass, field


@dataclass
class HighwayEnvConfig:
    seeds: List[int] = field(default_factory=lambda: [4213])

    simulation_frequency: int = 15
    policy_frequency: int = 5
    lanes_count: int = 4
    duration: float = 30  # [s]
    vehicles_density: float = 3.0

    maximum_lanelet_length: float = 1500.0

    action_input: bool = True
    save_frame: bool = True

    def get_save_folder(self, model_name: str, seed: int, use_sonia: bool = False) -> str:
        if use_sonia:
            return f"results-{self.action_input}-{model_name}-{self.lanes_count}-{self.vehicles_density}-{seed}-spot"
        else:
            return f"results-{self.action_input}-{model_name}-{self.lanes_count}-{self.vehicles_density}-{seed}"


@dataclass
class SanDRAConfiguration:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4o"  # "ft:gpt-4o-2024-08-06:tum::BsuinSqR" #"gpt-4o" # "qwen3:14b"  # systemctl stop ollama

    use_ollama: bool = False
    use_sonia: bool = False
    visualize_reach: bool = False

    a_lim = 0.2
    v_err = 0.1

    k = 3  # number of returned actions
    h: int = 15  # time horizon of decision-making
    dt: float = 0.2

    length: float = 5.0
    width: float = 2.0

    perception_radius: float = 100.0

    plot_limits: list = field(default_factory=lambda: [-6.36, 79.56, 4.07, 25.65])

    highway_env: HighwayEnvConfig = field(default_factory=HighwayEnvConfig)


COMMONROAD_REACH_SEMANTIC_ROOT = (
    "/home/sebastian/Documents/Uni/Sandra/commonroad-reach-semantic"
)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SUPPRESS_PLOTS = False
