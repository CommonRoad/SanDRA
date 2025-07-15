import os
from typing import List
from dataclasses import dataclass, field


@dataclass
class HighwayEnvConfig:
    seeds: List[int] = field(default_factory=lambda: [4213])
    action_input: bool = True
    simulation_frequency: int = 15
    policy_frequency: int = 5
    lanes_count: int = 4
    duration: float = 30 # [s]
    vehicles_density: float = 2.0

@dataclass
class SanDRAConfiguration:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4o"
    use_ollama = False
    a_lim = 0.2
    v_err = 0.1

    k = 3   # number of returned actions
    h: int = 15  # time horizon of decision-making
    dt: float = 0.2

    length: float = 5.0
    width: float = 2.0

    perception_radius: float = 100.0

    plot_limits: list = field(default_factory=lambda: [-6.36, 79.56, 4.07, 25.65])

    highway_env: HighwayEnvConfig = field(default_factory=HighwayEnvConfig)

COMMONROAD_REACH_SEMANTIC_ROOT = (
    "/home/liny/repairverse/commonroad-reach-semantic"
)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SUPPRESS_PLOTS = False
