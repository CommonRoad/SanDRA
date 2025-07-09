import os
from dataclasses import dataclass


@dataclass
class SanDRAConfiguration:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4.1"  # "qwen3:14b"
    use_ollama = False  # True
    a_lim = 0.1

    h = 15  # time horizon of decision-making
    dt = 0.2

    length = 5.0  # 4.5
    width = 2.0

    perception_radius = 100

    plot_limits = [-6.36, 79.56, 4.07, 25.65]
COMMONROAD_REACH_SEMANTIC_ROOT = (
    "/home/sebastian/Documents/Uni/Sandra/commonroad-reach-semantic"
)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SUPPRESS_PLOTS = False
