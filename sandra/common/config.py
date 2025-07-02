import os
from dataclasses import dataclass


@dataclass
class SanDRAConfiguration:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "qwen3:14b"
    use_ollama = True
    a_lim = 0.1

    h = 25  # time horizon of decision-making

    length = 4.5
    width = 2.0

    perception_radius = 100

    plot_limits = [-6.36, 79.56, 4.07, 25.65]
COMMONROAD_REACH_SEMANTIC_ROOT = (
    "/home/sebastian/Documents/Uni/GuidedResearch/Repos/commonroad-reach-semantic"
)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SUPPRESS_PLOTS = False
