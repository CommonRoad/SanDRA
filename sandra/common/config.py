import os
from dataclasses import dataclass


@dataclass
class SanDRAConfiguration:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4o"
    use_ollama = False
    a_lim = 0.2
    v_err = 0.1

    m = 3   # number of returned actions

    h = 25  # time horizon of decision-making

    length = 4.5
    width = 2.0

    perception_radius = 100

    plot_limits = [-6.36, 79.56, 4.07, 25.65]


COMMONROAD_REACH_SEMANTIC_ROOT = (
    "/home/liny/repairverse/commonroad-reach-semantic"
)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SUPPRESS_PLOTS = False
