import os
from dataclasses import dataclass


@dataclass
class SaLaRAConfiguration:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4.1"

    a_lim = 1.0

COMMONROAD_REACH_SEMANTIC_ROOT = "/home/sebastian/Documents/Uni/GuidedResearch/Repos/commonroad-reach-semantic"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPRESS_PLOTS = False
