from enum import Enum

from commonroad_reach_semantic.data_structure.config.semantic_configuration import SemanticConfiguration


class LongitudinalAction(Enum):
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    IDLE = "idle"


class LateralAction(Enum):
    LEFT = "left"
    RIGHT = "right"
    FORWARD = "forward"
    STOP = "stop"


Action = tuple[LongitudinalAction, LateralAction]


def set_speed(action: LongitudinalAction, semantic_config: SemanticConfiguration) -> SemanticConfiguration:
    if action == LongitudinalAction.ACCELERATE:
        semantic_config.vehicle.ego.a_lon_min = 0
    elif action == LongitudinalAction.DECELERATE:
        semantic_config.vehicle.ego.a_lon_max = 0
    return semantic_config


def get_ltl_formula(action: LateralAction) -> str:
    # TODO
    pass


def get_all_actions():
    longitudinal_values = [action.value for action in LongitudinalAction]
    lateral_values = [action.value for action in LateralAction]
    return longitudinal_values + lateral_values
