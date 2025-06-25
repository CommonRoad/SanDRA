from enum import Enum


class LongitudinalAction(Enum):
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    KEEP = "keep"
    STOP = "stop"


class LateralAction(Enum):
    CHANGE_LEFT = "left"
    CHANGE_RIGHT = "right"
    FOLLOW_LANE = "follow_lane"


Action = tuple[LongitudinalAction, LateralAction]


def set_speed(action: LongitudinalAction, semantic_config):
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
