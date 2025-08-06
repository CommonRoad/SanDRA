from enum import Enum


class LongitudinalAction(Enum):
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    KEEP = "keep"
    STOP = "stop"
    UNKNOWN = "unknown"


class LateralAction(Enum):
    CHANGE_LEFT = "left"
    CHANGE_RIGHT = "right"
    FOLLOW_LANE = "follow_lane"
    UNKNOWN = "unknown"


Action = tuple[LongitudinalAction, LateralAction]
