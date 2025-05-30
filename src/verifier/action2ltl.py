from enum import Enum
from typing import Union

from src.actions import LongitudinalAction, LateralAction


class ActionLTL(Enum):
    ACCELERATE = "G (a > {A_LIM})"
    DECELERATE = "G (a < -{A_LIM})"
    KEEP = "G (a <= {A_LIM} & a >= -{A_LIM})"
    STOP = "FG (InStandstill)"

    CHANGE_LEFT = "FG (InLeftAdjacentLane)"
    CHANGE_RIGHT = "FG (InRightAdjacentLane)"
    FOLLOW_LANE = "G (InCurrentLane)"


    @classmethod
    def from_action(cls, action: Union[LongitudinalAction, LateralAction]) -> str:
        return f"LTL {cls[action.name].value}"