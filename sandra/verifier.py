import enum
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, List

from sandra.actions import LongitudinalAction, LateralAction


class ActionLTL(Enum):
    ACCELERATE = "G (a > {A_LIM})"
    DECELERATE = "G (a < -{A_LIM})"
    KEEP = "G (a <= {A_LIM} & a >= -{A_LIM})"
    STOP = "FG (InStandstill)"

    CHANGE_LEFT = "FG (InLeftAdjacentLane)"
    CHANGE_RIGHT = "FG (InRightAdjacentLane)"
    FOLLOW_LANE = "G (InCurrentLane)"

    @classmethod
    def from_action(cls, action: Union["LongitudinalAction", "LateralAction"]) -> str:
        """Obtains LTL formula from the given action"""
        try:
            return f"LTL {cls[action.name].value}"
        except KeyError:
            raise ValueError(f"No LTL mapping for action: {action}")


class VerificationStatus(enum.Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"


class VerifierBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def verify(
        self,
        actions: List[Union[LongitudinalAction, LateralAction]],
        visualization=False,
    ) -> VerificationStatus:
        pass


class DummyVerifier(VerifierBase):
    def verify(
        self,
        actions: List[Union[LongitudinalAction, LateralAction]],
        visualization=False,
    ) -> VerificationStatus:
        return VerificationStatus.SAFE
