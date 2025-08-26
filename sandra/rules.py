from enum import Enum


class InterstateRule(Enum):
    # Safe distance to preceding vehicle
    RG_1 = ("The ego vehicle following vehicles within the same lane must "
            "maintain a safe distance to ensure collision freedom for all "
            "of them, even if one or several vehicles suddenly stop.")
    # Unnecessary braking
    RG_2 = "The ego vehicle is not allowed to brake abruptly without reason."
    # Maximum speed limit
    RG_3 = "The ego vehicle must not exceed the speed limit."