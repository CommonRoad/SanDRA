from enum import Enum


class InterstateRule(Enum):
    # Safe distance to preceding vehicle
    RG_1 = ("The distance to a vehicle ahead must generally be large enough "
            "that one can stop safely even if that vehicle brakes suddenly.")
    # Unnecessary braking
    RG_2 = "The ego vehicle is not allowed to brake abruptly without reason."
    # Maximum speed limit
    RG_3 = "The ego vehicle must not exceed the speed limit."