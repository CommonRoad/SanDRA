from enum import Enum


class InterstateRule(Enum):
    # Safe distance to preceding vehicle
    RG_1 = ("The ego vehicle must keep a safe distance from vehicles "
            "ahead in the same lane to prevent collisions.")
    # Unnecessary braking
    RG_2 = "The ego vehicle is not allowed to brake abruptly without reason."
    # Maximum speed limit
    RG_3 = ("The ego vehicle must not exceed lane limits, type limits, safe "
            "speed for unseen vehicles, or a speed allowing safe reaction.")