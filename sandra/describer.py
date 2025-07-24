import math
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Any, Literal, overload, Union, List

from commonroad.scenario.state import InitialState, CustomState, KSState
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from openai import BaseModel
import numpy as np

from sandra.actions import LateralAction, LongitudinalAction
from sandra.common.config import SanDRAConfiguration


class Thoughts(BaseModel):
    observation: list[str]
    conclusion: str
    model_config = {"extra": "forbid"}


class Action(BaseModel):
    lateral_action: Literal["placeholder"]
    longitudinal_action: Literal["placeholder"]
    model_config = {"extra": "forbid"}


class HighLevelDrivingDecision(BaseModel):
    thoughts: Thoughts
    action_ranking: list[Action]
    model_config = {"extra": "forbid"}


class DescriberBase(ABC):
    def __init__(
        self,
        timestep: int,
        config: SanDRAConfiguration,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        scenario_type: Optional[str] = None,
    ):
        self.timestep = timestep
        self.config = config
        self.role = "" if role is None else role
        self.goal = "" if goal is None else goal
        self.scenario_type = (
            ""
            if scenario_type is None
            else f"{scenario_type}"
        )
        self.update(timestep=timestep)

    def update(self, timestep=None):
        if timestep is not None:
            self.timestep = timestep
        else:
            self.timestep = self.timestep + 1

    @staticmethod
    def velocity_descr(state: Union[InitialState, CustomState, KSState] = None, velocity: float = None,  to_km=False) -> str:
        if velocity is not None:
            v = velocity
        elif state is not None:
            v = state.velocity
        else:
            raise ValueError("Either 'state' or 'velocity' must be provided.")
        if to_km:
            v *= 3.6
            return f"{v:.1f} km/h"
        return f"{v:.1f} m/s"

    @staticmethod
    def acceleration_descr(
            state:  Union[InitialState, CustomState, KSState] = None,
            acceleration: float = None,
            to_km: bool = False
    ) -> str:
        if acceleration is not None:
            a = acceleration
        elif state is not None:
            a = state.acceleration
        else:
            raise ValueError("Either 'state' or 'acceleration' must be provided.")
        if to_km:
            a *= 12960
            return f"{a:.1f} km/h²"
        return f"{a:.1f} m/s²"

    @staticmethod
    def orientation_descr(
            state: Union[InitialState, CustomState, KSState] = None,
            orientation: float = None,
            degrees: bool = False
    ) -> str:
        if orientation is not None:
            theta = orientation
        elif state is not None:
            theta = state.orientation
        else:
            raise ValueError("Either 'state' or 'orientation' must be provided.")
        # Normalize to [-pi, pi]
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        if degrees:
            theta_deg = theta * 180 / math.pi
            return f"{theta_deg:.1f}°"
        return f"{theta:.3f} rad"

    @staticmethod
    def steering_descr(
            state: Union[InitialState, CustomState, KSState] = None,
            steering_angle: float = None,
            degrees: bool = False,
            wheelbase: float = 2.578
    ) -> str:
        if steering_angle is not None:
            delta = steering_angle
        elif state is not None:
            if hasattr(state, "steering_angle"):
                delta = state.steering_angle
            elif hasattr(state, "yaw_rate"):
                delta = np.arctan2(wheelbase * state.yaw_rate, state.velocity)
            else:
                delta = 0.0
        else:
            raise ValueError("Either 'state' or 'steering_angle' must be provided.")

        # Normalize to [-pi, pi]
        delta = (delta + math.pi) % (2 * math.pi) - math.pi

        if degrees:
            delta_deg = delta * 180 / math.pi
            return f"{delta_deg:.1f}°"
        return f"{delta:.3f} rad"

    @staticmethod
    def angle_description(theta: float) -> str:
        if abs(0 - theta) < np.pi / 4:
            return "in front of"
        elif abs(np.pi / 2 - theta) < np.pi / 4:
            return "left of"
        elif abs(np.pi - theta) < np.pi / 4:
            return "behind"
        else:
            return "right of"

    @staticmethod
    def distance_description(
        ego_position: np.ndarray, obstacle_position: np.ndarray
    ) -> str:
        dist = np.linalg.norm(obstacle_position - ego_position)
        return f"{dist:.1f} meters"

    @staticmethod
    def distance_description_clcs(
        ego_position: np.ndarray,
        obstacle_position: np.ndarray,
        clcs: "CurvilinearCoordinateSystem",
        direction="",
    ) -> str:
        points = [
            np.array(ego_position).reshape(2, 1),
            np.array(obstacle_position).reshape(2, 1),
        ]
        # Use the appropriate integer parameter here (e.g., 0)
        curvilinear_points = clcs.convert_list_of_points_to_curvilinear_coords(
            points, 0
        )
        try:
            ego_position_clcs, obstacle_position_clcs = curvilinear_points

            if len(curvilinear_points) != 2:
                warnings.warn(f"Unexpected number of curvilinear points: {len(curvilinear_points)}")
                return "far away from you"

            s_dist = obstacle_position_clcs[0] - ego_position_clcs[0]
            d_dist = obstacle_position_clcs[1] - ego_position_clcs[1]
            if direction == "incoming":
                if d_dist > 0:
                    return f"{d_dist:.1f} meters right of you"
                else:
                    return f"{d_dist:.1f} meters left of you"
            else:
                if s_dist > 0:
                    return f"{s_dist:.1f} meters in front of you"
                else:
                    return f"{abs(s_dist):.1f} meters behind you"
        except Exception as e:
            print(f"[Error] Failed to compute curvilinear distance: {e}")
            return "far away from you"

    @abstractmethod
    def _describe_traffic_signs(self) -> str:
        pass

    @abstractmethod
    def _describe_traffic_lights(self) -> str:
        pass

    @abstractmethod
    def _describe_obstacles(self) -> str:
        pass

    @abstractmethod
    def _describe_ego_state(self) -> str:
        pass

    @abstractmethod
    def _describe_schema(self) -> str:
        pass

    @abstractmethod
    def _describe_reminders(self) -> list[str]:
        pass

    @abstractmethod
    def _get_available_actions(
        self,
    ) -> tuple[list[LateralAction], list[LongitudinalAction]]:
        pass

    def get_available_actions(self) -> tuple[list[str], list[str]]:
        laterals, longitudinals = self._get_available_actions()
        return [x.value for x in laterals], [x.value for x in longitudinals]

    def schema(self) -> dict[str, Any]:
        laterals, longitudinals = self.get_available_actions()
        schema_dict = HighLevelDrivingDecision.model_json_schema()
        lateral_action = schema_dict["$defs"]["Action"]["properties"]["lateral_action"]
        lateral_action["enum"] = laterals
        if len(laterals) == 1:
            lateral_action["const"] = laterals[0]
        else:
            lateral_action.pop("const", None)
        longitudinal_action = schema_dict["$defs"]["Action"]["properties"][
            "longitudinal_action"
        ]
        longitudinal_action["enum"] = longitudinals
        if len(longitudinals) == 1:
            longitudinal_action["const"] = longitudinals[0]
        else:
            longitudinal_action.pop("const", None)

        return schema_dict

    def user_prompt(self) -> str:
        parts = [
            self._describe_ego_state(),
            self._describe_traffic_signs(),
            self._describe_traffic_lights(),
            self._describe_obstacles(),
        ]

        # Filter out None or empty strings (including whitespace-only)
        filtered_parts = [part.strip() for part in parts if part and part.strip()]

        return (
            "Here is an overview of your environment:\n"
            + "\n".join(filtered_parts)
            + "\n"
        )

    def system_prompt(self, past_actions: List[List[Union[LongitudinalAction, LateralAction]]] = None) -> str:
        # reminders = self._describe_reminders()
        # reminder_description = "Keep these things in mind:\n"
        # for reminder in reminders:
        #     reminder_description += f"  - {reminder}\n"

        role = f"{self.role}\n" if self.role else ""
        goal = f"{self.goal}\n" if self.goal else ""

        system_prompt = (
            "You are driving a car and need to make a high-level driving decision.\n"
            f"{role}"
            f"{goal}"
            # f"{reminder_description}"
            f"{self._describe_schema()}"
            f"{self._describe_past_actions(past_actions)}"
        )
        if self.config.use_ollama:
            system_prompt += f"\n /no_think"
        # Keep these things in mind:
        # {reminder_description}
        # """

    @staticmethod
    def _describe_past_actions(
            past_actions: List[List[Union[LongitudinalAction, LateralAction]]]
    ) -> str:
        """
        Returns a string description of the past action pairs.
        """
        if not past_actions:
            return "No past actions recorded. "

        pair_strs = []
        for pair in past_actions:
            if len(pair) != 2:
                pair_strs.append("(Invalid action pair)")
                continue

            a1, a2 = pair

            # Get readable names
            n1 = a1.name if hasattr(a1, "name") else str(a1)
            n2 = a2.name if hasattr(a2, "name") else str(a2)

            pair_strs.append(f"({n1}, {n2})")

        actions_str = "; ".join(pair_strs)
        return f"The past {len(past_actions)} action pairs are: {actions_str}. "
