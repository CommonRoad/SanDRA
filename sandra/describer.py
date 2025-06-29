from abc import ABC, abstractmethod
from typing import Optional, Any, Literal, overload

from commonroad_clcs.clcs import CurvilinearCoordinateSystem
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
            else f"You are currently in an {scenario_type} scenario."
        )
        self.update(timestep=timestep)

    def update(self, timestep=None):
        if timestep is not None:
            self.timestep = timestep
        else:
            self.timestep = self.timestep + 1

    @staticmethod
    def velocity_descr(v: float, to_km=False) -> str:
        if to_km:
            v *= 3.6
            return f"{v:.1f} km/h"
        return f"{v:.1f} m/s"

    @staticmethod
    def acceleration_descr(a: float, to_km=False) -> str:
        if to_km:
            a *= 12960
            return f"{a:.1f} km/h²"
        return f"{a:.1f} m/s²"

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
        clcs: 'CurvilinearCoordinateSystem',
        direction="",
    ) -> str:
        points = [
            np.array(ego_position).reshape(2, 1),
            np.array(obstacle_position).reshape(2, 1)
        ]
        # Use the appropriate integer parameter here (e.g., 0)
        curvilinear_points = clcs.convert_list_of_points_to_curvilinear_coords(points, 0)
        ego_position_clcs, obstacle_position_clcs = curvilinear_points

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

        return "Here is an overview of your environment:\n" + "\n".join(filtered_parts) + "\n"

    def system_prompt(self) -> str:
        reminders = self._describe_reminders()
        reminder_description = "Keep these things in mind:\n"
        for reminder in reminders:
            reminder_description += f"  - {reminder}\n"

        role = f"{self.role}\n" if self.role else ""
        goal = f"{self.goal}\n" if self.goal else ""

        return (
            "You are driving a car and need to make a high-level driving decision.\n"
            f"{role}"
            f"{goal}"
            #f"{reminder_description}"
            f"{self._describe_schema()}"
        )
        #Keep these things in mind:
        #{reminder_description}
        #"""