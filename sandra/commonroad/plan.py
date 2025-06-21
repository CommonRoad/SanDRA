from abc import abstractmethod, ABC
from typing import Dict, List
from pathlib import Path

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.scenario import Scenario
from commonroad_clcs.pycrccosy import CurvilinearCoordinateSystem
from commonroad_reach.data_structure.reach.driving_corridor import ConnectedComponent
from commonroad_reach.pycrreach import ReachNode

from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.reactive_planner import ReactivePlanner as CRReactivePlanner


class PlannerBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def plan(self, driving_corridor: Dict[int, ConnectedComponent]) -> Trajectory:
        """
        Given a sequence of reachable sets, returns a planned trajectory.
        """
        pass


class ReactivePlanner(PlannerBase):
    def __init__(
        self, scenario: Scenario = None, planning_problem: PlanningProblem = None
    ):
        super().__init__()

        # configurations
        config_path = (
            Path(__file__).resolve().parents[3] / "config" / "reactive_planner.yaml"
        )
        self.config_planner = ReactivePlannerConfiguration.load(config_path)
        self.config_planner.update(scenario=scenario, planning_problem=planning_problem)

        self.planner = CRReactivePlanner(self.config_planner)

    def reset(self, cosys: CurvilinearCoordinateSystem = None):
        if cosys:
            self.planner.set_reference_path(cosys)

    def plan(self, driving_corridor: Dict[int, ConnectedComponent]) -> Trajectory:
        # limit the sampling space
        self.planner.sampling_space.set_corridor(driving_corridor)
        self.planner.set_desired_velocity(current_speed=self.planner.x_0.velocity)
        optimal = self.planner.plan()
        return optimal[0]
