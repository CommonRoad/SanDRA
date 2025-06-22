import logging
from abc import abstractmethod, ABC
from typing import Dict, List
from pathlib import Path

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad_dc.geometry.geometry import CurvilinearCoordinateSystem

from commonroad_reach.data_structure.reach.driving_corridor import ConnectedComponent
import commonroad_reach.utility.visualization as util_visual
from commonroad_reach.data_structure.reach.reach_interface import ReachableSetInterface

import commonroad_rp.utility.logger as util_logger_rp
from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.reactive_planner import ReactivePlanner as CRReactivePlanner
from commonroad_rp.utility.visualization import visualize_planner_at_timestep


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
            Path(__file__).resolve().parents[2] / "config" / "reactive_planner.yaml"
        )
        self.config_planner = ReactivePlannerConfiguration.load(config_path)
        self.config_planner.update(scenario=scenario, planning_problem=planning_problem)

        # adaptive corridor sampling
        self.config_planner.sampling.sampling_method = 2

        self.planner = CRReactivePlanner(self.config_planner)
        self.trajectory = None

        # logger
        util_logger_rp.initialize_logger(self.config_planner)

    def reset(self, cosys: CurvilinearCoordinateSystem = None):
        if cosys:
            # todo: fix by aligning the clcs used in planner and reach
            self.planner.set_reference_path(
                coordinate_system=CurvilinearCoordinateSystem(cosys.reference_path())
            )

    @property
    def ego_vehicle(self):
        return self.planner.convert_state_list_to_commonroad_object(
            self.trajectory.state_list
        )

    def plan(self, driving_corridor: Dict[int, ConnectedComponent]) -> Trajectory:
        # limit the sampling space
        self.planner.sampling_space.set_corridor(driving_corridor)
        self.planner.set_desired_velocity(current_speed=self.planner.x_0.velocity)

        # planning for the current time step
        optimal = self.planner.plan()
        self.trajectory = optimal[0]
        return self.trajectory

    def visualize(
        self,
        driving_corridor: Dict[int, ConnectedComponent] = None,
        reach_interface: ReachableSetInterface = None,
    ):
        self.planner.config.debug.save_plots = True
        renderer = MPRenderer(figsize=(20, 10))
        if driving_corridor and reach_interface:
            util_visual.draw_driving_corridor_2d(
                driving_corridor, 0, reach_interface, rnd=renderer
            )
        visualize_planner_at_timestep(
            scenario=self.config_planner.scenario,
            planning_problem=self.config_planner.planning_problem,
            ego=self.ego_vehicle,
            traj_set=self.planner.stored_trajectories,
            ref_path=self.planner.reference_path,
            timestep=0,
            config=self.config_planner,
            rnd=renderer,
        )
