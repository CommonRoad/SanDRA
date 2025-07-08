import logging
from abc import abstractmethod, ABC
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
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
from matplotlib import pyplot as plt

from sandra.common.config import SanDRAConfiguration


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
        self,
        config: SanDRAConfiguration,
        scenario: Scenario = None,
        planning_problem: PlanningProblem = None,
    ):
        super().__init__()

        # configurations
        self.sandra_config = config
        self.scenario = scenario
        self.planning_problem = planning_problem

        config_path = (
            Path(__file__).resolve().parents[2] / "config" / "reactive_planner.yaml"
        )
        self.config_planner = ReactivePlannerConfiguration.load(config_path)
        self.config_planner.update(scenario=scenario, planning_problem=planning_problem)

        # adaptive corridor sampling
        self.config_planner.sampling.sampling_method = 2
        self.config_planner.planning.time_steps_computation = self.sandra_config.h

        # fix the dimension
        self.config_planner.vehicle.length = config.length
        self.config_planner.vehicle.width = config.width

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

    def extract_desired_velocity(self, connected_component: ConnectedComponent):
        # get min and max values for each reachable set in the connected set
        min_max_array = np.asarray(
            [
                [reach_node.polygon_lon.v_min, reach_node.polygon_lon.v_max]
                for reach_node in connected_component.list_nodes_reach
            ]
        )
        min_connected_set = np.min(min_max_array[:, 0])
        max_connected_set = np.max(min_max_array[:, 1])
        return (max_connected_set + min_connected_set) / 2

    def plan(
        self, driving_corridor: Dict[int, ConnectedComponent]
    ) -> Optional[Trajectory]:
        # limit the sampling space
        self.planner.sampling_space.driving_corridor = driving_corridor
        desired_velocity = self.extract_desired_velocity(
            driving_corridor[self.sandra_config.h]
        )
        if desired_velocity < 0.1:
            # self.planner.config.sampling.t_min = self.sandra_config.h * self.scenario.dt
            self.planner.set_desired_velocity(
                desired_velocity=desired_velocity, stopping=True
            )
            # not working, no sample is found
            # self.planner.config.sampling.longitudinal_mode = "stopping"
        else:
            self.planner.set_desired_velocity(desired_velocity=desired_velocity)

        # planning for the current time step
        print("[Planner] Planning trajectory...")
        optimal = self.planner.plan()
        if optimal:
            self.trajectory = optimal[0]
            return self.trajectory
        else:
            print("[Planner] Planning failed to find an optimal trajectory.")
            return None

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
            plot_limits=self.sandra_config.plot_limits
        )


def compute_input_bounds():
    from vehiclemodels.vehicle_parameters import VehicleParameters
    from commonroad.common.solution import VehicleType
    from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping
    id_type_vehicle: int = 2
    vehicle_parameters: VehicleParameters = VehicleParameterMapping.from_vehicle_type(VehicleType(id_type_vehicle))
    delta_min: float = vehicle_parameters.steering.min
    delta_max: float = vehicle_parameters.steering.max
    a_max = 8.0
    a_min = -a_max
    return delta_min, delta_max, a_min, a_max


def make_demo(seed=4213):
    from sandra.commonroad.reach import ReachVerifier
    from sandra.common.config import SanDRAConfiguration
    from sandra.common.road_network import RoadNetwork, EgoLaneNetwork

    from sandra.highenv.highenv_scenario import HighwayEnvScenario
    config = SanDRAConfiguration()

    # get vehicle parameters from CommonRoad vehicle models given cr_vehicle_id
    delta_min, delta_max, a_min, a_max = compute_input_bounds()
    v_min = 0.0
    v_max = 30.0
    # a_max: float = self.reach_ver.reach_config.vehicle.ego.a_max
    # v_max: float = self.reach_ver.reach_config.vehicle.ego.v_max
    env_config = {
        "highway-v0": {
            "observation": {
                "type": "OccupancyGrid",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                "grid_step": [5, 5],
                "absolute": False
            },
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": (a_min, a_max),
                "steering_range": (delta_min, delta_max),
                "speed_range": (v_min, v_max),
            },
            "lanes_count": 4,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "duration": 30,
            "vehicles_density": 2.0,
            "show_trajectories": True,
            "render_agent": True,
            "scaling": 5,
            "initial_lane_id": None,
            "ego_spacing": 4,
            "simulation_frequency": 15,
            "policy_frequency": 15,
        }
    }

    scenario = HighwayEnvScenario(env_config, seed=seed)
    cr_scenario, _, cr_planning_problem = scenario.commonroad_representation

    road_network = RoadNetwork.from_lanelet_network_and_position(
        cr_scenario.lanelet_network,
        cr_planning_problem.initial_state.position,
        consider_reversed=True,
        consider_incoming=True,
    )

    ego_lane_network = EgoLaneNetwork.from_route_planner(
        cr_scenario.lanelet_network,
        cr_planning_problem,
        road_network,
    )
    reach_ver = ReachVerifier(cr_scenario, config, ego_lane_network=ego_lane_network, initial_state=cr_planning_problem.initial_state)
    return config, cr_scenario, cr_planning_problem, scenario, reach_ver

def update_demo(scenario, config, seed=4213):
    from sandra.commonroad.reach import ReachVerifier
    from sandra.common.config import SanDRAConfiguration
    from sandra.common.road_network import RoadNetwork, EgoLaneNetwork

    from sandra.highenv.highenv_scenario import HighwayEnvScenario
    scenario = HighwayEnvScenario(scenario._env, seed=seed)
    cr_scenario, _, cr_planning_problem = scenario.commonroad_representation

    road_network = RoadNetwork.from_lanelet_network_and_position(
        cr_scenario.lanelet_network,
        cr_planning_problem.initial_state.position,
        consider_reversed=True,
        consider_incoming=True,
    )

    ego_lane_network = EgoLaneNetwork.from_route_planner(
        cr_scenario.lanelet_network,
        cr_planning_problem,
        road_network,
    )
    reach_ver = ReachVerifier(cr_scenario, config, ego_lane_network=ego_lane_network, initial_state=cr_planning_problem.initial_state)
    return cr_scenario, cr_planning_problem, scenario, reach_ver


def run_demo(seed=4213):
    from sandra.actions import LongitudinalAction, LateralAction
    config, cr_scenario, cr_planning_problem, scenario, reach_ver = make_demo(seed)

    delta_min, delta_max, a_min, a_max = compute_input_bounds()
    simulation_length = 60
    replanning_frequency = 5
    current_ego_prediction = None

    def normalize(v, a, b):
        normalized_v = (v - a) / (b - a)
        return 2 * normalized_v - 1

    for i in range(simulation_length):
        if i % replanning_frequency == 0:
            if i > 0:
                cr_scenario, cr_planning_problem, scenario, reach_ver = update_demo(scenario, config, seed=seed)
            reach_ver.verify([LateralAction.CHANGE_LEFT])
            planner = ReactivePlanner(config, cr_scenario, cr_planning_problem)
            planner.reset(reach_ver.reach_config.planning.CLCS)
            driving_corridor = reach_ver.reach_interface.extract_driving_corridors(
                to_goal_region=False
            )[0]
            planner.plan(driving_corridor)
            current_ego_prediction = planner.ego_vehicle.prediction.trajectory.state_list[1:]

        ego_state = current_ego_prediction[i % replanning_frequency]
        action_first = -normalize(ego_state.steering_angle, delta_min, delta_max)
        action_second = normalize(ego_state.acceleration, a_min, a_max)
        action = action_second, action_first
        _ = scenario.step(action)
    scenario._env.close()


if __name__ == "__main__":
    run_demo()

