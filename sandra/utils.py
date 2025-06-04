import copy
import os

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import TraceState
from commonroad.scenario.trajectory import Trajectory
from commonroad.visualization.draw_params import MPDrawParams, LaneletNetworkParams, TrajectoryParams, ShapeParams
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_reach_semantic.data_structure.reach.semantic_reach_interface import SemanticReachableSetInterface
from commonroad_reach_semantic.utility import visualization as util_visual

from sandra.config import SUPPRESS_PLOTS

if SUPPRESS_PLOTS:
    import matplotlib
    matplotlib.use('Agg')


def plot_reachable_set(reach_interface: SemanticReachableSetInterface):
    config = reach_interface.config
    config.general.path_output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output"
    )
    semantic_model = reach_interface.semantic_model
    # ==== plot computation results
    if config.reachable_set.mode_computation in [5, 6]:
        node_to_group = util_visual.groups_from_propositions(
            reach_interface._reach.labeler.reachable_set_to_propositions
        )
    else:
        node_to_group = util_visual.groups_from_states(
            reach_interface._reach.reachable_set_to_label
        )

    util_visual.plot_reach_graph(reach_interface, node_to_group=node_to_group)
    util_visual.plot_scenario_with_regions(semantic_model, "CVLN")
    util_visual.plot_scenario_with_reachable_sets(reach_interface, save_gif=True)


def extract_scenario_and_planning_problem(absolute_scenario_path: str) -> tuple[Scenario, PlanningProblem]:
    scenario, planning_problem_set = CommonRoadFileReader(absolute_scenario_path).open(
        True
    )
    planning_problem = copy.deepcopy(
        list(planning_problem_set.planning_problem_dict.values())[0]
    )
    return scenario, planning_problem


def extract_ego_vehicle(scenario: Scenario, planning_problem: PlanningProblem) -> DynamicObstacle:
    ego_vehicle = None
    for vehicle in scenario.dynamic_obstacles:
        diff: np.ndarray = (
            vehicle.initial_state.position - planning_problem.initial_state.position
        )
        if np.linalg.norm(diff) < 0.1:
            ego_vehicle = vehicle
    return ego_vehicle


def find_lanelet_id_from_state(state: TraceState, lanelet_network: LaneletNetwork) -> int:
    try:
        return lanelet_network.find_most_likely_lanelet_by_state(
            [state]
        )[0]
    except IndexError:
        return -1


def plot_scenario(scenario: Scenario, planning_problem: PlanningProblem, plot_limits=None, save_path: str = None):
    rnd = MPRenderer(figsize=(12, 8), plot_limits=plot_limits)
    params = MPDrawParams()
    params.lanelet_network.traffic_sign.draw_traffic_signs = True
    scenario.draw(rnd, draw_params=params)
    planning_problem.draw(rnd)
    rnd.render(show=True, filename=save_path)


def plot_predicted_trajectory(
    scenario: Scenario, vehicle: DynamicObstacle, save_path: str = None
):
    assert isinstance(
        vehicle.prediction, TrajectoryPrediction
    ), "Can not plot a prediction which is not a TrajectoryPrediction object."
    rnd = MPRenderer(figsize=(12, 8), focus_obstacle=vehicle, plot_limits=[-30, 30, -30, 30])
    params = LaneletNetworkParams()
    params.traffic_sign.draw_traffic_signs = True
    scenario.lanelet_network.draw(rnd, draw_params=params)
    vehicle.draw(rnd)
    params = TrajectoryParams()
    params.draw_continuous = True
    params.facecolor = "red"
    params.line_width = 0.7
    vehicle.prediction.trajectory.draw(rnd, draw_params=params)
    rnd.render(show=True, filename=save_path)


def plot_lanelet(lanelet: Lanelet, lanelet_network: LaneletNetwork, save_path: str = None):
    # draw network
    rnd = MPRenderer(figsize=(12, 8))
    params = LaneletNetworkParams()
    params.traffic_sign.draw_traffic_signs = True
    lanelet_network.draw(rnd, draw_params=params)

    # draw lanelet center vertices
    params = ShapeParams()
    params.opacity = 1.0
    params.edgecolor = "red"
    params.linewidth = 0.7

    rnd.draw_polygon(lanelet.center_vertices, params)

    if lanelet.successor:
        for succ in lanelet.successor:
            ll = lanelet_network.find_lanelet_by_id(succ)
            rnd.draw_polygon(ll.center_vertices, params)

    if lanelet.predecessor:
        for pred in lanelet.predecessor:
            ll = lanelet_network.find_lanelet_by_id(pred)
            rnd.draw_polygon(ll.center_vertices, params)
    rnd.render(show=True, filename=save_path)


def calculate_relative_orientation(ego_direction: np.ndarray, other_direction: np.ndarray) -> float:
    """
    Calculates the angle in radians between ego direction and other direction.
    Front is 0 PI
    Left is PI/2
    Back is PI
    Right is 3*PI/2
    """
    ego_direction /= np.linalg.norm(ego_direction)
    other_direction /= np.linalg.norm(other_direction)
    cos_angle = np.dot(ego_direction, other_direction)
    cross_product = np.cross(ego_direction, other_direction)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    if cross_product < 0:
        angle = 2 * np.pi - angle
    return angle
