import copy
import os
from enum import Enum

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import TraceState
from commonroad.visualization.draw_params import (
    MPDrawParams,
    LaneletNetworkParams,
    TrajectoryParams,
    ShapeParams,
)
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt

from sandra.common.config import SUPPRESS_PLOTS
from sandra.common.road_network import EgoLaneNetwork, RoadNetwork


class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMred = (227 / 255, 27 / 255, 35 / 255)
    TUMdarkred = (139 / 255, 0, 0)
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMgray = (156 / 255, 157 / 255, 159 / 255)
    TUMdarkgray = (88 / 255, 88 / 255, 99 / 255)
    TUMorange = (227 / 255, 114 / 255, 34 / 255)
    TUMdarkblue = (0, 82 / 255, 147 / 255)
    TUMwhite = (1, 1, 1)
    TUMblack = (0, 0, 0)
    TUMlightgray = (217 / 255, 218 / 255, 219 / 255)


if SUPPRESS_PLOTS:
    import matplotlib

    matplotlib.use("Agg")


def plot_reachable_set(reach_interface):
    from commonroad_reach_semantic.utility import visualization as util_visual

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


def plot_scenario(
    scenario: Scenario,
    planning_problem: PlanningProblem,
    plot_limits=None,
    save_path: str = None,
):
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
    rnd = MPRenderer(
        figsize=(12, 8), focus_obstacle=vehicle, plot_limits=[-30, 30, -30, 30]
    )
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


def plot_lanelet(
    lanelet: Lanelet, lanelet_network: LaneletNetwork, save_path: str = None
):
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


def plot_road_network(
    road_network: RoadNetwork,
    ego_lane_network: EgoLaneNetwork = None,
    save_path: str = None,
):
    """
    Plot the road network with optional highlighting of the ego vehicle's lane and adjacent lanes.

    Args:
        road_network (RoadNetwork): The full road network to be visualized.
        ego_lane_network (EgoLaneNetwork, optional): The ego lane and its adjacent lanes to be highlighted.
        save_path (str, optional): Path to save the rendered figure. If None, the figure is only displayed.
    """
    rnd = MPRenderer(figsize=(12, 8))
    params = ShapeParams()
    params.opacity = 0.5
    params.facecolor = TUMcolor.TUMgray
    params.edgecolor = TUMcolor.TUMdarkgray
    params.linewidth = 0.7
    # Draw all lanes in the road network
    for lane in road_network.lanes:
        for lanelet in lane.lanelets:
            rnd.draw_polygon(lanelet.polygon.vertices, params)

    if ego_lane_network:
        # Set visual parameters for ego lane and its adjacent lanes
        params.opacity = 0.5
        params.facecolor = TUMcolor.TUMdarkblue

        # Draw ego lane
        for lanelet in ego_lane_network.lane.lanelets:
            rnd.draw_polygon(lanelet.polygon.vertices, params)

        # Reset color for adjacent lanes
        params.facecolor = TUMcolor.TUMgreen
        params.opacity = 0.2

        # Draw left adjacent lanes if they exist
        if ego_lane_network.lane_left_adjacent:
            for lane_left in ego_lane_network.lane_left_adjacent:
                for lanelet in lane_left.lanelets:
                    rnd.draw_polygon(lanelet.polygon.vertices, params)

        # Draw right adjacent lanes if they exist
        if ego_lane_network.lane_right_adjacent:
            for lane_right in ego_lane_network.lane_right_adjacent:
                for lanelet in lane_right.lanelets:
                    rnd.draw_polygon(lanelet.polygon.vertices, params)

        # Reset color for incoming lanes
        params.facecolor = TUMcolor.TUMwhite
        params.opacity = 0.2
        if ego_lane_network.lane_incoming_left:
            for lane in ego_lane_network.lane_incoming_left:
                for lanelet in lane.lanelets:
                    rnd.draw_polygon(lanelet.polygon.vertices, params)
        if ego_lane_network.lane_incoming_right:
            for lane in ego_lane_network.lane_incoming_right:
                for lanelet in lane.lanelets:
                    rnd.draw_polygon(lanelet.polygon.vertices, params)

        # Reset color for adjacent lanes
        params.facecolor = TUMcolor.TUMorange
        params.opacity = 0.2

        # Draw left reversed adjacent lanes if they exist
        if ego_lane_network.lane_left_reversed:
            for lane_left in ego_lane_network.lane_left_reversed:
                for lanelet in lane_left.lanelets:
                    rnd.draw_polygon(lanelet.polygon.vertices, params)

        # Draw right reversed adjacent lanes if they exist
        if ego_lane_network.lane_right_reversed:
            for lane_right in ego_lane_network.lane_right_reversed:
                for lanelet in lane_right.lanelets:
                    rnd.draw_polygon(lanelet.polygon.vertices, params)

    # Render and optionally save the figure
    rnd.render(show=True, filename=save_path)
    plt.show()
