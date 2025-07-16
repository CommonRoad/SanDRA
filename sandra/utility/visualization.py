from enum import Enum
from typing import Optional, List, Tuple
import os

import numpy as np
from commonroad_reach.data_structure.reach.reach_interface import ReachableSetInterface
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
from pathlib import Path

from commonroad.scenario.trajectory import Trajectory
from commonroad.visualization.draw_params import (
    DynamicObstacleParams,
    OccupancyParams,
    PlanningProblemParams,
)
from commonroad.geometry.shape import Rectangle, Polygon

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_params import (
    MPDrawParams,
    LaneletNetworkParams,
    TrajectoryParams,
    ShapeParams,
)
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad_reach.utility import coordinate_system as util_coordinate_system

from sandra.common.config import SUPPRESS_PLOTS, SanDRAConfiguration
from sandra.common.road_network import EgoLaneNetwork, RoadNetwork


class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMred = (227 / 255, 27 / 255, 35 / 255)
    TUMdarkred = (139 / 255, 0, 0)
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMgray = (156 / 255, 157 / 255, 159 / 255)
    TUMdarkgray = (88 / 255, 88 / 255, 99 / 255)
    TUMorange = (227 / 255, 114 / 255, 34 / 255)
    TUMdarkblue = (10 / 255, 45 / 255, 87 / 255)
    TUMwhite = (1, 1, 1)
    TUMblack = (0, 0, 0)
    TUMlightgray = (217 / 255, 218 / 255, 219 / 255)
    TUMyellow = (254 / 255, 215 / 255, 2 / 255)


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


def draw_scenario_paper(
    scenario: Scenario,
    planning_problem: PlanningProblem,
    step: int,
    config: SanDRAConfiguration,
    ego_vehicle: Optional[DynamicObstacle] = None,
    draw_planning_problem: bool = False,
    plot_limits: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    rotate_deg: float = 0.0,
) -> None:
    rnd = MPRenderer(figsize=(20, 10))

    rnd.plot_limits = plot_limits

    # scenario params
    scenario_params = MPDrawParams()
    scenario_params.time_begin = step
    scenario_params.time_end = config.h

    scenario_params.traffic_light.draw_traffic_lights = True
    scenario_params.dynamic_obstacle.draw_icon = True
    scenario_params.dynamic_obstacle.trajectory.draw_trajectory = False
    scenario_params.dynamic_obstacle.occupancy.draw_occupancies = True
    scenario_params.dynamic_obstacle.occupancy.shape.facecolor = TUMcolor.TUMgray
    scenario_params.dynamic_obstacle.occupancy.shape.edgecolor = TUMcolor.TUMblack
    scenario_params.dynamic_obstacle.occupancy.shape.opacity = 0.25
    scenario_params.dynamic_obstacle.vehicle_shape.facecolor = TUMcolor.TUMgray
    scenario_params.dynamic_obstacle.vehicle_shape.edgecolor = TUMcolor.TUMblack

    # draw scenario
    scenario.draw(rnd, draw_params=scenario_params)

    # draw planning problem
    if draw_planning_problem:
        pp_params = PlanningProblemParams()
        pp_params.initial_state.state.draw_arrow = False
        pp_params.initial_state.state.radius = 0.4
        pp_params.initial_state.state.facecolor = TUMcolor.TUMblue
        pp_params.initial_state.state.edgecolor = TUMcolor.TUMdarkblue
        planning_problem.draw(rnd, draw_params=pp_params)
    else:

        # ego params
        ego_params = DynamicObstacleParams()
        ego_params.time_begin = step
        ego_params.trajectory.draw_trajectory = False
        ego_params.vehicle_shape.occupancy.shape.facecolor = TUMcolor.TUMblue
        ego_params.vehicle_shape.occupancy.shape.edgecolor = TUMcolor.TUMdarkblue
        ego_params.draw_icon = True
        ego_params.vehicle_shape.direction.zorder = 55
        ego_params.vehicle_shape.occupancy.shape.zorder = 55

        # draw ego
        if ego_vehicle is None:
            # draw ego vehicle at initial state only
            ego_shape = Rectangle(config.length, config.width)
            pred = TrajectoryPrediction(
                Trajectory(
                    initial_time_step=planning_problem.initial_state.time_step,
                    state_list=[planning_problem.initial_state],
                ),
                ego_shape,
            )
            ego = DynamicObstacle(
                42, ObstacleType.CAR, ego_shape, planning_problem.initial_state, pred
            )
            ego.draw(rnd, draw_params=ego_params)
        else:
            # draw ego vehicle from trajectory
            ego_vehicle.draw(rnd, draw_params=ego_params)

            # draw ego trajectory occupancies
            occ_params = OccupancyParams()
            occ_params.shape.facecolor = "#E37222"
            occ_params.shape.edgecolor = "#9C4100"
            occ_params.shape.opacity = 0.25
            i = 0
            for occ in ego_vehicle.prediction.occupancy_set:
                if i >= 23:
                    occ_params.shape.facecolor = "#a2ad00"
                occ.draw(rnd, draw_params=occ_params)
                i += 1

    # render
    rnd.render()

    # 🔄 Apply rotation if requested
    if rotate_deg != 0:
        ax = plt.gca()
        center_x = 0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1])
        center_y = 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
        rotation = Affine2D().rotate_deg_around(center_x, center_y, rotate_deg)
        for artist in ax.get_children():
            artist.set_transform(rotation + artist.get_transform())

    plt.axis("off")
    if not output_path:
        output_path = Path(__file__).resolve().parents[2] / "output"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        f"{output_path}/{str(scenario.scenario_id)}/scenario.svg",
        format="svg",
        dpi=100,
        bbox_inches="tight",
    )

    plt.show()


def plot_reachable_sets(
    reach_interface: ReachableSetInterface,
    driving_corridor=None,
    step_start: int = 0,
    step_end: int = 0,
    plot_limits: List = None,
    path_output: str = None,
):
    """
    Plots scenario with computed reachable sets.
    """
    config = reach_interface.config
    scenario = config.scenario

    path_output = path_output or Path(__file__).resolve().parents[2] / "output"
    os.makedirs(path_output, exist_ok=True)

    step_start = step_start or reach_interface.step_start
    step_end = step_end or reach_interface.step_end

    steps = [step_start] + list(range(step_start, step_end + 1))

    rnd = MPRenderer(figsize=(20, 10))
    rnd.plot_limits = plot_limits

    # scenario params
    scenario_params = MPDrawParams()
    scenario_params.time_begin = step_start
    scenario_params.time_end = step_end

    scenario_params.traffic_light.draw_traffic_lights = True
    scenario_params.dynamic_obstacle.draw_icon = True
    scenario_params.dynamic_obstacle.trajectory.draw_trajectory = False
    scenario_params.dynamic_obstacle.occupancy.draw_occupancies = True
    scenario_params.dynamic_obstacle.occupancy.shape.facecolor = TUMcolor.TUMgray
    scenario_params.dynamic_obstacle.occupancy.shape.edgecolor = TUMcolor.TUMblack
    scenario_params.dynamic_obstacle.occupancy.shape.opacity = 0.25
    scenario_params.dynamic_obstacle.vehicle_shape.facecolor = TUMcolor.TUMgray
    scenario_params.dynamic_obstacle.vehicle_shape.edgecolor = TUMcolor.TUMblack

    reach_params = MPDrawParams()
    reach_params.shape.facecolor = TUMcolor.TUMyellow
    reach_params.shape.edgecolor = TUMcolor.TUMblack

    # draw scenario
    scenario.draw(rnd, draw_params=scenario_params)

    for step in steps:
        if driving_corridor:
            list_nodes = driving_corridor.dict_step_to_cc[step].list_nodes_reach
        else:
            list_nodes = reach_interface.reachable_set_at_step(step)
        # Draw reachable sets
        for node in list_nodes:
            position_rectangle = node.position_rectangle
            list_polygons_cart = util_coordinate_system.convert_to_cartesian_polygons(
                position_rectangle, config.planning.CLCS, True
            )
            for polygon in list_polygons_cart:
                Polygon(vertices=np.array(polygon.vertices)).draw(rnd, reach_params)

    rnd.render()
    plt.axis("off")

    plt.savefig(
        f"{path_output}/{str(scenario.scenario_id)}/reach.svg",
        format="svg",
        dpi=100,
        bbox_inches="tight",
    )

    plt.show()
