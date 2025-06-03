import re
import time
from typing import Optional

import yaml
import logging

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.visualization.draw_params import LaneletNetworkParams, TrajectoryParams
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_reach.data_structure.reach.driving_corridor import DrivingCorridor
from matplotlib import pyplot as plt
import numpy as np
import os

# commonroad_reach
from commonroad_reach.utility.visualization import draw_driving_corridor_2d

# commonroad-reach-semantic
from commonroad_reach_semantic.data_structure.config.semantic_configuration_builder import (
    SemanticConfigurationBuilder,
)
from commonroad_reach_semantic.data_structure.environment_model.semantic_model import (
    SemanticModel,
)
from commonroad_reach_semantic.data_structure.model_checking.spot_interface import (
    SpotInterface,
)
from commonroad_reach_semantic.data_structure.reach.semantic_reach_interface import (
    SemanticReachableSetInterface,
)
from commonroad_reach_semantic.data_structure.rule.traffic_rule_interface import (
    TrafficRuleInterface,
)

# commonroad_dc
from commonroad_dc.geometry.geometry import CurvilinearCoordinateSystem

# commonroad_qp_planner
from commonroad_qp_planner.qp_planner import (
    QPPlanner,
    QPLongDesired,
    QPLongState,
    LongitudinalTrajectoryPlanningError,
    LateralTrajectoryPlanningError,
)
from commonroad_qp_planner.initialization import (
    create_optimization_configuration_vehicle,
)
from commonroad_qp_planner.constraints import LatConstraints, LonConstraints
from commonroad_qp_planner.utility.compute_constraints import (
    longitudinal_position_constraints,
    lateral_position_constraints,
    longitudinal_velocity_constraints,
)
from commonroad_qp_planner.utils import plot_result, plot_position_constraints
import xml.etree.ElementTree as ET

from src.config import COMMONROAD_REACH_SEMANTIC_ROOT, PROJECT_ROOT
from src.actions import set_speed, get_ltl_formula, Action, LongitudinalAction


def initialize_scenario_xml(xml_file_path: str, goal_lane_id: Optional[str]):
    def extract_initial_position(thing) -> tuple[float, float]:
        initial_state = thing.find("./initialState")
        position = initial_state.find("./position")
        point = position.find("./point")
        x = point.find("./x")
        y = point.find("./y")
        return float(x.text), float(y.text)

    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    planning_problem = root.find(".//planningProblem")
    goal_state = planning_problem.find("./goalState")
    position = goal_state.find("./position")
    initial_x, initial_y = extract_initial_position(planning_problem)
    pp_pos = np.array([initial_x, initial_y])

    if goal_lane_id:
        # Remove initial goal position
        polygon = position.find("polygon")
        if polygon is not None:
            position.remove(polygon)
        lanelet = position.find("lanelet")
        if lanelet is not None:
            position.remove(lanelet)

        # Create a new lanelet reference element
        lanelet_ref = ET.Element("lanelet")
        lanelet_ref.set("ref", f"{goal_lane_id}")

        # Add the lanelet reference to the position element
        position.append(lanelet_ref)

    # Remove ego vehicle
    cars = root.findall(".//dynamicObstacle[type='car']")
    for car in cars:
        a, b = extract_initial_position(car)
        car_pos = np.array([a, b])
        if np.linalg.norm(car_pos - pp_pos) < 0.1:
            root.remove(car)
            break

    # Remove pedestrians
    pedestrians = root.findall(".//dynamicObstacle[type='pedestrian']")
    for pedestrian in pedestrians:
        root.remove(pedestrian)
    tree.write(os.path.join(PROJECT_ROOT, "scenario.xml"))


def plan(
        formula: str,
        speed: LongitudinalAction,
        stop_at_lateral=False,
        stop_at_longitudinal=False,
        num_time_steps: int = 20,
) -> DrivingCorridor | DynamicObstacle:
    name_scenario = "scenario"
    spec = [formula]
    # ****************************************************
    # Reachability Analysis with semantic
    # ****************************************************
    # compute reachable sets and driving corridor
    path_root = COMMONROAD_REACH_SEMANTIC_ROOT
    config = SemanticConfigurationBuilder(path_root=path_root).build_configuration(
        name_scenario
    )

    config.general.path_scenarios = os.getcwd() + "/"
    config.general.path_scenario = os.getcwd() + "/" + name_scenario + ".xml"
    config.planning.reference_point = "REAR"
    config.planning.dt = 0.2
    config.planning.steps_computation = num_time_steps
    config = set_speed(speed, config)

    config.update()
    config.reachable_set.mode_computation = 8
    config.traffic_rule.activated_rules = spec  # ["LTL FG(InLanelet_1001)"]

    # initialize semantic model and traffic ruleslength
    semantic_model = SemanticModel(config)
    rule_interface = TrafficRuleInterface(config, semantic_model)
    reach_interface = SemanticReachableSetInterface(
        config, semantic_model, rule_interface
    )

    # ==== compute reachable sets using reachability interface
    config.update(scenario=config.scenario, planning_problem=config.planning_problem)

    reach_interface.reset(config=config)
    reach_interface.compute_reachable_sets()

    if config.reachable_set.mode_computation in [5, 6]:
        # only necessary for labeling reachable set
        spot_interface = SpotInterface(reach_interface, rule_interface)
        spot_interface.translate_ltl_formulas()
        spot_interface.translate_reachability_graph()
        spot_interface.check()

    longitudinal_driving_corridors = reach_interface.extract_driving_corridors(
        to_goal_region=False,
    )
    lon_dc = longitudinal_driving_corridors[0]

    if stop_at_longitudinal:
        return lon_dc
    # ****************************************************
    # QP Planner
    # ****************************************************
    # load qp YAML settings
    yaml_file = os.path.join(PROJECT_ROOT, "planner_config.yaml")
    with open(yaml_file, "r") as stream:
        try:
            settings_qp = yaml.load(stream, Loader=yaml.Loader)
            temp: dict = settings_qp["vehicle_settings"]
            temp_value = list(temp.values())[0]
            settings_qp["vehicle_settings"] = {
                config.planning_problem.planning_problem_id: temp_value
            }
            settings_qp["scenario_settings"] = {
                "draw": True,
                "scenario_name": str(config.scenario.scenario_id),
            }
        except yaml.YAMLError as exc:
            print(exc)

    # get scenario, planning problem, reference path, cosy from reach config
    scenario = config.scenario
    planning_problem = config.planning_problem
    reference_path = config.planning.reference_path
    lanelets_to_goal = config.planning.route.lanelet_ids

    curvilinear_cosy = CurvilinearCoordinateSystem(
        np.array(config.planning.CLCS.reference_path_original()),
        default_projection_domain_limit=30.0,
        eps=0.1,
        eps2=1e-4,
        resample=False,
    )

    # construct qp vehicle configuration
    configuration_qp = create_optimization_configuration_vehicle(
        scenario,
        planning_problem,
        settings_qp["vehicle_settings"],
        reference_path=reference_path,
        lanelets_leading_to_goal=lanelets_to_goal,
        cosy=curvilinear_cosy,
    )
    configuration_qp.a_max = 10
    # Initialize QP Planner
    qp_planner = QPPlanner(
        configuration_qp,
        num_planning_steps=config.planning.steps_computation,
        qp_long_parameters=settings_qp["qp_planner"]["longitudinal_parameters"],
        qp_lat_parameters=settings_qp["qp_planner"]["lateral_parameters"],
        verbose=True,
        logger_level=logging.INFO,
        failsafe=False,
        parameterization=False,
        solvers=settings_qp["qp_planner"]["solvers_to_use"],
    )

    qp_planner.reset(scenario)
    qp_planner.step(
        initial_state=planning_problem.initial_state,
        desired_velocity=QPPlanner.desired_speed_from_planning_problem(
            planning_problem
        ),
    )

    # Longitudinal Trajectory Planning
    # Derive LonConstraints from driving corridor
    s_min, s_max = longitudinal_position_constraints(lon_dc)
    v_min, v_max = longitudinal_velocity_constraints(lon_dc)
    c_tv_lon = LonConstraints.construct_constraints(
        s_min, s_max, s_min, s_max, v_max, v_min
    )

    # construct longitudinal reference
    x_ref = list()
    v_des = settings_qp["vehicle_settings"][planning_problem.planning_problem_id][
        "desired_speed"
    ]
    for i in range(len(s_min)):
        x_ref.append(QPLongState(s_min[i], v_des, 0.0, 0.0, 0.0))
    desired_lon_states = QPLongDesired(x_ref)

    # Plan longitudinal trajectory
    traj_lon, status = qp_planner.longitudinal_trajectory_planning(
        c_long=c_tv_lon, x_des=desired_lon_states
    )

    # Lateral Trajectory Planning
    # get longitudinal positions from planned lon trajectory
    traj_lon_positions = traj_lon.get_positions()[:, 0]

    if status == "optimal":
        try:
            lateral_driving_corridors = reach_interface.extract_driving_corridors(
                corridor_lon=lon_dc, list_p_lon=traj_lon_positions
            )
            lat_dc = lateral_driving_corridors[0]
        except:
            print("No lateral driving corridor found, use the longitudinal one")
            lat_dc = lon_dc
    else:
        raise LongitudinalTrajectoryPlanningError(
            f"<QPPlanner/_longitudinal_trajectory_planning> "
            f"failed, status: {status}"
        )

    if stop_at_lateral:
        return lat_dc
    # Derive LatConstraints
    d_min, d_max = lateral_position_constraints(
        lat_dc, lon_dc, traj_lon_positions, configuration_qp
    )
    c_tv_lat = LatConstraints.construct_constraints(d_min, d_max, d_min, d_max)

    # Plan lateral trajectory
    trajectory_cvln, status = qp_planner.lateral_trajectory_planning(traj_lon, c_tv_lat)
    if status == "optimal":
        pass
    else:
        raise LateralTrajectoryPlanningError(
            f"<QPPlanner/_longitudinal_trajectory_planning> "
            f"failed, status: {status}"
        )

    # Transform planned trajectory back to Cartesian
    trajectory_cartesian = qp_planner.transform_trajectory_to_cartesian_coordinates(
        trajectory_cvln
    )

    # ****************************************************
    # Visualization and Evaluation
    # ****************************************************
    # create ego vehicle object
    ego_vehicle = trajectory_cartesian.convert_to_cr_ego_vehicle(
        configuration_qp.width,
        configuration_qp.length,
        configuration_qp.wheelbase,
        configuration_qp.wb_ra,
        configuration_qp.vehicle_id,
    )

    # plot trajectory
    plot_result(scenario, ego_vehicle)

    # plot position constraints
    plot_position_constraints(trajectory_cvln, (s_min, s_max), (d_min, d_max))

    # plot full driving corridor with (x,y) positions of the planned trajectory
    trajectory_positions = np.asarray(
        [state.position for state in ego_vehicle.prediction.trajectory.state_list]
    )
    draw_driving_corridor_2d(lat_dc, 0, reach_interface, trajectory_positions)
    rnd = MPRenderer(
        figsize=(26, 18),
        focus_obstacle=ego_vehicle,
        plot_limits=[-20.0, 150.0, -20.0, 20.0],
    )
    params = LaneletNetworkParams()
    params.traffic_sign.draw_traffic_signs = True
    scenario.draw(rnd)
    params = TrajectoryParams()
    params.draw_continuous = True
    params.facecolor = "red"
    params.line_width = 3
    ego_vehicle.prediction.trajectory.draw(rnd, draw_params=params)
    ego_vehicle.draw(rnd)
    rnd.render(show=True)
    plt.show()
    return ego_vehicle


def is_drivable(
        scenario_path: str, action: Optional[Action], only_longitudinal=True
) -> Optional[DrivingCorridor]:
    if action:
        lon_action, lat_action = action
        formula = get_ltl_formula(lat_action)
    else:
        lon_action = LongitudinalAction.IDLE
        formula = "LTL true"

    def extract_number(s):
        match = re.search(r"_(\d+)$", s)
        if match:
            return match.group(1)
        return None

    initialize_scenario_xml(scenario_path, extract_number(formula))
    time.sleep(0.4)
    try:
        return plan(
            formula,
            lon_action,
            stop_at_lateral=False,
            stop_at_longitudinal=only_longitudinal,
        )
    except Exception as e:
        return None
