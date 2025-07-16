import numpy as np
from commonroad.common.solution import VehicleType
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import TraceState
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping
from vehiclemodels.vehicle_parameters import VehicleParameters


def find_lanelet_id_from_state(
    state: TraceState, lanelet_network: LaneletNetwork
) -> int:
    try:
        return lanelet_network.find_most_likely_lanelet_by_state([state])[0]
    except IndexError:
        return -1


def extract_ego_vehicle(
    scenario: Scenario, planning_problem: PlanningProblem
) -> DynamicObstacle:
    ego_vehicle = None
    for vehicle in scenario.dynamic_obstacles:
        diff: np.ndarray = (
            vehicle.initial_state.position - planning_problem.initial_state.position
        )
        if np.linalg.norm(diff) < 0.1:
            ego_vehicle = vehicle
    return ego_vehicle


def calculate_relative_orientation(
    ego_direction: np.ndarray, other_direction: np.ndarray
) -> float:
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


def get_input_bounds(
    vehicle_type: int = 2, a_max: float = 8.0, v_max: float = 30.0
) -> dict[str, float]:
    vehicle_parameters: VehicleParameters = VehicleParameterMapping.from_vehicle_type(
        VehicleType(vehicle_type)
    )
    return {
        "delta_min": vehicle_parameters.steering.min,
        "delta_max": vehicle_parameters.steering.max,
        "a_max": a_max,
        "a_min": -a_max,
        "v_max": v_max,
        "v_min": 0.0,
    }
