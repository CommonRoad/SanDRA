
import numpy as np
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import TraceState

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
