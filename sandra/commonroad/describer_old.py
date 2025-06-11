from enum import Enum

import numpy as np
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.measure import TTC


class GoalManeuver(Enum):
    LeftTurn = "left"
    RightTurn = "right"
    Straight = "straight"


# define actions
stop = '"stop" # decelerate to a complete halt'
wait = '"wait" # remain stationary at the stop line'
line_up_left = '"line_up_left" # enter the left-turn lane'
u_turn = '"u_turn" # perform a 180-degree turn'
cross = '"cross" # proceed straight through intersection'
left_turn = '"left_turn" # turn left at intersection'
right_turn = '"right_turn" # turn right at intersection'
back_up = '"reverse" # drive backwards'


basic_actions = [
    '"idle" # keep driving on the current lane',
    '"accelerate" # keep driving on the current lane and increase speed',
    '"decelerate" # keep driving on the current lane but slow down',
    '"stop" # decelerate to a complete halt / remain stationary',
    '"reverse" # drive backwards',
]


intersection_action_mapping_bendplatz = {
    "top": {
        15: [line_up_left] + basic_actions,
        18: [left_turn] + basic_actions,
        17: [cross, right_turn] + basic_actions,
        5: [left_turn] + basic_actions,
        1: [right_turn] + basic_actions,
        9: [cross] + basic_actions,
        21: basic_actions,
        25: basic_actions,
    },
    "right": {
        4: [left_turn] + basic_actions,
        0: [right_turn] + basic_actions,
        8: [cross] + basic_actions,
        14: basic_actions,
        22: basic_actions,
    },
    "bottom": {
        24: [line_up_left] + basic_actions,
        12: [left_turn] + basic_actions,
        13: [cross, right_turn] + basic_actions,
        7: [left_turn] + basic_actions,
        3: [right_turn] + basic_actions,
        11: [cross] + basic_actions,
        19: basic_actions,
        16: basic_actions,
    },
    "left": {
        6: [left_turn] + basic_actions,
        2: [right_turn] + basic_actions,
        10: [cross] + basic_actions,
        20: basic_actions,
        23: basic_actions,
    },
}


intersection_layout_bendplatz = {
    "top": {
        "routes": {
            GoalManeuver.LeftTurn: [15, 18, 5, 20],
            GoalManeuver.RightTurn: [15, 17, 1, 22],
            GoalManeuver.Straight: [15, 17, 9, 21, 25],
        },
        "oncoming": "bottom",
        "left": "right",
        "right": "left",
        "lane_ids": {15, 17, 18, 1, 5, 9, 21, 25},
    },
    "right": {
        "routes": {
            GoalManeuver.LeftTurn: [14, 4, 21, 25],
            GoalManeuver.RightTurn: [14, 0, 19, 16],
            GoalManeuver.Straight: [14, 8, 22],
        },
        "oncoming": "left",
        "left": "bottom",
        "right": "top",
        "lane_ids": {14, 0, 4, 8, 22},
    },
    "bottom": {
        "routes": {
            GoalManeuver.LeftTurn: [24, 12, 7, 22],
            GoalManeuver.RightTurn: [24, 13, 3, 20],
            GoalManeuver.Straight: [24, 13, 11, 19, 16],
        },
        "oncoming": "top",
        "left": "left",
        "right": "right",
        "lane_ids": {24, 12, 13, 3, 7, 11, 19, 16},
    },
    "left": {
        "routes": {
            GoalManeuver.LeftTurn: [23, 6, 19, 16],
            GoalManeuver.RightTurn: [23, 2, 21, 25],
            GoalManeuver.Straight: [23, 10, 20],
        },
        "oncoming": "right",
        "left": "top",
        "right": "bottom",
        "lane_ids": {23, 2, 6, 10, 20},
    },
}


def create_schema(actions: list[str], title: str, reasoning=True) -> dict:
    schema = {
        "additionalProperties": False,
        "properties": {
            "action_ranking": {
                "items": {"enum": actions, "type": "string"},
                "title": "Action Ranking",
                "description": "Rank all available actions from best to worst",
                "type": "array",
            }
        },
        "required": ["action_ranking"],
        "title": title,
        "type": "object",
    }
    if reasoning:
        schema["$defs"] = {
            "Reasoning": {
                "additionalProperties": False,
                "properties": {
                    "observations": {
                        "items": {"type": "string"},
                        "title": "Observations",
                        "type": "array",
                    },
                    "decision": {"title": "Decision", "type": "string"},
                },
                "required": ["observations", "decision"],
                "title": "Reasoning",
                "type": "object",
            }
        }
        schema["properties"] = {
            "reasoning": {"$ref": "#/$defs/Reasoning"},
            "action_ranking": {
                "items": {"enum": actions, "type": "string"},
                "title": "Action Ranking",
                "description": "Rank all available actions from best to worst",
                "type": "array",
            },
        }
        schema["required"] = ["reasoning", "action_ranking"]
    return schema


def road_name(lane_id: int) -> str:
    if lane_id in intersection_layout_bendplatz["top"]["lane_ids"]:
        return "top"
    elif lane_id in intersection_layout_bendplatz["right"]["lane_ids"]:
        return "right"
    elif lane_id in intersection_layout_bendplatz["bottom"]["lane_ids"]:
        return "bottom"
    elif lane_id in intersection_layout_bendplatz["left"]["lane_ids"]:
        return "left"
    else:
        raise ValueError(f"Unexpected lane_id {lane_id}")


def remaining_lanelet_ids(lane_id: int, maneuver: GoalManeuver):
    road = road_name(lane_id)
    route: list[int] = intersection_layout_bendplatz[road]["routes"][maneuver]
    current_idx = route.index(lane_id) + 1
    return route[current_idx:]


def has_right_of_way(lane_id: int) -> bool:
    return road_name(lane_id) == "top" or road_name(lane_id) == "bottom"


def in_intersection(lane_id: int) -> bool:
    return lane_id in range(12)


def before_intersection(lane_id: int, route: list[int]) -> bool:
    # Check if id is in the store at all
    if lane_id not in route:
        raise ValueError(f"Unexpected lane_id {lane_id}")

    # Find indices where range(12) numbers start and end
    range12_start = None

    for i in range(len(route)):
        if range12_start is None and route[i] in range(12):
            range12_start = i
    id_position = route.index(lane_id)
    return id_position < range12_start


def after_intersection(lane_id: int, route: list[int]) -> bool:
    return not in_intersection(lane_id) and not before_intersection(lane_id, route)


def get_current_maneuver(lane_id, routes: dict[GoalManeuver, list[int]]) -> str:
    in_left = lane_id in routes[GoalManeuver.LeftTurn]
    in_right = lane_id in routes[GoalManeuver.RightTurn]
    in_straight = lane_id in routes[GoalManeuver.Straight]

    if in_left and not in_right and not in_straight:
        if in_intersection(lane_id):
            return "is turning left"
        else:
            return "is lining up to turn left"
    elif in_right and not in_left and not in_straight:
        return "is turning right"
    else:
        if in_intersection(lane_id):
            return "is crossing the intersection"
        elif before_intersection(lane_id, routes[GoalManeuver.Straight]):
            return "is approaching the intersection"
        else:
            return "has already passed the intersection"


def get_ego_perspective_location(
    vehicle_maneuver: str, vehicle_road_name: str, ego_road_dict: dict
) -> str:
    if "turning" in vehicle_maneuver or "crossing" in vehicle_maneuver:
        if vehicle_road_name == ego_road_dict["oncoming"]:
            return "is currently inside the intersection and entered the intersection from the oncoming lane"
        elif vehicle_road_name == ego_road_dict["left"]:
            return "is currently inside the intersection and entered the intersection from the left incoming lane"
        elif vehicle_road_name == ego_road_dict["right"]:
            return "is currently inside the intersection and entered the intersection from the right incoming lane"
        else:
            return "is currently inside the intersection and entered the intersection from your lane"
    if "has already passed the intersection" in vehicle_maneuver:
        if vehicle_road_name == ego_road_dict["oncoming"]:
            return "is currently on the opposing lane"
        elif vehicle_road_name == ego_road_dict["left"]:
            return "is currently on the right outgoing lane"
        elif vehicle_road_name == ego_road_dict["right"]:
            return "is currently on the left outgoing lane"
        else:
            return "is currently on the straight outgoing lane"
    else:
        if vehicle_road_name == ego_road_dict["oncoming"]:
            return "is currently on the oncoming lane"
        elif vehicle_road_name == ego_road_dict["left"]:
            return "is currently on the left incoming lane"
        elif vehicle_road_name == ego_road_dict["right"]:
            return "is currently on the right incoming lane"
        else:
            return "is currently on the same lane as you"


def get_ego_maneuver(scenario: Scenario, planning_problem: PlanningProblem) -> str:
    intersection_layout = intersection_layout_bendplatz
    ego_lane_id = scenario.lanelet_network.find_most_likely_lanelet_by_state(
        [planning_problem.initial_state]
    )[0]
    ego_road_name = road_name(ego_lane_id)
    ego_road_dict = intersection_layout[ego_road_name]
    return get_current_maneuver(ego_lane_id, ego_road_dict["routes"])


def get_ego_target_id(
    ego_lane_id, ego_road_dict: dict, ego_maneuver: GoalManeuver
) -> int:
    route = ego_road_dict["routes"][ego_maneuver]
    start = False
    intersection = False
    target_id = None
    for lane_id in route:
        if ego_lane_id == lane_id:
            assert not start
            start = True
        if in_intersection(lane_id):
            intersection = True
        else:
            if intersection:
                assert not target_id
                target_id = lane_id
    assert target_id is not None
    return target_id


def find_lane_id(state, scenario) -> int:
    try:
        return scenario.ego_lane_network.find_most_likely_lanelet_by_state([state])[0]
    except IndexError:
        return -1


def clean_action_strings(action_list):
    """
    Removes quotes and comments from action strings.

    Args:
        action_list: List of strings containing quoted actions with comments

    Returns:
        List of cleaned action strings
    """
    cleaned_actions = []

    for action in action_list:
        # Extract content between quotes
        if '"' in action:
            # Find the content between the first and second quote
            start_quote = action.find('"')
            end_quote = action.find('"', start_quote + 1)

            if end_quote != -1:
                # Extract just the action name
                cleaned_action = action[start_quote + 1 : end_quote]
                cleaned_actions.append(cleaned_action)
        else:
            # If no quotes found, just add the action as is
            cleaned_actions.append(action)

    return cleaned_actions


def crossing_prompts(
    scenario: Scenario, planning_problem: PlanningProblem, goal_maneuver: GoalManeuver
) -> tuple[str, list[str], dict]:
    intersection_layout = intersection_layout_bendplatz
    intersection_action_mapping = intersection_action_mapping_bendplatz
    ego_velocity = planning_problem.initial_state.velocity
    ego_acceleration = planning_problem.initial_state.acceleration
    ego_lane_id = find_lane_id(planning_problem.initial_state, scenario)
    ego_road_name = road_name(ego_lane_id)
    ego_road_dict = intersection_layout[ego_road_name]
    # Catch the case where the vehicle already left the intersection and only one maneuver makes sense
    if ego_lane_id not in ego_road_dict["routes"][goal_maneuver]:
        goal_maneuver = GoalManeuver.Straight
    ego_route = ego_road_dict["routes"][goal_maneuver]
    ego_orientation = np.array(
        [
            np.cos(planning_problem.initial_state.orientation),
            np.sin(planning_problem.initial_state.orientation),
        ]
    )
    ego_maneuver = get_ego_maneuver(scenario, planning_problem)

    if in_intersection(ego_lane_id):
        location = "are currently inside"
    elif before_intersection(ego_lane_id, ego_route):
        location = "are currently approaching"
    else:
        location = "already passed"

    if has_right_of_way(ego_lane_id):
        right_of_way = "has"
    else:
        right_of_way = "does not have"

    if goal_maneuver == GoalManeuver.RightTurn:
        goal = "make a right turn"
    elif goal_maneuver == GoalManeuver.LeftTurn:
        goal = "make a left turn"
    else:
        goal = "cross the intersection"

    def describe_location(
        vehicle_name: str, vehicle_position: np.ndarray, lane_id: int
    ) -> str:
        def distance_descr() -> str:
            # vehicle_pos = curvilinear_cosy.convert_to_curvilinear_coords(vehicle_position[0], vehicle_position[1])
            # vehicle_dir = vehicle_pos - ego_position
            vehicle_dir = vehicle_position - planning_problem.initial_state.position
            dist = np.linalg.norm(vehicle_dir)
            vehicle_proj = np.dot(ego_orientation, vehicle_dir / dist)
            angle = np.arccos(vehicle_proj)
            cross_product = np.cross(ego_orientation, vehicle_dir / dist)
            if cross_product < 0:
                angle = -angle

            threshold = np.pi / 8

            # Front: -π/8 to π/8
            if -threshold <= angle <= threshold:
                return f"in front of you and {np.linalg.norm(dist):.1f} meters away"
            # Front-Left (grey zone): π/8 to π/4
            elif threshold < angle <= np.pi / 4:
                return f"front-left of you and {np.linalg.norm(dist):.1f} meters away"
            # Left: π/4 to 3π/4
            elif np.pi / 4 < angle <= 3 * np.pi / 4:
                return f"left of you and {np.linalg.norm(dist):.1f} meters away"
            # Back-Left (grey zone): 3π/4 to 7π/8
            elif 3 * np.pi / 4 < angle <= 7 * np.pi / 8:
                return f"back-left of you and {np.linalg.norm(dist):.1f} meters away"
            # Front-Right (grey zone): -π/8 to -π/4
            elif -threshold > angle >= -np.pi / 4:
                return f"front-right of you and {np.linalg.norm(dist):.1f} meters away"
            # Right: -π/4 to -3π/4
            elif -np.pi / 4 > angle >= -3 * np.pi / 4:
                return f"right of you and {np.linalg.norm(dist):.1f} meters away"
            # Back-Right (grey zone): -3π/4 to -7π/8
            elif -3 * np.pi / 4 > angle >= -7 * np.pi / 8:
                return f"back-right of you and {np.linalg.norm(dist):.1f} meters away"
            # Behind: 7π/8 to π or -7π/8 to -π
            else:
                return f"behind you and {np.linalg.norm(dist):.1f} meters away"

        vehicle_road_name = road_name(lane_id)
        vehicle_route_dict = intersection_layout[vehicle_road_name]["routes"]
        vehicle_maneuver = get_current_maneuver(lane_id, vehicle_route_dict)
        vehicle_description = f"{vehicle_name} {vehicle_maneuver}."
        vehicle_description += f" It {get_ego_perspective_location(vehicle_maneuver, vehicle_road_name, ego_road_dict)}."

        vehicle_description += f" It is {distance_descr()}."
        return vehicle_description

    def velocity_descr(v: float, to_km=True) -> str:
        if to_km:
            v *= 3.6
            return f"{v:.1f} km/h"
        return f"{v:.1f} m/s"

    def acceleration_descr(a: float, to_km=False) -> str:
        if to_km:
            a *= 12960
            return f"{a:.1f} km/h²"
        return f"{a:.1f} m/s²"

    ego_description = f"You {location} an intersection of two roads. The road you are driving on {right_of_way} the right of way. "
    ego_description += f"Your vehicle {ego_maneuver}. "
    if ego_maneuver != "is lining up to turn left":
        ego_description += f" Your goal is to {goal}. "
    ego_description += f" Your speed is {velocity_descr(ego_velocity)} and your acceleration is {acceleration_descr(ego_acceleration)}.\n"

    descr = f"{ego_description}Now, here is an overview of the current traffic:\n"
    # describe other vehicles' locations
    for vehicle in scenario.dynamic_obstacles:
        if vehicle.obstacle_type in [
            ObstacleType.CAR,
            ObstacleType.BUS,
            ObstacleType.BICYCLE,
        ]:
            try:
                vehicle_velocity = vehicle.initial_state.velocity
                vehicle_acceleration = vehicle.initial_state.acceleration
                vehicle_lane_id = find_lane_id(vehicle.initial_state, scenario)
                if vehicle_lane_id < 0:
                    continue

                descr += describe_location(
                    f"{vehicle.obstacle_type.name} {vehicle.obstacle_id}",
                    vehicle.initial_state.position,
                    vehicle_lane_id,
                )
                descr += f" Its speed is {velocity_descr(vehicle_velocity)} and its acceleration is {acceleration_descr(vehicle_acceleration)}.\n"
            except IndexError:
                print(f"WARNING: Skipped {vehicle.obstacle_id}")
        elif vehicle.obstacle_type == ObstacleType.PEDESTRIAN:
            print(f"WARNING: Skipped {vehicle.obstacle_id}")
        else:
            raise ValueError(f"Unexpected obstacle type: {vehicle.obstacle_type}")

    # figure out which actions you can take
    actions = intersection_action_mapping[ego_road_name][ego_lane_id]
    clean_actions = clean_action_strings(actions)
    return descr, actions, create_schema(clean_actions, "Response")


def highway_user_prompt(
    scenario: Scenario, planning_problem: PlanningProblem, ego_vehicle_id: int
) -> tuple[str, list[str], dict]:
    initial_lanelet = find_lane_id(planning_problem.initial_state, scenario)
    # validate lane layout
    lanelet_amount = len(scenario.lanelet_network.lanelets)
    assert (
        lanelet_amount == 3
    ), f"There should be exactly 3 lanes on the highway, instead there are {lanelet_amount}!"
    for lanelet in scenario.lanelet_network.lanelets:
        assert lanelet.lanelet_id in [
            1,
            2,
            3,
        ], f"Unknown lanelet id: {lanelet.lanelet_id}"

    # describe lane layout
    descr = """Environment description:
You are driving on a highway with 3 same-direction lanes."""

    def lane_descr(lane_id: int) -> str:
        if lane_id == 1:
            return "left-most"
        elif lane_id == 2:
            return "middle"
        elif lane_id == 3:
            return "right-most"
        else:
            raise ValueError(f"Unexpected laneID: {lane_id}")

    def velocity_descr(v: float, to_km=True) -> str:
        if to_km:
            v *= 3.6
            return f"{v:.1f} km/h"
        return f"{v:.1f} m/s"

    def acceleration_descr(a: float, to_km=False) -> str:
        if to_km:
            a *= 12960
            return f"{a:.1f} km/h²"
        return f"{a:.1f} m/s²"

    action_enums = [
        "idle",
        "accelerate",
        "decelerate",
        "stop",
    ]
    ego_actions = [
        '"idle" # keep on the current lane',
        '"accelerate" # keep on the current lane and increase speed',
        '"decelerate" # keep on the current lane and slow down',
        '"stop" # emergency break',
    ]

    ego_lane_descr = lane_descr(initial_lanelet)
    descr += f" You are currently in the {ego_lane_descr} lane."
    if "middle" in ego_lane_descr:
        ego_actions.insert(3, '"right_lane_change" # change to the lane right of you')
        ego_actions.insert(3, '"left_lane_change" # change to the lane left of you')
        action_enums += ["left_lane_change", "right_lane_change"]
    elif "left" in ego_lane_descr:
        ego_actions.insert(3, '"right_lane_change" # change to the lane right of you')
        action_enums.append("right_lane_change")
    else:
        ego_actions.insert(3, '"left_lane_change" # change to the lane left of you')
        action_enums.append("left_lane_change")

    # describe initial state of ego vehicle
    ego_position: np.ndarray = planning_problem.initial_state.position
    ego_velocity = planning_problem.initial_state.velocity
    ego_acceleration = planning_problem.initial_state.acceleration
    descr += f"\nYour speed is {velocity_descr(ego_velocity)} and your acceleration is {acceleration_descr(ego_acceleration)}."

    def distance_descr(v_pos: np.ndarray) -> str:
        dist = v_pos[0] - ego_position[0]
        if dist > 0:
            return f"{dist:.1f} meters in front of"
        else:
            return f"{dist:.1f} meters behind"

    # ==== build configuration
    config = CriMeConfiguration()
    config.update(ego_id=ego_vehicle_id, sce=scenario)
    ttc_evaluator = TTC(config)

    def ttc_descr(other_id: int):
        return f"{ttc_evaluator.compute(other_id, 0)} sec"

    # describe other vehicles
    descr += "\nThese are all other vehicles on the highway:"
    descr += "  (TTC stands for 'time to collision')"
    following_possible = False
    for vehicle in scenario.dynamic_obstacles:
        if ego_vehicle_id == vehicle.obstacle_id:
            continue
        v_lane_id = find_lane_id(vehicle.initial_state, scenario)
        v_lane_descr = lane_descr(v_lane_id)
        obstacle = scenario.obstacle_by_id(vehicle.obstacle_id)
        v_position: np.ndarray = obstacle.initial_state.position
        v_velocity = obstacle.initial_state.velocity
        v_acceleration = obstacle.initial_state.acceleration
        v_dist_descr = distance_descr(v_position)
        if "front" in v_dist_descr and v_lane_descr == ego_lane_descr:
            following_possible = True
        descr += f"\n  - Vehicle with ID {vehicle.obstacle_id} is driving on the {v_lane_descr} lane."
        descr += f" It is {v_dist_descr} you and its speed is {velocity_descr(v_velocity)} and its acceleration is {acceleration_descr(v_acceleration)}."
        descr += f" The TTC is {ttc_descr(vehicle.obstacle_id)}."

    if following_possible:
        ego_actions.insert(
            -1, '"follow_preceding_vehicle" # follow the vehicle in front of you'
        )
        action_enums.append("follow_preceding_vehicle")
    # describe goal state
    # goal_states = self.planning_problem.goal.state_list
    # assert len(goal_states) == 1, f"There should only be one goal state, instead there are: {len(goal_states)}!"
    # goal_state = goal_states[0]
    # assert isinstance(goal_state.position,
    #                   Polygon), f"Goal state is not a Polygon, but of type {type(goal_state.position)}"
    # goal_position: Polygon = goal_state.position
    # goal_dist_descr = distance_descr(goal_position.center)
    # goal_lanes = self.scenario.lanelet_network.find_lanelet_by_position([goal_position.center])[0]
    # assert len(goal_lanes) == 1, f" Expected exactly one goal lane, instead got: {goal_lanes}"
    # descr += f"\nYour goal is to get to the goal area which is located in the {lane_descr(goal_lanes[0])} lane and {goal_dist_descr} your current position."
    return descr, ego_actions, create_schema(action_enums, "Response")
