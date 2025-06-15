import math
import random
from typing import cast, Optional
import gymnasium
import numpy as np
from commonroad.common.common_lanelet import LineMarking
from commonroad.common.util import AngleInterval, Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario, ScenarioID
from commonroad.scenario.state import InitialState, ExtendedPMState, CustomState
from commonroad.scenario.trajectory import Trajectory
from crpred.basic_models.constant_velocity_predictor import ConstantVelocityCurvilinearPredictor
from crpred.utility.config import PredictorParams
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from highway_env.envs import AbstractEnv
from highway_env.envs.common.observation import TimeToCollisionObservation
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import LaneIndex
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from matplotlib import pyplot as plt

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.scenario.scenario import Tag

from sandra.actions import LongitudinalAction
from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.commonroad.reach import ReachVerifier
from sandra.utility.visualization import plot_scenario, plot_predicted_trajectory


class HighwayEnvScenario:
    def __init__(self, config: dict, seed: int = 4213, dt: float = 0.2, start_time: int = 0):
        env = gymnasium.make('highway-v0', render_mode='rgb_array', config=config['highway-v0'])
        self.seed = seed
        self._env = RecordVideo(env, video_folder="run", episode_trigger=lambda e: True)
        # self._env.unwrapped.set_record_video_wrapper(env)
        self.observation, _ = self._env.reset(seed=seed)
        self.done = self.truncated = False
        self.scenario: AbstractEnv = cast(AbstractEnv, self._env.unwrapped)
        self.dt = dt
        self.time_step = start_time
        self.prediction_length = 20
        self.minimum_interval = 1.
        self._ids: set[int] = {0}
        self._lanelet_mapping: dict[LaneIndex, int] = {}

    def get_lane_id(self, lane_id: LaneIndex | int) -> Optional[LaneIndex | int]:
        """
        Convert highway-env lane index to commonroad lanelet_id and vice-versa.
        """
        if not isinstance(lane_id, int) and lane_id in self._lanelet_mapping.keys():
            return self._lanelet_mapping[lane_id]
        else:
            for key, value in self._lanelet_mapping.items():
                if value == lane_id:
                    return key
        return None

    @staticmethod
    def _highenv_coordinate_to_commonroad(coordinates: np.ndarray) -> np.ndarray:
        """
        Flip y-coordinates to match commonroad coordinate system.
        """
        result = coordinates.copy()
        if coordinates.ndim == 1 and coordinates.shape[0] == 2:
            result[1] = -result[1]
        elif coordinates.ndim == 2 and coordinates.shape[1] == 2:
            result[:, 1] = -result[:, 1]
        else:
            raise ValueError(f"Invalid input shape: {coordinates.shape}. ")
        return result

    def _create_vertices_along_line(self, start: np.ndarray, end: np.ndarray, direction: np.ndarray,
                                    interval: float = 1000.0) -> np.ndarray:
        """
        Creates vertices along a straight line at regular intervals.
        """
        # Calculate number of intervals (including start point)
        line_vector = end - start
        total_distance = float(np.linalg.norm(line_vector))
        num_intervals = int(np.ceil(total_distance / interval))
        vertices = [start.copy()]
        # Add intermediate vertices at regular intervals
        for i in range(1, num_intervals):
            distance = i * interval
            if total_distance - distance > self.minimum_interval:
                vertex = start + direction * distance
                vertices.append(vertex)
        vertices.append(end.copy())
        return np.array(vertices)

    @staticmethod
    def _line_type_to_line_marking(line_type: LineType) -> LineMarking:
        """
        Convert highway-env line type to commonroad line marking.
        """
        if line_type == LineType.STRIPED:
            return LineMarking.DASHED
        elif line_type == LineType.CONTINUOUS_LINE or line_type == LineType.CONTINUOUS:
            return LineMarking.SOLID
        elif line_type == LineType.NONE:
            return LineMarking.DASHED  # todo: I think there is no other work-around
        else:
            raise ValueError(f"Invalid line type: {line_type}")

    def _make_commonroad_lanelet(self, lane: StraightLane) -> Lanelet:
        road_network = self.scenario.vehicle.road.network

        # Generate lanelet vertices
        left_vertices = self._highenv_coordinate_to_commonroad(
            self._create_vertices_along_line(lane.start, lane.end, lane.direction))
        center_offset = lane.direction_lateral * lane.width / 2
        center_vertices = self._highenv_coordinate_to_commonroad(
            self._create_vertices_along_line(lane.start + center_offset, lane.end + center_offset, lane.direction))
        right_offset = lane.direction_lateral * lane.width
        right_vertices = self._highenv_coordinate_to_commonroad(
            self._create_vertices_along_line(lane.start + right_offset, lane.end + right_offset, lane.direction))

        # Generate lanelet id
        lane_index = road_network.get_closest_lane_index(lane.start + center_offset, lane.heading)
        lanelet_id = self._next_id()
        self._lanelet_mapping[lane_index] = lanelet_id

        # Add adjacent lanelet ids
        neighbors = road_network.side_lanes(lane_index)
        next_center_offset = lane.direction_lateral * (lane.width / 2) * 3
        adjacent_right = None
        adjacent_right_same_direction = None
        if (
                adj_right := road_network.get_closest_lane_index(lane.start + next_center_offset,
                                                                 lane.heading)) in neighbors:
            adjacent_right = adj_right[2]
            adjacent_right_lane: StraightLane = cast(StraightLane, road_network.get_lane(adj_right))
            adjacent_right_same_direction = True if adjacent_right_lane.heading == lane.heading else False
        adjacent_left = None
        adjacent_left_same_direction = None
        if (
                adj_left := road_network.get_closest_lane_index(lane.start - next_center_offset,
                                                                lane.heading)) in neighbors:
            adjacent_left = adj_left[2]
            adjacent_left_lane: StraightLane = cast(StraightLane, road_network.get_lane(adj_left))
            adjacent_left_same_direction = True if adjacent_left_lane.heading == lane.heading else False

        # Add line-markings
        line_marking_left_vertices, line_marking_right_vertices = lane.line_types

        return Lanelet(
            left_vertices,
            center_vertices,
            right_vertices,
            lanelet_id,
            adjacent_left=adjacent_left,
            adjacent_left_same_direction=adjacent_left_same_direction,
            adjacent_right=adjacent_right,
            adjacent_right_same_direction=adjacent_right_same_direction,
            line_marking_left_vertices=self._line_type_to_line_marking(line_marking_left_vertices),
            line_marking_right_vertices=self._line_type_to_line_marking(line_marking_right_vertices),
        )

    def _make_commonroad_obstacle(self, vehicle: MDPVehicle | IDMVehicle, obstacle_id: int) -> DynamicObstacle:
        def make_extended_pm_state(position: np.ndarray, orientation: float, time_step: int) -> ExtendedPMState:
            return ExtendedPMState(
                position=position,
                velocity=vehicle.speed,
                orientation=-orientation,
                acceleration=0.0,
                time_step=time_step,
            )

        obstacle_type = ObstacleType.CAR
        top_left_corner = vehicle.position
        # First, transform top-left into center coordinates, then
        center = top_left_corner + np.array([vehicle.LENGTH, vehicle.WIDTH])
        center = self._highenv_coordinate_to_commonroad(center)
        obstacle_shape = Rectangle(vehicle.LENGTH, vehicle.WIDTH, center=np.array([0.0, 0.0]), orientation=vehicle.heading)
        obstacle_state = InitialState(
            position=center,
            orientation=(-vehicle.heading),
            velocity=vehicle.speed,
            acceleration=vehicle.action["acceleration"],
            time_step=self.time_step,
            # todo: check if this is correct
            slip_angle=math.atan2(vehicle.velocity[1], vehicle.velocity[0]),
            yaw_rate=0.0,
        )
        predicted_positions, predicted_orientations = vehicle.predict_trajectory_constant_speed(
            np.full(self.prediction_length, self.dt, dtype=float))
        predicted_positions = self._highenv_coordinate_to_commonroad(np.array(predicted_positions))
        predicted_states = [make_extended_pm_state(a, b, self.time_step + i + 1) for i, (a, b) in
                            enumerate(zip(predicted_positions, predicted_orientations))]
        predicted_trajectory = Trajectory(
            self.time_step + 1,
            predicted_states,
        )
        prediction = TrajectoryPrediction(
            predicted_trajectory,
            obstacle_shape,
        )

        lane_index = vehicle.lane_index
        lanelet_id = self.get_lane_id(lane_index)
        return DynamicObstacle(
            obstacle_id,
            obstacle_type,
            obstacle_shape,
            obstacle_state,
            # prediction=prediction,
            initial_center_lanelet_ids={lanelet_id},
            initial_shape_lanelet_ids={lanelet_id},
            history=[],
        )

    def _make_commonroad_planning_problem(self, ego_vehicle: MDPVehicle, initial_state: InitialState) -> PlanningProblem:
        road = ego_vehicle.road
        goal_lane: StraightLane = cast(StraightLane, road.network.get_lane(ego_vehicle.target_lane_index))
        goal_x: float = ego_vehicle.position[0] + ego_vehicle.speed * self.dt * self.prediction_length
        goal_y: float = goal_lane.start[1] + goal_lane.width / 2
        goal_center = self._highenv_coordinate_to_commonroad(np.array([goal_x, goal_y]))
        goal_state = CustomState(
            position=Rectangle(
                ego_vehicle.LENGTH,
                ego_vehicle.WIDTH,
                center=goal_center,
            ),
            orientation=AngleInterval(
                -math.pi / 2,
                math.pi / 2,
            ),
            time_step=Interval(
                self.time_step,
                self.time_step + self.prediction_length,
            ),
            velocity=Interval(
                ego_vehicle.MIN_SPEED,
                ego_vehicle.MAX_SPEED,
            )
        )
        goal_region = GoalRegion(
            [goal_state]
        )
        return PlanningProblem(
            self._next_id(),
            initial_state,
            goal_region
        )

    def _next_id(self):
        """
        Generate unique ids
        """
        last_id = max(self._ids)
        next_id = last_id + 1
        self._ids.add(next_id)
        return next_id

    @property
    def commonroad_representation(self) -> tuple[Scenario, DynamicObstacle, PlanningProblem]:
        """
        Get the commonroad representation of the scenario
        """
        ego_vehicle: MDPVehicle = cast(MDPVehicle, self.scenario.vehicle)
        road = ego_vehicle.road
        scenario = Scenario(
            self.dt,
            scenario_id=ScenarioID(map_name="Sandra", map_id=self.seed),
        )

        # Add all lanelets
        lanelets: list[Lanelet] = []
        for lane in road.network.lanes_list():
            lanelet = self._make_commonroad_lanelet(cast(StraightLane, lane))
            lanelets.append(lanelet)
        lanelet_network = LaneletNetwork.create_from_lanelet_list(lanelets)
        scenario.add_objects(lanelet_network)

        # Add all obstacles
        ego_vehicle_commonroad = self._make_commonroad_obstacle(ego_vehicle, self._next_id())
        scenario.add_objects(ego_vehicle_commonroad)
        ego_obstacle_id = ego_vehicle_commonroad.obstacle_id
        for vehicle in road.vehicles[1:]:
            scenario.add_objects(self._make_commonroad_obstacle(vehicle, self._next_id()))

        # Add all obstacle predictions
        config = PredictorParams(num_steps_prediction=self.prediction_length, dt=self.dt)
        predictor = ConstantVelocityCurvilinearPredictor(config)
        scenario = predictor.predict(scenario, initial_time_step=self.time_step)

        # Create planning problem
        planning_problem = self._make_commonroad_planning_problem(ego_vehicle, ego_vehicle_commonroad.initial_state)
        planning_problem_set = PlanningProblemSet([planning_problem])

        # write new scenario
        author = 'Sandra Müller'
        affiliation = 'Technical University of Munich, Germany'
        source = ''
        tags = {Tag.HIGHWAY}
        fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)
        path_scenario = (
                PROJECT_ROOT + "/scenarios/" + str(scenario.scenario_id) + ".xml"
        )
        fw.write_to_file(path_scenario, OverwriteExistingFile.ALWAYS)
        return scenario, scenario.obstacle_by_id(ego_obstacle_id), planning_problem

    @property
    def highway_env_representation(self) -> Env:
        return self._env

    def plot(self, plot_commonroad=False):
        plt.imshow(self._env.render())
        plt.show()
        if plot_commonroad:
            scenario, ego_vehicle, planning_problem = self.commonroad_representation
            plot_scenario(scenario, planning_problem, plot_limits=[350, 500, -30, 0])
            plot_predicted_trajectory(scenario, ego_vehicle)

    def step(self, action_id: int) -> bool:
        self.observation, reward, self.done, self.truncated, info = self._env.step(action_id)
        self.time_step += 0
        self._env.render()
        self._ids = {0}
        if self.done or self.truncated:
            self._env.close()
            return False
        return True

    @staticmethod
    def make() -> "HighwayEnvScenario":
        env_config = {
            'highway-v0':
                {
                    "observation": {
                        "type": "TimeToCollision",
                        "horizon": 10
                    },
                    "action": {
                        "type": "DiscreteMetaAction",
                        "target_speeds": np.linspace(5, 32, 9),
                    },
                    "lanes_count": 4,
                    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                    "duration": 30,
                    "vehicles_density": 2.0,
                    "show_trajectories": True,
                    "render_agent": True,
                    "scaling": 5,
                    'initial_lane_id': None,
                    "ego_spacing": 4,
                }
        }
        return HighwayEnvScenario(env_config)


if __name__ == "__main__":
    scenario_ = HighwayEnvScenario.make()
    commonrad_scenario, _ ,_  = scenario_.commonroad_representation
    verifier = ReachVerifier(commonrad_scenario, SanDRAConfiguration())
    verifier.verify([LongitudinalAction.KEEP])
    # scenario_.plot(plot_commonroad=True)
    # while scenario_.step(1):
    #     scenario_.plot(plot_commonroad=True)
    # scenario_.highway_env_representation.close()
