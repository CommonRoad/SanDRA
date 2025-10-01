import math
from typing import cast
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
from commonroad.scenario.state import InitialState, CustomState
from commonroad.scenario.trajectory import Trajectory

from crpred.basic_models.constant_velocity_predictor import (
    ConstantVelocityCurvilinearPredictor,
)
from crpred.utility.config import PredictorParams
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from highway_env.envs import AbstractEnv
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import LaneIndex
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from matplotlib import pyplot as plt

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.scenario.scenario import Tag

from sandra.config import PROJECT_ROOT
from sandra.utility.road_network import RoadNetwork, EgoLaneNetwork
from sandra.utility.visualization import plot_scenario


class HighwayEnvScenario:
    def __init__(
        self,
        config: dict | RecordVideo,
        seed: int = 4213,
        dt: float = 0.2,
        start_time: int = 0,
        horizon: int = 30,
        maximum_lanelet_length: float = 1000,
        use_sonia: bool = False,
        video_folder: str = None,
    ):
        self.seed = seed
        if isinstance(config, dict):
            env = gymnasium.make(
                "highway-v0", render_mode="rgb_array", config=config["highway-v0"]
            )
            name_prefix = f"highway_{seed}"
            if use_sonia:
                name_prefix += "_spot"
            if not video_folder:
                video_folder = "run"
            self._env = RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda e: True,
                name_prefix=name_prefix,
            )
            # self._env.unwrapped.set_record_video_wrapper(env)
            self.observation, _ = self._env.reset(seed=seed)
        else:
            self._env = config
            self.observation = None

        self.done = self.truncated = False
        self.use_sonia = use_sonia  # whether set-based prediction
        self.scenario: AbstractEnv = cast(AbstractEnv, self._env.unwrapped)
        self.dt = dt
        self.time_step = start_time

        self.prediction_length = horizon + 1
        self.minimum_interval = 1.0
        self._commonroad_ids: set[int] = {0}
        self._lanelet_ids: dict[LaneIndex, dict[float, int]] = {}
        self.maximum_lanelet_length = maximum_lanelet_length

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

    @staticmethod
    def _convert_lane_id(lane_id: LaneIndex) -> int:
        return lane_id[2] + 1

    def _create_vertices_along_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        direction: np.ndarray,
        interval: float = 10.0,
    ) -> np.ndarray:
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
        end = lane.start + lane.direction * self.maximum_lanelet_length
        # Generate lanelet vertices
        center_vertices = self._highenv_coordinate_to_commonroad(
            self._create_vertices_along_line(lane.start, end, lane.direction)
        )
        center_offset = lane.direction_lateral * (lane.width / 2)
        left_vertices = self._highenv_coordinate_to_commonroad(
            self._create_vertices_along_line(
                lane.start - center_offset, end - center_offset, lane.direction
            )
        )
        right_vertices = self._highenv_coordinate_to_commonroad(
            self._create_vertices_along_line(
                lane.start + center_offset, end + center_offset, lane.direction
            )
        )

        # Generate lanelet id
        lane_index = road_network.get_closest_lane_index(lane.start, lane.heading)
        lanelet_id = self._next_id()
        assert lanelet_id == self._convert_lane_id(
            lane_index
        ), "Commonroad LaneletID should match its HighEnv counterpart."

        # Add adjacent lanelet ids
        neighbors = road_network.side_lanes(lane_index)
        adjacent_right = None
        adjacent_right_same_direction = None
        if (
            adj_right := road_network.get_closest_lane_index(
                lane.start + 2 * center_offset, lane.heading
            )
        ) in neighbors:
            adjacent_right = self._convert_lane_id(adj_right)
            adjacent_right_lane: StraightLane = cast(
                StraightLane, road_network.get_lane(adj_right)
            )
            adjacent_right_same_direction = (
                True if adjacent_right_lane.heading == lane.heading else False
            )
        adjacent_left = None
        adjacent_left_same_direction = None
        if (
            adj_left := road_network.get_closest_lane_index(
                lane.start - 2 * center_offset, lane.heading
            )
        ) in neighbors:
            adjacent_left = self._convert_lane_id(adj_left)
            adjacent_left_lane: StraightLane = cast(
                StraightLane, road_network.get_lane(adj_left)
            )
            adjacent_left_same_direction = (
                True if adjacent_left_lane.heading == lane.heading else False
            )

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
            line_marking_left_vertices=self._line_type_to_line_marking(
                line_marking_left_vertices
            ),
            line_marking_right_vertices=self._line_type_to_line_marking(
                line_marking_right_vertices
            ),
        )

    def _make_commonroad_obstacle(
        self, vehicle: MDPVehicle | IDMVehicle, obstacle_id: int
    ) -> DynamicObstacle:
        obstacle_type = ObstacleType.CAR
        center = vehicle.position
        center = self._highenv_coordinate_to_commonroad(center)
        obstacle_shape = Rectangle(
            vehicle.LENGTH, vehicle.WIDTH, center=np.array([0.0, 0.0]), orientation=0.0
        )
        obstacle_state = InitialState(
            position=center,
            orientation=(-vehicle.heading),
            velocity=vehicle.speed,
            acceleration=vehicle.action["acceleration"],
            time_step=0,
            slip_angle=math.atan2(vehicle.velocity[1], vehicle.velocity[0]),
            yaw_rate=-vehicle.action["steering"],
        )

        lanelet_id = self._convert_lane_id(vehicle.lane_index)
        return DynamicObstacle(
            obstacle_id,
            obstacle_type,
            obstacle_shape,
            obstacle_state,
            initial_center_lanelet_ids={lanelet_id},
            initial_shape_lanelet_ids={lanelet_id},
            history=[],
        )

    def _make_commonroad_planning_problem(
        self, ego_vehicle: MDPVehicle, initial_state: InitialState
    ) -> PlanningProblem:
        road = ego_vehicle.road
        if hasattr(ego_vehicle, "target_lane_index"):
            goal_lane: StraightLane = cast(
                StraightLane, road.network.get_lane(ego_vehicle.target_lane_index)
            )
        else:
            goal_lane: StraightLane = cast(
                StraightLane, road.network.get_lane(ego_vehicle.lane_index)
            )
        goal_x: float = (
            ego_vehicle.position[0]
            + ego_vehicle.speed * self.dt * self.prediction_length
        )
        goal_y: float = goal_lane.start[1] + goal_lane.width / 2 - 4.0
        goal_center = self._highenv_coordinate_to_commonroad(np.array([goal_x, goal_y]))
        goal_center[1] = initial_state.position[1]

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
                0,
                self.prediction_length,
            ),
            velocity=Interval(
                ego_vehicle.MIN_SPEED,
                ego_vehicle.MAX_SPEED,
            ),
        )
        goal_region = GoalRegion([goal_state])
        return PlanningProblem(self._next_id(), initial_state, goal_region)

    def _next_id(self):
        """
        Generate unique ids
        """
        last_id = max(self._commonroad_ids)
        next_id = last_id + 1
        self._commonroad_ids.add(next_id)
        return next_id

    def _prediction(self, scenario: Scenario, preceding_obs: DynamicObstacle) -> Scenario:
        if self.use_sonia:
            state_list = []
            for ts in range(1, self.prediction_length):
                if preceding_obs is not None and preceding_obs.prediction is not None:
                    preceding_state = preceding_obs.state_at_time(ts)
                    state_list.append(CustomState(
                        position=preceding_state.position,
                        velocity=preceding_state.velocity,
                        orientation=preceding_state.orientation,
                        time_step=ts,
                    ))
                else:
                    state_list.append(CustomState(position=np.array([0, 0]), velocity=0, time_step=ts, orientation=0.0))
            if preceding_obs is not None:
                phantom_obstacle = DynamicObstacle(
                    obstacle_id=85748,
                    obstacle_type=ObstacleType.CAR,
                    obstacle_shape=preceding_obs.obstacle_shape,
                    initial_state=preceding_obs.initial_state,
                    prediction=TrajectoryPrediction(trajectory=Trajectory(1, state_list),
                                                    shape=preceding_obs.obstacle_shape)
                )
            else:
                phantom_obstacle = DynamicObstacle(
                    obstacle_id=85748,
                    obstacle_type=ObstacleType.CAR,
                    obstacle_shape=Rectangle(5, 2),
                    initial_state=InitialState(position=np.array([0.0, 0.0]), orientation=0.0, velocity=0.0,
                                               acceleration=0.0, yaw_rate=0.0, slip_angle=0.0),
                    prediction=TrajectoryPrediction(trajectory=Trajectory(1, state_list),
                                                    shape=Rectangle(5, 2)),
                )
            scenario.add_objects(phantom_obstacle)
            return scenario
        else:
            # Add all obstacle predictions
            predict_config = PredictorParams(
                num_steps_prediction=self.prediction_length, dt=self.dt
            )
            predictor = ConstantVelocityCurvilinearPredictor(predict_config)
            # predictor = ConstantAccelerationLinearPredictor(predict_config)
            return predictor.predict(scenario, initial_time_step=1)

    @property
    def commonroad_representation(
        self, add_ego=False
    ) -> tuple[Scenario, DynamicObstacle, PlanningProblem]:
        """
        Get the commonroad representation of the scenario
        """
        ego_vehicle: MDPVehicle = cast(MDPVehicle, self.scenario.vehicle)
        road = ego_vehicle.road
        scenario = Scenario(
            self.dt,
            scenario_id=ScenarioID(
                map_name="Sandra",
                map_id=self.seed,
                obstacle_behavior="T",
                prediction_id=self.time_step,
            ),
        )

        # Add all lanelets
        lanelets: list[Lanelet] = []
        for lane in road.network.lanes_list():
            lanelet = self._make_commonroad_lanelet(cast(StraightLane, lane))
            lanelets.append(lanelet)
        lanelet_network = LaneletNetwork.create_from_lanelet_list(lanelets)
        scenario.add_objects(lanelet_network)

        # Add all obstacles
        ego_vehicle_commonroad = self._make_commonroad_obstacle(
            ego_vehicle, self._next_id()
        )
        if add_ego:
            scenario.add_objects(ego_vehicle_commonroad)

        obstacles = cast(
            list,
            road.close_vehicles_to(
                ego_vehicle,
                self.scenario.PERCEPTION_DISTANCE,
                see_behind=True,
                sort=True,
            ),
        )

        preceding_vehicle = None
        distance_min = np.inf

        initial_lanelet = scenario.lanelet_network.find_most_likely_lanelet_by_state(
            [ego_vehicle_commonroad.initial_state]
        )[0]
        for vehicle in road.vehicles:
            vehicle_lane = road.network.get_lane(vehicle.lane_index)
            s, _ = vehicle_lane.local_coordinates(vehicle.position)
            if s > self.maximum_lanelet_length or vehicle not in obstacles:
                continue
            commonroad_obs = self._make_commonroad_obstacle(vehicle, self._next_id())
            scenario.add_objects(
                commonroad_obs
            )

            # find preceding vehicle
            obstacle_lanelets = scenario.lanelet_network.find_lanelet_by_position(
                [commonroad_obs.initial_state.position]
            )[0]

            if initial_lanelet not in obstacle_lanelets:
                continue

            # Check whether obstacle is in front of ego
            if self.is_in_front(ego_vehicle_commonroad.initial_state, commonroad_obs.initial_state, threshold=0.0):
                distance = np.linalg.norm(
                    np.array(commonroad_obs.initial_state.position) - np.array(commonroad_obs.initial_state.position)
                )
                if distance < distance_min:
                    distance_min = distance
                    preceding_vehicle = commonroad_obs

        # Add all obstacle predictions
        try:
            scenario = self._prediction(scenario, preceding_vehicle)
        except ValueError as e:
            print(f"[Warning] Prediction failed due to value error: {e}")
            pass
        except Exception as e:
            print(f"[Error] Unexpected prediction failure: {e}")
            pass

        # Create planning problem
        planning_problem = self._make_commonroad_planning_problem(
            ego_vehicle, ego_vehicle_commonroad.initial_state
        )
        planning_problem_set = PlanningProblemSet([planning_problem])

        # write new scenario
        author = "Sandra Müller"
        affiliation = "Technical University of Munich, Germany"
        source = ""
        tags = {Tag.HIGHWAY}
        fw = CommonRoadFileWriter(
            scenario, planning_problem_set, author, affiliation, source, tags
        )
        path_scenario = (
            PROJECT_ROOT + "/scenarios/" + str(scenario.scenario_id) + ".xml"
        )
        fw.write_to_file(path_scenario, OverwriteExistingFile.ALWAYS)
        return scenario, ego_vehicle_commonroad, planning_problem

    def is_in_front(self, ego_state, obs_state, threshold=0.0):
        """
        Orientation check ensures that only obstacles in the driving direction of the ego
         are considered “in front, not just geometrically ahead along the lanelet.
         """
        # Ego position and heading
        ego_pos = np.array([ego_state.position[0], ego_state.position[1]])
        ego_heading = ego_state.orientation  # in radians
        ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)])

        # Obstacle position
        obs_pos = np.array([obs_state.position[0], obs_state.position[1]])

        # Vector from ego to obstacle
        v = obs_pos - ego_pos

        # Projection of v onto ego's direction
        projection = np.dot(v, ego_dir)

        return projection > threshold

    @property
    def highway_env_representation(self) -> Env:
        return self._env

    def plot(self, plot_commonroad=False):
        plt.imshow(self._env.render())
        plt.show()
        if plot_commonroad:
            scenario, ego_vehicle, planning_problem = self.commonroad_representation
            plot_scenario(scenario, planning_problem, plot_limits=[350, 500, -30, 0])
            # plot_predicted_trajectory(scenario, ego_vehicle)

    def step(self, action_id) -> bool:
        self.observation, reward, self.done, self.truncated, info = self._env.step(
            action_id
        )
        self.time_step += 0
        self._env.render()
        self._commonroad_ids = {-1}
        if self.done or self.truncated:
            self._env.close()
            return False
        return True

    @staticmethod
    def make() -> "HighwayEnvScenario":
        env_config = {
            "highway-v0": {
                "observation": {"type": "TimeToCollision", "horizon": 10},
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
                "initial_lane_id": None,
                "ego_spacing": 4,
            }
        }
        return HighwayEnvScenario(env_config)


if __name__ == "__main__":
    scenario_ = HighwayEnvScenario.make()
    commonrad_scenario, _, planning_problem_ = scenario_.commonroad_representation

    road_network = RoadNetwork.from_lanelet_network_and_position(
        commonrad_scenario.lanelet_network,
        planning_problem_.initial_state.position,
    )

    ego_lane_network = EgoLaneNetwork.from_route_planner(
        commonrad_scenario.lanelet_network, planning_problem_, road_network
    )

    scenario_.step(1)
    scenario_.plot(plot_commonroad=True)
    scenario_.highway_env_representation.close()
