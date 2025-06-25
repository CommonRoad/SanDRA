import os
import warnings
from typing import Optional, List

import numpy as np
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork

from commonroad_route_planner.route_planner import RoutePlanner

from commonroad_dc.geometry.geometry import CurvilinearCoordinateSystem


class Lane:
    """A series of ordered lanelets representing a lane."""

    def __init__(
        self,
        lane_id: int,
        lanelets: Optional[List[Lanelet]] = None,
        contained_ids: Optional[List[int]] = None,
    ):
        self.id = lane_id  # unique identifier
        self.lanelets = lanelets if lanelets is not None else []

        if contained_ids is not None:
            self.contained_ids = contained_ids
        else:
            self.contained_ids = [l.lanelet_id for l in self.lanelets]

        # Adjacency relationships
        self.left_adjacent: Optional[Lane] = None
        self.right_adjacent: Optional[Lane] = None

        self.clcs: Optional[CurvilinearCoordinateSystem] = None

    def __repr__(self):
        return f"Lane(id={self.id}, lanelets={self.contained_ids})"

    @property
    def center_vertices(self):
        """Center vertices along the lane."""
        return np.concatenate([lanelet.center_vertices for lanelet in self.lanelets])

    def add_lanelet(self, lanelet: Lanelet):
        """Add a lanelet to the end of the lane"""
        self.lanelets.append(lanelet)
        self.contained_ids.append(lanelet.lanelet_id)

    def insert_lanelet(self, index, lanelet):
        """Insert a lanelet at a specific position"""
        self.lanelets.insert(index, lanelet)
        self.contained_ids.insert(index, lanelet.lanelet_id)

    def set_left_adjacent(self, lane: "Lane") -> None:
        assert isinstance(lane, Lane), "left_adjacent must be a Lane instance"
        self.left_adjacent = lane

    def set_right_adjacent(self, lane: "Lane") -> None:
        assert isinstance(lane, Lane), "right_adjacent must be a Lane instance"
        self.right_adjacent = lane

    def contains(self, lanelet_id: int) -> bool:
        return lanelet_id in self.contained_ids


class RoadNetwork:
    """Road network consisting of lanes"""

    def __init__(self, lanes: Optional[list[Lane]] = None):
        self.lanes: list[Lane] = lanes if lanes else []

    @classmethod
    def from_lanelet_network_and_position(
        cls,
        lanelet_network: LaneletNetwork,
        position: np.asarray,
        consider_reversed=True,
    ) -> "RoadNetwork":

        initial_lanelet_ids = lanelet_network.find_lanelet_by_position([position])[0]

        # Collect predecessors of those lanelets (flattened)
        # --- same direction
        lanelet_ids_same_dir = []
        # --- reversed direction
        lanelet_ids_reversed = []

        # iterate the lanelet ids (for better clcs, one lanelet easier)
        for lanelet_id in initial_lanelet_ids:
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)

            if lanelet.predecessor:
                lanelet_ids_same_dir.extend(lanelet.predecessor)
            else:
                lanelet_ids_same_dir.extend([lanelet_id])

            # left/right adjacent lanelet in the same direction
            for side in ["adj_left", "adj_right"]:
                adj_id = getattr(lanelet, side)
                if not adj_id:
                    continue
                adj_lanelet = lanelet_network.find_lanelet_by_id(adj_id)
                same_dir_attr = f"{side}_same_direction"
                if getattr(lanelet, same_dir_attr):
                    if adj_lanelet.predecessor:
                        lanelet_ids_same_dir.extend(adj_lanelet.predecessor)
                    else:
                        lanelet_ids_same_dir.extend([adj_lanelet.lanelet_id])
                elif consider_reversed:
                    if adj_lanelet.successor:
                        lanelet_ids_reversed.extend(adj_lanelet.successor)
                    else:
                        lanelet_ids_reversed.extend([adj_lanelet.lanelet_id])

        def merge_lanelets(ids, merge_func):
            merged = []
            for lanelet_id in ids:
                lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
                # todo: param
                merged_lanelets, merge_jobs = merge_func(lanelet, lanelet_network, 500)
                if not merged_lanelets or not merge_jobs:
                    merged_lanelets = [lanelet]
                    merge_jobs = [[lanelet.lanelet_id]]
                for l, j in zip(merged_lanelets, merge_jobs):
                    merged.append((l, j))
            return merged

        lane_lanelets = []
        # same direction: get the successors
        lane_lanelets.extend(
            merge_lanelets(
                lanelet_ids_same_dir,
                Lanelet.all_lanelets_by_merging_successors_from_lanelet,
            )
        )
        # revered direction: get the predecessors
        lane_lanelets.extend(
            merge_lanelets(
                lanelet_ids_reversed,
                Lanelet.all_lanelets_by_merging_predecessors_from_lanelet,
            )
        )

        lanes = [
            Lane(lane_id=i, lanelets=[lane_element[0]], contained_ids=lane_element[1])
            for i, lane_element in enumerate(lane_lanelets)
        ]

        return cls(lanes=lanes)

    def get_lane_by_id(self, lane_id: int) -> Optional[Lane]:
        """Return the lane with the given lane ID, or None if not found."""
        return next((lane for lane in self.lanes if lane.id == lane_id), None)

    def get_lanes_by_lanelets(self, lanelets: list[Lanelet]) -> list[Lane]:
        """
        Returns a list of lanes that contain at least one of the specified Lanelet objects.

        Args:
            lanelets: A list of Lanelet objects.

        Returns:
            List of Lane instances that include at least one of the given lanelets.
        """
        lanelet_set = set(lanelets)
        return [
            lane for lane in self.lanes if any(l in lanelet_set for l in lane.lanelets)
        ]

    def get_unique_lane_by_lanelet_ids(self, lanelet_ids: list[int]) -> Optional[Lane]:
        """
        Returns the unique Lane that contains at least one of the given lanelet IDs.

        Args:
            lanelet_ids: A list of lanelet IDs (integers).

        Returns:
            The unique Lane instance that includes at least one of the given lanelet IDs,
            or None if no match is found.

        Raises:
            ValueError: If multiple lanes match the given lanelet IDs.
        """
        lanelet_id_set = set(lanelet_ids)
        matching_lanes = [
            lane
            for lane in self.lanes
            if lanelet_id_set.issubset(set(lane.contained_ids))
        ]

        if len(matching_lanes) == 0:
            return None
        if len(matching_lanes) > 1:
            warnings.warn(
                f"Multiple lanes found containing all given lanelet IDs: {lanelet_ids}. "
                f"Returning the first match with lane.id = {matching_lanes[0].id}."
            )

        return matching_lanes[0]

    def get_lanes_by_lanelet_ids(self, lanelet_ids: list[int]) -> list[Lane]:
        """
        Returns a list of lanes that contain at least one lanelet with an ID in the given list.

        Args:
            lanelet_ids: A list of lanelet IDs (integers).

        Returns:
            List of Lane instances that include at least one lanelet with a matching ID.
        """
        id_set = set(lanelet_ids)
        return [
            lane for lane in self.lanes if any(l in id_set for l in lane.contained_ids)
        ]


class EgoLaneNetwork:
    def __init__(self, road_network: RoadNetwork):
        self.road_network = road_network

        self.lane: Optional[Lane] = None
        # could be multiple
        self.lane_left_adjacent: Optional[List[Lane]] = None
        self.lane_right_adjacent: Optional[List[Lane]] = None

        # left right adjacent lane but in a reversed direction
        self.lane_left_reversed: Optional[List[Lane]] = None
        self.lane_right_reversed: Optional[List[Lane]] = None

    @property
    def neighbor_dict(self) -> dict[tuple[str, str], Optional[List[Lane]]]:
        return {
            ("same", "left"): self.lane_left_adjacent,
            ("opposite", "left"): self.lane_left_reversed,
            ("same", "right"): self.lane_right_adjacent,
            ("opposite", "right"): self.lane_right_reversed,
        }

    @classmethod
    def from_route_planner(
        cls,
        lanelet_network: LaneletNetwork,
        planning_problem: PlanningProblem,
        road_network: RoadNetwork,
    ) -> "EgoLaneNetwork":

        # get the high-level route
        route_planner = RoutePlanner(
            lanelet_network=lanelet_network, planning_problem=planning_problem
        )
        route_generator = route_planner.plan_routes()
        route = route_generator.retrieve_shortest_route()
        route_ids = route.lanelet_ids  # list[int]

        ego_lane = road_network.get_unique_lane_by_lanelet_ids(route_ids)
        if ego_lane is None:
            warnings.warn("Ego lane could not be identified from route IDs.")
            # todo: discussion
            ego_lanelet = lanelet_network.find_most_likely_lanelet_by_state(
                [planning_problem.initial_state]
            )
            ego_lane = road_network.get_lanes_by_lanelet_ids(ego_lanelet)[0]

        instance = cls(road_network=road_network)
        # building the curvilinear coordinate system
        ego_lane.clcs = CurvilinearCoordinateSystem(
            ego_lane.center_vertices, 20, 0.1, 5.0
        )
        instance.lane = ego_lane

        start_lanelet = lanelet_network.find_lanelet_by_id(route_ids[0])
        if start_lanelet.adj_left:
            if start_lanelet.adj_left_same_direction:
                instance.lane_left_adjacent = road_network.get_lanes_by_lanelet_ids(
                    [start_lanelet.adj_left]
                )
            else:
                instance.lane_left_reversed = road_network.get_lanes_by_lanelet_ids(
                    [start_lanelet.adj_left]
                )
        if start_lanelet.adj_right:
            if start_lanelet.adj_right_same_direction:
                instance.lane_right_adjacent = road_network.get_lanes_by_lanelet_ids(
                    [start_lanelet.adj_right]
                )
            else:
                instance.lane_right_reversed = road_network.get_lanes_by_lanelet_ids(
                    [start_lanelet.adj_right]
                )

        return instance


if __name__ == "__main__":
    from sandra.common.config import PROJECT_ROOT
    from sandra.utility.general import extract_scenario_and_planning_problem
    from sandra.utility.visualization import plot_road_network, plot_scenario

    scenario_paths = [
        "DEU_AachenAseag-1_80_T-99.xml",
        "DEU_AachenBendplatz-1_80_T-19.xml",
        "DEU_AachenFrankenburg-1_2120_T-39.xml",
        "DEU_AachenHeckstrasse-1_30520_T-539.xml",
        "DEU_LocationALower-11_10_T-1.xml",
    ]

    scenario_folder = os.path.join(PROJECT_ROOT, "scenarios")
    scenario, planning_problem = extract_scenario_and_planning_problem(
        scenario_folder + "/" + scenario_paths[-2]
    )
    plot_scenario(scenario, planning_problem)
    road_network = RoadNetwork.from_lanelet_network_and_position(
        scenario.lanelet_network, planning_problem.initial_state.position
    )
    lane_network = EgoLaneNetwork.from_route_planner(
        scenario.lanelet_network, planning_problem, road_network
    )

    plot_road_network(road_network, lane_network)
