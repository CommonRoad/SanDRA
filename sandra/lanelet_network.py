from typing import Optional

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet

from sandra.actions import LateralAction, LongitudinalAction

# The basic idea is that the lanelet network is a graph with 2 node types:
# 1. Normal Lanelet -> from this node, you can only go to your successor, adj_left, or adj_right (if they exist).
# 2. Intersection Incoming -> from this node, you can only go to the straight, left, or right outgoing Normal Lanelet node (if they exist).
# Now, whenever the driver takes an action like "left", "right", or "forward", the assumed route is that he wants to stay on the corresponding node and all its successors.
# As soon as the successor is an Intersection Incoming node, the route is terminated.
# Furthermore, for (e.g. left) lane change there are 3 possibilities:
# 1) change to the lane left to you
# 2) whenever there are multiple successors, follow the left most
# 3) whenever you encounter an intersection, turn left

Route = list[int]


class LaneletNode:
    def __init__(self, lanelet: Lanelet, lanelet_network: LaneletNetwork):
        self.lanelet = lanelet

        # determine node type:
        self.incoming_element = None
        for intersection in lanelet_network.intersections:
            for incoming in intersection.incomings:
                if self.lanelet.lanelet_id in incoming.incoming_lanelets:
                    self.incoming_element = incoming
                    break

        self.next_node_dict: dict[LateralAction, Optional[LaneletNode | int]] = {a: None for a in LateralAction}
        self.route_dict: dict[LateralAction, Optional[Route]] = {a: None for a in LateralAction}

    def is_normal(self) -> bool:
        return self.incoming_element is None

    def instantiate_next_nodes(self, lanelet_to_node_dict: dict) -> None:
        for key, value in self.next_node_dict.items():
            if value is None:
                continue
            self.next_node_dict[key] = lanelet_to_node_dict[value]

    def calculate_routes(self):
        for action in LateralAction:
            if action == LateralAction.KEEP_STRAIGHT:
                start = self
            else:
                start = self.next_node_dict[action]
            route = []

            while start is not None and start.lanelet not in route:
                route.append(start.lanelet.lanelet_id)
                start = start.next_node_dict[LateralAction.KEEP_STRAIGHT]
            self.route_dict[action] = route


class EgoCenteredLaneletNetwork:
    def __init__(self, lanelet_network: LaneletNetwork, ego_lane_id):
        frontier = {ego_lane_id}
        visited = set()
        self.lanelet_network = lanelet_network
        self.ego_node = None
        self.nodes: dict[int, LaneletNode] = {}

        while frontier:
            lanelet_id = frontier.pop()
            visited.add(lanelet_id)
            lanelet: Lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
            node = LaneletNode(lanelet, lanelet_network)

            if lanelet.adj_left:
                node.next_node_dict[LateralAction.CHANGE_LEFT] = lanelet.adj_left
                if lanelet.adj_left not in visited:
                    frontier.add(lanelet.adj_left)
            if lanelet.adj_right:
                node.next_node_dict[LateralAction.CHANGE_RIGHT] = lanelet.adj_right
                if lanelet.adj_right not in visited:
                    frontier.add(lanelet.adj_right)

            if node.is_normal():
                if len(lanelet.successor) == 1:
                    node.next_node_dict[LateralAction.KEEP_STRAIGHT] = lanelet.successor[0]
                    if lanelet.successor[0] not in visited:
                        frontier.add(lanelet.successor[0])
                elif len(lanelet.successor) > 1:
                    # TODO: add support for branching lanelets
                    pass
            else:
                if left_outgoings := node.incoming_element.successors_left:
                    assert len(left_outgoings) == 1, "Can not handle multiple intersection outgoings at the moment."
                    left_outgoing_id = next(iter(left_outgoings))
                    node.next_node_dict[LateralAction.TURN_LEFT] = left_outgoing_id
                    if left_outgoing_id not in visited:
                        frontier.add(left_outgoing_id)
                if right_outgoings := node.incoming_element.successors_right:
                    assert len(right_outgoings) == 1, "Can not handle multiple intersection outgoings at the moment."
                    right_outgoing_id = next(iter(right_outgoings))
                    node.next_node_dict[LateralAction.TURN_RIGHT] = right_outgoing_id
                    if right_outgoing_id not in visited:
                        frontier.add(right_outgoing_id)
                if straight_outgoings := node.incoming_element.successors_straight:
                    assert len(right_outgoings) == 1, "Can not handle multiple intersection outgoings at the moment."
                    straight_outgoing_id = next(iter(straight_outgoings))
                    node.next_node_dict[LateralAction.KEEP_STRAIGHT] = straight_outgoing_id
                    if straight_outgoing_id not in visited:
                        frontier.add(straight_outgoing_id)

            if self.ego_node is None:
                self.ego_node = node
            self.nodes[lanelet_id] = node

        for node in self.nodes.values():
            node.instantiate_next_nodes(self.nodes)
        for node in self.nodes.values():
            node.calculate_routes()

    def describe(self, dead_ends=False) -> str:
        ego_node = self.ego_node
        description = ""

        if ego_node.next_node_dict[LateralAction.TURN_LEFT] is not None or ego_node.next_node_dict[LateralAction.TURN_RIGHT] is not None:
            description += "You are currently approaching an intersection.\n"

        lane_count_ahead = 0
        for action in [LateralAction.KEEP_LEFT, LateralAction.KEEP_STRAIGHT, LateralAction.KEEP_RIGHT]:
            lane_count_ahead += int(ego_node.next_node_dict[action] is not None)
        if lane_count_ahead == 0 and dead_ends:
            description += "Ahead of you is a dead end.\n"
        elif lane_count_ahead > 1:
            description += f"Ahead of you, the current lane is branching into {lane_count_ahead} different lanes.\n"

        if ego_node.next_node_dict[LateralAction.CHANGE_LEFT] is None:
            description += f"There is no lane left of your current lane.\n"
        else:
            direction = "same" if self.ego_node.lanelet.adj_left_same_direction else "opposite"
            description += f"There is a {direction}-direction lane left of your current lane.\n"

        if ego_node.next_node_dict[LateralAction.CHANGE_RIGHT] is None:
            description += f"There is no lane right of your current lane.\n"
        else:
            direction = "same" if self.ego_node.lanelet.adj_right_same_direction else "opposite"
            description += f"There is a {direction}-direction lane right of your current lane.\n"
        return description

    def describe_lanelet(self, lanelet_id: int) -> str:
        action_to_location: dict[LateralAction, str] = {
            LateralAction.KEEP_STRAIGHT: "in your current lane",
            LateralAction.CHANGE_RIGHT: "in the lane to your right",
            LateralAction.CHANGE_LEFT: "in the lane to your left",
        }
        lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
        if self.ego_node.is_normal():
            while lanelet is not None:
                for action, route in self.ego_node.route_dict.items():
                    if lanelet.lanelet_id in route:
                        return action_to_location[action]
                lanelet_successors = lanelet.successor
                if lanelet_successors is not None and len(lanelet_successors) == 1:
                    lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_successors[0])
                elif lanelet_successors is not None and len(lanelet_successors) > 0:
                    for successor in lanelet_successors:
                        if successor in self.ego_node.route_dict[LateralAction.KEEP_STRAIGHT]:
                            return "in a successor lane"
                else:
                    break
        else:
            # TODO: Handle intersection scenarios
            pass
        return ""

    def lateral_actions(self) -> list[str]:
        return [a.value for a, b in self.ego_node.next_node_dict.items() if b is not None] + [LateralAction.KEEP_STRAIGHT.value]

    def longitudinal_actions(self) -> list[str]:
        return [x.value for x in LongitudinalAction]
