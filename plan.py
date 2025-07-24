from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_reach.utility import visualization as util_visual

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.utility.visualization import plot_reachable_sets
from sandra.commonroad.plan import ReactivePlanner
from sandra.commonroad.reach import ReachVerifier, VerificationStatus
from sandra.actions import LongitudinalAction, LateralAction

import matplotlib

print(matplotlib.get_backend())
matplotlib.use("TkAgg")


# scenario basic information
name_scenario = "DEU_MONAEast-2_4316_T-4341"
path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + ".xml"
scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
    lanelet_assignment=True
)
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]


# configuration
config = SanDRAConfiguration()

# road network
road_network = RoadNetwork.from_lanelet_network_and_position(
    scenario.lanelet_network,
    planning_problem.initial_state.position,
    consider_reversed=True,
)
ego_lane_network = EgoLaneNetwork.from_route_planner(
    scenario.lanelet_network,
    planning_problem,
    road_network,
)

# reachability analysis
reach_ver = ReachVerifier(scenario, planning_problem, config, ego_lane_network)

status = reach_ver.verify(
    [LongitudinalAction.DECELERATE, LateralAction.FOLLOW_LANE],
)

# plot the reachable set
# plot_reachable_sets(reach_ver.reach_interface, plot_limits=config.plot_limits)


# planning
if status == VerificationStatus.SAFE:
    planner = ReactivePlanner(config, scenario, planning_problem)
    planner.config_planner.debug.draw_traj_set = True
    planner.reset(reach_ver.reach_config.planning.CLCS)
    driving_corridor = reach_ver.reach_interface.extract_driving_corridors(
        to_goal_region=False
    )[0]

    _ = planner.plan(driving_corridor)

    # visualization
    planner.visualize(
        driving_corridor=driving_corridor,
        reach_interface=reach_ver.reach_interface,
    )
