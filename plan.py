from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_reach.utility import visualization as util_visual

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT
from sandra.common.road_network import RoadNetwork, EgoLaneNetwork
from sandra.rules import InterstateRule
from sandra.utility.visualization import plot_reachable_sets
from sandra.commonroad.plan import ReactivePlanner
from sandra.commonroad.reach import ReachVerifier, VerificationStatus
from sandra.actions import LongitudinalAction, LateralAction

import matplotlib

print(matplotlib.get_backend())
matplotlib.use("TkAgg")


# scenario basic information
name_scenario = "DEU_Goeppingen-37_1_T-4"
name_scenario = "DEU_Goeppingen-37_1_T-5" # for set + rules

path_scenario = PROJECT_ROOT + "/scenarios/" + name_scenario + ".xml"
scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(
    lanelet_assignment=True
)

planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

# configuration
config = SanDRAConfiguration()
config.a_lim = 0.2
config.h = 25
config.use_sonia = True
config.dt = 0.1
config.use_rules_in_reach = True

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
    rules=[InterstateRule.RG_1, InterstateRule.RG_3],
)

# plot the reachable set

util_visual.plot_scenario_with_reachable_sets(
    reach_ver.reach_interface, save_gif=True, plot_limits=config.plot_limits
)

plot_reachable_sets(reach_ver.reach_interface, plot_limits=config.plot_limits)


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
