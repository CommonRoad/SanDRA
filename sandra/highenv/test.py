import random
from typing import cast
import numpy as np
import gymnasium
import highway_env
from gymnasium.wrappers import RecordVideo
from matplotlib import pyplot as plt
# %matplotlib inline

def get_relevant_obstacles(environment) -> list:
    return cast(list, environment.road.close_vehicles_to(
        environment.ego, environment.PERCEPTION_DISTANCE, see_behind=True,
        sort=True
    ))

env_config = {
        'highway-v0':
        {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": 15,
                "see_behind": True,
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

env = gymnasium.make('highway-v0', render_mode='rgb_array')
env.configure(env_config['highway-v0'])
result_prefix = f"highway_1"
env = RecordVideo(env, 'results', name_prefix=result_prefix)
env.unwrapped.set_record_video_wrapper(env)
test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31]
seed = random.choice(test_list_seed)
obs, info = env.reset(seed=seed)
env.render()
env.reset()
print(env.spec.id)
obstacles = get_relevant_obstacles(env)
for _ in range(3):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()