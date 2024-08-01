import manipulator_learning.sim.envs as manlearn_envs
from icecream import ic
import time
import numpy as np

env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos','force_torque'))
#env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos'))
obs = env.reset()
for i in range(50):
    a = env.action_space.sample()
    a = np.array([-0.1,0,-0.02,0,0,0,0])
    next_obs, rew, done, info = env.step(a)

    ic(next_obs['obs'])
    time.sleep(0.2)