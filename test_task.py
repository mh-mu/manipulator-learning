import manipulator_learning.sim.envs as manlearn_envs
from icecream import ic
import time
import numpy as np
import matplotlib.pyplot as plt

#env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos','force_torque'))
#env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos'))
env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','contact_force'), gripper_control_method='dp')
obs = env.reset()

for i in range(100):
    a = env.action_space.sample()
    a = np.array([-0.0,0,-0.0,0.,0,0,0.01])
    next_obs, rew, done, info = env.step(a)

    # ic(next_obs['obs'])
    ic(next_obs['img'].shape)

    image_array = next_obs['img']
    #plt.imsave('image.png', image_array)

    time.sleep(0.1)