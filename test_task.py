import manipulator_learning.sim.envs as manlearn_envs
from icecream import ic
import time
import numpy as np
import matplotlib.pyplot as plt

#env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos','force_torque'))
#env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos'))
#env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImageCustom')(state_data = ('pos','contact_force'), gripper_control_method='dp')
env = getattr(manlearn_envs, 'ThingInsertImage')(state_data = ('pos','obj_pose', 'contact_force'), gripper_control_method='bool_p') #, render_opengl_gui=True, force_pb_direct=False)

for i in range(100):
    obs = env.reset()
    time.sleep(5)
for i in range(100):
    a = env.action_space.sample()
    #a = np.zeros(6)
    a = np.array([0, 0.,-0.1,0.,0.,0.]) #,-1])
    # # if i < 20:
    # #     a = np.array([-0.0,0,-0.0,0.,0,0,0.1])
    # # else: 
    # #     a = np.array([-0.0,0,-0.0,0.,0,0,-0.1])
    next_obs, rew, done, info = env.step(a)
    ic(rew)

    ic(next_obs)
    #ic(next_obs['img'].shape)

    #image_array = next_obs['img']
    #plt.imsave('image.png', image_array)

    time.sleep(0.1)