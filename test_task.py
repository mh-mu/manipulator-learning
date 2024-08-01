import manipulator_learning.sim.envs as manlearn_envs
from icecream import ic
import time

env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')()

obs = env.reset()
for i in range(100):
    next_obs, rew, done, info = env.step(env.action_space.sample())

    #ic(next_obs)
    time.sleep(1)