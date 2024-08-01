import manipulator_learning.sim.envs as manlearn_envs
from icecream import ic


env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')()

obs = env.reset()
next_obs, rew, done, info = env.step(env.action_space.sample())

ic(next_obs)