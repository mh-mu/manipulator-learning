import manipulator_learning.sim.envs as manlearn_envs
env = getattr(manlearn_envs, 'PandaPlayInsertTrayXYZState')()

obs = env.reset()
next_obs, rew, done, info = env.step(env.action_space.sample())