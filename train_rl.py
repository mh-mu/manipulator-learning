from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import manipulator_learning.sim.envs as manlearn_envs
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode reward and length
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.locals['rewards'][0])
            self.episode_lengths.append(self.locals['infos'][0]['episode']['l'])
            wandb.log({
                "episode_reward": self.episode_rewards[-1],
                "episode_length": self.episode_lengths[-1],
                "mean_100ep_reward": np.mean(self.episode_rewards[-100:]),
                "mean_100ep_length": np.mean(self.episode_lengths[-100:])
            })
        return True


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 500,
    "env_name": "pb_insertion",
}

# run = wandb.init(
#     project="insertion",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     monitor_gym=True,  # auto-upload the videos of agents playing the game
#     save_code=True,  # optional
# )

env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos','contact_force'))
env = EnvCompatibility(env, 'none')
check_env(env)

# Create the callbacks
# wandb_callback = WandbCallback()
# custom_callback = CustomWandbCallback()

model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(
#     total_timesteps=config["total_timesteps"],
#     callback=[wandb_callback, custom_callback]
# )
for i in range(10):
    print(f'starts training for {config["total_timesteps"]} steps')

model.learn(
    total_timesteps=config["total_timesteps"])

model.save("ppo_cartpole_final")

# Test the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
# wandb.finish()