import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers import EnvCompatibility, StepAPICompatibility
from stable_baselines3.common.env_util import make_vec_env
import manipulator_learning.sim.envs as manlearn_envs
from stable_baselines3.common.evaluation import evaluate_policy
from icecream import ic
import cv2
# Custom callback for evaluation and video recording
class EvalVideoCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=0):
        super(EvalVideoCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Start video recording
            frames = []
            
            # Evaluate the agent
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                episode_reward = 0
                done = False
                obs, _ = self.eval_env.reset()
                img, depth = self.eval_env.render()
                while not done:
                    frames.append(img)
                    action, _ = self.model.predict(obs, deterministic=True)
                    #obs, reward, done, _ = self.eval_env.step(action)
                    #obs, rewards, dones, info = env.step(action) #this is gym format
                    obs, reward, done, truncated, info = self.eval_env.step(action) #this is gymnasium
                    img, depth = self.eval_env.render()
                    episode_reward += reward
                episode_rewards.append(episode_reward)
            

            # Save video using OpenCV
            ic(frames[0].shape)
            height, width, layers = frames[0].shape
            video_name = f"videos/rl-video-{self.n_calls}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video.release()

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            # # Log to wandb
            # wandb.log({
            #     "eval/mean_reward": mean_reward,
            #     "eval/std_reward": std_reward,
            #     "eval/best_mean_reward": max(self.best_mean_reward, mean_reward),
            # }, step=self.n_calls)
            
            # Update best mean reward
            # if mean_reward > self.best_mean_reward:
            #     self.best_mean_reward = mean_reward
            #     # Save best model
            #     self.model.save(f"best_model_{wandb.run.id}")
            
            self.last_mean_reward = mean_reward
            
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.n_calls}, " 
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Best mean reward: {self.best_mean_reward:.2f}")
        
        return True

def main():

    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 1e4,
        "env_name": "pb_insertion",
        "eval_every": 1e3,
        "n_eval_episodes": 5,
        "video_length": 1000,
    }

    # run = wandb.init(
    #     project="insertion_test",
    #     name ="test",
    #     config=config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     monitor_gym=True,  # auto-upload the videos of agents playing the game
    #     save_code=True,  # optional
    # )

    # env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos','contact_force'))
    env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos'))
    env = EnvCompatibility(env, 'none')
    check_env(env)
    #env = DummyVecEnv([lambda: env])
    
    # model
    model = PPO(config['policy_type'], env, verbose=1)

    # Create the callbacks and eval env
    eval_env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','grip_pos', 'prev_grip_pos'))
    eval_env = EnvCompatibility(eval_env, 'none')
    check_env(eval_env)
    #eval_env = DummyVecEnv([lambda: eval_env])
    eval_env.render_mode = "rgb_array"
    
    # eval_env = VecVideoRecorder(eval_env, "videos", record_video_trigger=lambda x: x % config['eval_every'] == 0, video_length=config["video_length"])
    
    # Set up callbacks
    eval_video_callback = EvalVideoCallback(eval_env, eval_freq=config['eval_every'], n_eval_episodes=config["n_eval_episodes"])

    # wandb_callback = WandbCallback(
    #     gradient_save_freq=0,
    #     model_save_path=f"wandb/{wandb.run.id}",
    #     verbose=2,
    # )

    callback = CallbackList([eval_video_callback]) #, wandb_callback])

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback
    )

    print(f'starts training for {config["total_timesteps"]} steps')

    model_save_path = os.path.join('data_rl', config['env_name'], 'test')
    model.save(model_save_path)

   # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()