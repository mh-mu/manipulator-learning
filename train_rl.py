import wandb
import numpy as np
from stable_baselines3 import PPO,SAC,TD3
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
import torchvision.models as v_models

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import FlattenExtractor
from gym import spaces

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 32):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space['img'].shape[0]
        
        # Define a CNN with a more gradual reduction in size and additional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce the spatial dimensions
            nn.Flatten(),
        )
        
        # Calculate the size of the flattened output
        with th.no_grad():
            sample_img = observation_space.sample()['img'].astype(float) / 255.0
            sample_img = th.as_tensor(sample_img).unsqueeze(0).float()  # Convert to float32
            n_flatten = self.cnn(sample_img).shape[1]
        
        # Linear layers to gradually reduce the feature size
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Convert the image to a float32 tensor and normalize (0-1 range)
        img = observations['img'].float() / 255.0
        features = self.cnn(img)
        return self.linear(features)


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CombinedExtractor, self).__init__(observation_space, features_dim)
        self.img_feature_extractor = CustomCNN(observation_space, features_dim)
        self.obs_feature_extractor = FlattenExtractor(observation_space.spaces['obs'])
        
        # Combine features from both extractors
        combined_features_dim = features_dim + self.obs_feature_extractor.features_dim
        self.linear = nn.Linear(combined_features_dim, features_dim)

    def forward(self, observations):
        img_features = self.img_feature_extractor(observations)
        obs_features = self.obs_feature_extractor(observations['obs'])
        combined_features = th.cat([img_features, obs_features], dim=1)
        return self.linear(combined_features)

    def _get_features_dim(self):
        return self.linear.out_features


class CustomCNN2(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 32):
        super(CustomCNN2, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space['img'].shape[0]
        
        # Define a CNN with a more gradual reduction in size and additional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the size of the flattened output
        with th.no_grad():
            sample_img = observation_space.sample()['img'].astype(float) / 255.0
            sample_img = th.as_tensor(sample_img).unsqueeze(0).float()  # Convert to float32
            n_flatten = self.cnn(sample_img).shape[1]
        
        # Linear layers to gradually reduce the feature size
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Convert the image to a float32 tensor and normalize (0-1 range)
        img = observations['img'].float() / 255.0
        features = self.cnn(img)
        return self.linear(features)

class CombinedExtractor2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CombinedExtractor2, self).__init__(observation_space, features_dim)
        self.img_feature_extractor = CustomCNN2(observation_space, features_dim)
        self.obs_feature_extractor = FlattenExtractor(observation_space.spaces['obs'])
        self.obs_linear = nn.Linear(self.obs_feature_extractor.features_dim, features_dim)

        # Combine features from both extractors
        combined_features_dim = features_dim*2
        self.linear = nn.Linear(combined_features_dim, features_dim)

    def forward(self, observations):
        img_features = self.img_feature_extractor(observations)
        obs_features = self.obs_linear(self.obs_feature_extractor(observations['obs']))
        combined_features = th.cat([img_features, obs_features], dim=1)
        return self.linear(combined_features)

    def _get_features_dim(self):
        return self.linear.out_features

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
            height, width, layers = frames[0].shape
            video_name = f"{wandb.run.dir}/rl-video-{self.n_calls}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video.release()

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            # Log to wandb
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward,
                "eval/best_mean_reward": max(self.best_mean_reward, mean_reward),
            }, step=self.n_calls)
            
            #Update best mean reward
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Save best model
                self.model.save(f"{wandb.run.dir}/best_model")
            
            self.last_mean_reward = mean_reward
            
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.n_calls}, " 
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Best mean reward: {self.best_mean_reward:.2f}")
        
        return True


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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    DEBUG = False

    config = {
        "policy_type": "MultiInputPolicy", #"MlpPolicy", # 
        "task":'insertion', #pickandinsertion
        "use_image": False,
        "algo": "PPO",
        "use_force": True,
        "task_name":'insertion_PPO_P_withForce',
        "total_timesteps": 5e6,
        "env_name": "pb_insertion",
        "eval_every": 5e4,
        "n_eval_episodes": 5,
        "video_length": 1000,
        "ee_rod_reward": 0.,
        "rod_box_reward": 10.,
        
    }

    if not DEBUG:
        run = wandb.init(
            project="insertion_test",
            name =config['task_name'],
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    if config['use_image']:
        if config['task'] == 'insertion':
            if config['use_force']:
                state_data = ('pos','grip_pos','contact_force')
            else:
                state_data = state_data = ('pos','grip_pos')
            task_name = 'ThingInsertImage'
        elif config['task'] == 'pickandinsertion':
            if config['use_force']:
                state_data = ('pos','grip_pos', 'prev_grip_pos','contact_force')
            else:
                state_data = state_data = ('pos','grip_pos', 'prev_grip_pos')
            task_name = 'ThingPickAndInsertSucDoneImage'
    else:
        task_name = 'ThingInsertGT'
        if config['use_force']:
            state_data = ('pos','grip_pos', 'obj_pose', 'contact_force')
        else:
            state_data = state_data = state_data = ('pos','grip_pos', 'obj_pose')

    env = getattr(manlearn_envs, task_name)(state_data = state_data, gripper_control_method='bool_p',ee_rod_reward = config['ee_rod_reward'],
        rod_box_reward = config['rod_box_reward'])
    env = EnvCompatibility(env, 'none')
    check_env(env)
    #env = DummyVecEnv([lambda: env])
    # model
    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor
    )
    #model = PPO(config['policy_type'], env, policy_kwargs=policy_kwargs, verbose=1)
    if config['use_image']:
        if config['algo'] == "PPO":
            model = PPO(config['policy_type'], env, policy_kwargs=policy_kwargs, verbose=1)
        elif config['algo'] == "SAC":
            model = SAC(config['policy_type'], env, policy_kwargs=policy_kwargs, verbose=1, buffer_size = 100000)
        elif config['algo'] == "RecurrentPPO":
            model = RecurrentPPO(config['policy_type'], env, policy_kwargs=policy_kwargs, verbose=1)
        elif config['algo'] == 'TD3':
            model = TD3(config['policy_type'], env, policy_kwargs=policy_kwargs, verbose=1, buffer_size = 100000)
    else:
        model = PPO(config['policy_type'], env, verbose=1)
    
    # # model = PPO(config['policy_type'], env, verbose=1)

    # policy = model.policy

    # # print(dir(policy))
    # # Get the feature extractor
    # feature_extractor = policy.features_extractor
    # print("Feature Extractor:")
    # print(feature_extractor)
    # print(count_parameters(feature_extractor))

    # mlp_extractor = policy.mlp_extractor
    # print("Mlp Extractor:")
    # print(mlp_extractor)
    # print(count_parameters(mlp_extractor))

    # #Get the action network
    # action_net = policy.action_net
    # print("\nAction Network:")
    # print(action_net)
    # print(count_parameters(action_net))

    # # Get the value network
    # value_net = policy.value_net
    # print("\nValue Network:")
    # print(value_net)
    # print(count_parameters(value_net))
    # # print('*******')
    # # print(policy.actor)
    # # print(policy.critic)

    # Create the callbacks and eval env
    eval_env = getattr(manlearn_envs, task_name)(state_data = state_data, gripper_control_method='bool_p',ee_rod_reward = config['ee_rod_reward'],
        rod_box_reward = config['rod_box_reward'])
    eval_env = EnvCompatibility(eval_env, 'none')
    check_env(eval_env)
    #eval_env = DummyVecEnv([lambda: eval_env])
    eval_env.render_mode = "rgb_array"

    if DEBUG:
        exit()

    # Set up callbacks
    eval_video_callback = EvalVideoCallback(eval_env, eval_freq=config['eval_every'], n_eval_episodes=config["n_eval_episodes"])

    if not DEBUG:
        wandb_callback = WandbCallback(
            gradient_save_freq=0,
            model_save_path=f"wandb/{wandb.run.id}",
            verbose=2,
        )
        custom_callback = CustomWandbCallback()

    callback = CallbackList([eval_video_callback, custom_callback])

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback
    )

    print(f'starts training for {config["total_timesteps"]} steps')

    model_save_path = f'{wandb.run.dir}/final_model.pth'
    model.save(model_save_path)

   # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()