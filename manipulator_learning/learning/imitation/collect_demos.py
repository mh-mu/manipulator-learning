import argparse
from datetime import datetime

from manipulator_learning.sim.envs import *

from manipulator_learning.learning.imitation.device_utils import CollectDevice
from manipulator_learning.learning.imitation.collect_utils import *
# import manipulator_learning.learning.data.img_depth_dataset as img_depth_dataset
import manipulator_learning.learning.data.img_dataset as img_dataset
from manipulator_learning.learning.utils.absorbing_state import Mask

from icecream import ic
import gymnasium as gym

parser = argparse.ArgumentParser()
parser.add_argument('--environment', type=str, default="ThingReachingXYState")
parser.add_argument('--directory', type=str, default='/tmp/demonstrations')
parser.add_argument('--demo_name', type=str, default=datetime.now().strftime("%y-%m-%d_%H-%M-%S"))
parser.add_argument('--device', type=str, default='gamepad')
parser.add_argument('--collect_interval', type=int, default=0)
parser.add_argument('--action_multiplier', type=float, default=.3)
parser.add_argument('--enforce_real_time', action='store_true', default=True,
                    help='if true, attempt to play environment at real time, based on .01s per'
                         'low level timestep and chosen n_substeps')
parser.add_argument('--ros_env_vr_teleop_help', action='store_true', default=False,
                    help='if true, when reset is called, user controls robot to assist in resetting the env'
                         'before the env is fully reset (e.g. to place objs in new positions).')
parser.add_argument('--show_opengl', action='store_true', help="show regular pybullet opengl renderer")
parser.add_argument('--save_on_success_only', action='store_true', help="always save on and only on success")

args = parser.parse_args()


env = globals()[args.environment](action_multiplier=1.0, state_data=('pos','grip_pos','contact_force'), \
                                    egl=not args.show_opengl, render_opengl_gui=args.show_opengl)
dev = CollectDevice(args.device, valid_t_dof=env.env.gripper.valid_t_dof, valid_r_dof=env.env.gripper.valid_r_dof,
                    output_grip=env.grip_in_action, action_multiplier=args.action_multiplier)
if env.env.gripper.control_method == 'dp' and dev.dev_type == 'keyboard':
    dev.action_multiplier = .003
if args.device == 'vr':
    dev.dev.vel_pid.Kp = 5.0

# handle images in obs
obs_is_dict = False
if type(env.observation_space) == gym.spaces.dict.Dict:
    obs_is_dict = True
    img_traj_data = []

env.seed()
ic(obs_is_dict)

data_dir = os.path.join(args.directory, args.demo_name)

# create and/or read existing dataset -- if existing, user must ensure dataset matches env
act_dim = env.action_space.shape[0]
if obs_is_dict:
    obs_shape = env.observation_space.spaces['obs'].shape[0]
else:
    obs_shape = env.observation_space.shape[0]
ds = img_dataset.Dataset(data_dir, state_dim=obs_shape, act_dim=act_dim)

# collection variables
status_dict = {'record': False, 'num_demos': 0,
               't': 0, 'success': False}
traj_data = []
data = []
traj_lens = []
ts = 0
dataset_total_ts = 0
fr = 0
fr_since_collect = args.collect_interval
time_per_frame = env.env._time_step * 10  # always 10 substeps in sim
ep_r = 0
old_data = None

# load file if it exists
traj_lens_filename = data_dir + '/traj_lens.npy'
if not obs_is_dict:
    os.makedirs(data_dir, exist_ok=True)
    np_filename = data_dir + '/data.npy'
    if os.path.exists(np_filename):
        traj_lens = np.load(traj_lens_filename).tolist()
        old_data = np.load(np_filename)
        dataset_total_ts = np.sum(traj_lens)

cur_base_pos = None
cur_pos = None

if args.ros_env_vr_teleop_help:
    env.set_reset_teleop_complete()
obs = env.reset()

while(True):
    frame_start = time.time()

    cancel, _, start, _, delete, success_fb_suc, success_fb_fail = dev.update_and_get_state()

    # mei
    save = False
    
    if delete:
        print('Deleting previous trajectory')
        if obs_is_dict:
            ds.remove_last_traj()
        else:
            if len(data) > 0:
                data.pop()
                traj_lens.pop()

    if env._control_type == 'v' or env._control_type == 'dp':
        if args.device == 'vr':
            obs_dict = env.env.gripper.receive_observation(ref_frame_pose=env.env.vel_control_frame,
                                                            ref_frame_vel=env.env.vel_ref_frame)
            cur_pos = np.concatenate([obs_dict['pos'], obs_dict['orient']])
        act = dev.get_ee_vel_action(cur_pos)
    elif env._control_type == 'p':
        #TODO get ee and base pose
        act = dev.get_ee_pos_action(None, None)

    next_obs, rew, done, info = env.step(act)
    save = done # save when an episode is done

    # mei
    dev.recording = True

    if dev.recording:
        ep_r += rew
    if info['done_success']:
        status_dict['success'] = True
    env.render()
    if dev.recording:
        if fr_since_collect == args.collect_interval:
            if obs_is_dict:
                # useful to ensure Q function is stationary on timeout
                # so even if env gives "done", this mask will be NOT_DONE on timeout
                if not (done or ts + 1 == env._max_episode_steps):
                    done_mask = Mask.NOT_DONE.value
                else:
                    done_mask = Mask.DONE.value
                # mei
                # traj_data.append(np.concatenate((obs['obs'], np.array(act).flatten(),
                #                                  np.array([rew]), np.array([done_mask]), np.array([done]) )))
                traj_data.append(np.concatenate((obs['obs'], np.array(act).flatten())))
                img_traj_data.append(obs['img'])
            else:
                traj_data.append(np.concatenate((obs, np.array(act).flatten(), np.array([rew]))))
            fr_since_collect = 0
        else:
            fr_since_collect += 1
        ts += 1
    else:
        env.ep_timesteps = 0  # don't start incrementing env time until recording starts
        ts = 0

    if cancel:
        traj_data = []
        if obs_is_dict:
            img_traj_data = []

    # otherwise, save is defined by device pressing reset
    if args.save_on_success_only:
        if done and dev.recording and info['done_success']:
            save = True
        else:
            save = False

    if save:
        if obs_is_dict:
            # add one more observation as final obs, with no action
            # mei
            # traj_data.append(np.concatenate([
            #     next_obs['obs'], np.zeros_like(act).flatten(), np.array([0]), np.array([done_mask]), np.array([done])
            # ]))
            # traj_data.append(np.concatenate([
            #     next_obs['obs'], np.zeros_like(act).flatten()]))
            # img_traj_data.append(next_obs['img'])

            # ds.append_traj_data_lists(traj_data, img_traj_data, final_obs_included=True)
            ds.append_traj_data_lists(traj_data, img_traj_data, final_obs_included=False)
        else:
            data.append(np.array(traj_data))
            traj_lens.append(ts)

            if old_data is None:
                np.save(np_filename, np.vstack(data))
                np.save(traj_lens_filename, np.vstack(traj_lens))
            else:
                np.save(np_filename, np.concatenate([old_data, np.vstack(data)]))
                np.save(traj_lens_filename, np.vstack(traj_lens))
        # dev.recording = False # mei
        traj_data = []
        if obs_is_dict:
            img_traj_data = []

    # if reset or done:
    if done:
        if args.device == 'vr':
            dev.dev.reset_ref_poses()
        print('Episode reward: %4.3f' % ep_r)
        obs = env.reset()

        fr_since_collect = args.collect_interval
        traj_data = []
        if obs_is_dict:
            img_traj_data = []
        ts = 0
        status_dict['success'] = False
        ep_r = 0

    status_dict['record'] = dev.recording
    if obs_is_dict:
        status_dict['num_demos'] = len(ds.data['traj_lens'])
    else:
        if traj_lens is not None:
            status_dict['num_demos'] = len(traj_lens)
    status_dict['t'] = len(traj_data)
    if fr % 10 == 0:
        print(status_dict, "Reward: %3.3f" % rew)
    fr += 1

    frame_time = time.time() - frame_start
    if args.enforce_real_time:
        leftover_time = time_per_frame - frame_time
        if leftover_time > 0:
            time.sleep(leftover_time)

    obs = next_obs