import os
import cv2
import numpy as np
import zarr
from termcolor import cprint
from icecream import ic

# Define directories and file names
data_source_dir = 'demonstrations/ThingInsertImage01'
img_source_dir = 'demonstrations/ThingInsertImage01/img/00000'
depth_source_dir = 'demonstrations/ThingInsertImage01/depth/00000'
data_target_dir = 'demonstrations/dp3.zarr'
filename = 'data.npz'

# Construct the full file path for the .npz file
input_file_path = os.path.join(data_source_dir, filename)

# Load the .npz file
npz_data = np.load(input_file_path, allow_pickle=True)

# print("Contents of the .npz file:")
# for array_name in npz_data.files:
#     array_data = npz_data[array_name]
#     print(f"Array name: {array_name}")
#     print(f"  Shape: {array_data.shape}")
#     print(f"  Data type: {array_data.dtype}")
#     print(f"  Example data (first 5 elements): {array_data.flatten()[:5]}")

# Ensure the output directory exists
os.makedirs(data_target_dir, exist_ok=True)

# Retrieve and reshape the data from .npz
state_data = npz_data['state_data']
traj_lens = npz_data['traj_lens']
traj_lens_including_last_obs = npz_data['traj_lens_including_last_obs']
valid_indices = npz_data['valid_indices']

# Extract and reshape the required subsets from `state_data`
action_arrays = state_data[:, [12, 13, 14, -1]]  # 13, 14, 15, 19 element for xyz and gripper open/close
state_arrays = state_data[:, :9]  # First 9 numbers
episode_ends_arrays = traj_lens

# Placeholder arrays for other data
num_samples = npz_data['state_data'].shape[0]
point_cloud_arrays = np.zeros((num_samples, 1024, 6)).astype(np.float32)

# read images
image_files = sorted([f for f in os.listdir(img_source_dir) if f.endswith('.png')])
image_list = []

for file_name in image_files:
    image_path = os.path.join(img_source_dir, file_name)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_list.append(img_rgb.transpose(2, 0, 1))

img_arrays = np.stack(image_list)

# read depths
depth_files = sorted([f for f in os.listdir(depth_source_dir) if f.endswith('.npy')])
depth_list = []

for file_name in depth_files:
    depth_path = os.path.join(depth_source_dir, file_name)
    depth_array = np.load(depth_path)
    depth_list.append(depth_array)

depth_arrays = np.stack(depth_list)

# Create a Zarr store and group
zarr_root = zarr.group(data_target_dir)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

# Create datasets in the Zarr group
compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
state_chunk_size = (100, state_arrays.shape[1])
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
action_chunk_size = (100, action_arrays.shape[1])

zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

cprint(f'-'*50, 'cyan')
# print shape
cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'Saved zarr file to {data_target_dir}', 'green')

# clean up
del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
del zarr_root, zarr_data, zarr_meta
