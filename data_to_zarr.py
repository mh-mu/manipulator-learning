import os
import numpy as np
import zarr

# Define directories and file names
data_source_dir = 'demonstrations/ThingInsertImage01'
data_target_dir = 'demonstrations/dp3.zarr'
filename = 'data.npz'

# Construct the full file path for the .npz file
input_file_path = os.path.join(data_source_dir, filename)

# Load the .npz file
npz_data = np.load(input_file_path, allow_pickle=True)

print("Contents of the .npz file:")
for array_name in npz_data.files:
    array_data = npz_data[array_name]
    print(f"Array name: {array_name}")
    print(f"  Shape: {array_data.shape}")
    print(f"  Data type: {array_data.dtype}")
    print(f"  Example data (first 5 elements): {array_data.flatten()[:5]}")

# Ensure the output directory exists
os.makedirs(data_target_dir, exist_ok=True)

# Create a Zarr store and group
zarr_root = zarr.group(save_dir)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

# Retrieve and reshape the data from .npz
state_data = npz_data['state_data']
traj_lens = npz_data['traj_lens']
traj_lens_including_last_obs = npz_data['traj_lens_including_last_obs']
valid_indices = npz_data['valid_indices']

# Extract and reshape the required subsets from `state_data`
action = state_data[:, -7:]  # Last 7 numbers
full_state = state_data[:, :]  # Entire state_data
state = state_data[:, :3]  # First 3 numbers

# Placeholder arrays for other data
combined_img = np.random.rand(num_samples, 6, 128, 128).astype(np.float32)  # Example placeholder data
depth = np.random.rand(num_samples, 128, 128).astype(np.float32)  # Example placeholder data
point_cloud = np.random.rand(num_samples, 1024, 6).astype(np.float32)  # Example placeholder data

# Create datasets in the Zarr group
compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
state_chunk_size = (100, state_arrays.shape[1])
full_state_chunk_size = (100, full_state_arrays.shape[1])
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
action_chunk_size = (100, action_arrays.shape[1])

zarr_data.create_dataset('data/action', data=action, dtype='f4', compressor=compressor)
zarr_data.create_dataset('data/combined_img', data=combined_img, dtype='f4', compressor=compressor)
zarr_data.create_dataset('data/depth', data=depth, dtype='f4', shape=(num_samples, 128, 128), compressor=compressor)
zarr_data.create_dataset('data/full_state', data=full_state, dtype='f4', compressor=compressor)
zarr_data.create_dataset('data/point_cloud', data=point_cloud, dtype='f4', compressor=compressor)
zarr_meta.create_dataset('data/state', data=state, dtype='f4', compressor=compressor)

# Create metadata dataset
episode_ends = traj_lens_including_last_obs[:5]  # Example data for episode_ends
zarr_group.create_dataset('meta/episode_ends', data=episode_ends, dtype='i8', shape=(5,))

print('Conversion completed.')
