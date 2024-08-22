import os
import numpy as np


data_source_dir = 'demonstrations/ThingInsertImage01'
data_target_dir = ''

filename = 'data.npz'

# Construct the full file path
file_path = os.path.join(data_source_dir, filename)

# Using a context manager to read the .npz file
with np.load(file_path, allow_pickle=True) as data:
    # List all available arrays in the .npz file
    print("Available arrays:", data.files)
    
    # Access and print each array
    for array_name in data.files:
        array = data[array_name]
        print(f"Array name: {array_name}")
        # print("Array content:\n", array)
        print("Array shape:", array.shape)
        print("Array dtype:", array.dtype)
