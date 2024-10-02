import os
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def read_probe_file(args):
    file_path, probe_index_map, n_temporal, n_columns = args
    file_name = os.path.basename(file_path)
    probe_index = int(file_name.split('_')[-1])
    df = pd.read_csv(file_path, delim_whitespace=True, header=None,
                     names=['timestep', 'nparticles', 'u_vel', 'v_vel', 'z_vel', 'temperature', 'pressure'])
    num_rows = len(df)
    if num_rows > n_temporal:
        # Take the last n_temporal rows
        df = df.tail(n_temporal)
        data = df.values
    elif num_rows < n_temporal:
        # Pad with NaNs at the top
        pad_rows = n_temporal - num_rows
        pad_array = np.full((pad_rows, n_columns), np.nan)
        data = np.vstack((pad_array, df.values))
    else:
        data = df.values
    array_index = probe_index_map[probe_index]
    return array_index, data

def write_snapshot(args):
    t, snapshot_data, probe_indices, output_dir = args
    df = pd.DataFrame(snapshot_data, columns=['timestep', 'nparticles', 'u_vel', 'v_vel',
                                              'z_vel', 'temperature', 'pressure'])
    df.insert(0, 'probe_index', probe_indices)
    df.to_csv(os.path.join(output_dir, f'snapshot_{t:04d}.csv'), index=False)
    return t  # Return t for progress tracking

input_dir = '.'  # Current directory
output_dir = 'snapshots'
os.makedirs(output_dir, exist_ok=True)

# Get all probe files and sort them correctly
probe_files = sorted([f for f in os.listdir(input_dir) if f.startswith('Pxx_')],
                     key=natural_sort_key)

# Determine spatial and temporal dimensions
n_spatial = len(probe_files)
n_temporal = 4000  # As specified in the problem

print(f"Number of probe files: {n_spatial}")
print(f"Temporal dimension: {n_temporal}")

# Determine the number of columns in the data
sample_file = probe_files[0]
sample_data = pd.read_csv(os.path.join(input_dir, sample_file), delim_whitespace=True, header=None, nrows=1)
n_columns = len(sample_data.columns)

# Create a mapping of probe indices to array positions
probe_index_map = {}
for i, file in enumerate(probe_files):
    probe_index = int(file.split('_')[-1])
    probe_index_map[probe_index] = i

# Prepare list of file paths and probe_index_map for processing
file_args = [(os.path.join(input_dir, file), probe_index_map, n_temporal, n_columns) for file in probe_files]

# Allocate the main array
main_array = np.zeros((n_spatial, n_temporal, n_columns), dtype=np.float64)

# Process files in parallel with progress reporting
print("Processing files in parallel:")
with Pool(processes=4) as pool:
    total_files = len(file_args)
    for idx, result in enumerate(pool.imap_unordered(read_probe_file, file_args), 1):
        array_index, data = result
        main_array[array_index] = data

        # Simple progress indicator
        if idx % 10 == 0 or idx == n_spatial:
            print(f"  Processed {idx}/{n_spatial} files")

print("Processing complete.")

# Prepare data for snapshot writing
probe_indices = [k for k, v in sorted(probe_index_map.items(), key=lambda item: item[1])]

# Prepare arguments for snapshot writing
snapshot_args = [
    (t, main_array[:, t, :], probe_indices, output_dir)
    for t in range(n_temporal)
]

# Write snapshot files sequentially
print("Writing snapshots:")
total_snapshots = n_temporal
for idx, args in enumerate(snapshot_args, 1):
    write_snapshot(args)

    # Simple progress indicator
    if idx % 20 == 0 or idx == n_temporal:
        print(f"  Written snapshot {idx}/{n_temporal}")

print("All snapshots written.")

