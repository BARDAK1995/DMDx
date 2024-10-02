import os
import numpy as np
import pandas as pd


def ReadSnapshot(snapshot_file, full_column_length):
    df = pd.read_csv(snapshot_file)
    # Extract probe_index and Nparticles
    probe_index = df['probe_index'].values
    nparticles = df['nparticles'].values
    temps = df['temperature'].values
    Uvels = df['u_vel'].values
    Vvels = df['v_vel'].values
    Pressures = df['pressure'].values
    # Calculate the dimensions of the full field
    num_probes = len(probe_index)
    max_probe_index = np.max(probe_index)
    full_row_length = (max_probe_index // full_column_length) + 1
    # Create a 2D array to hold the full field data
    N_field = np.full((full_column_length, full_row_length), np.nan)
    T_field = np.full((full_column_length, full_row_length), np.nan)
    U_field = np.full((full_column_length, full_row_length), np.nan)
    V_field = np.full((full_column_length, full_row_length), np.nan)
    P_field = np.full((full_column_length, full_row_length), np.nan)
    # Calculate 2D indices from probe_index (column-major order)
    col_indices = probe_index % full_column_length
    row_indices = probe_index // full_column_length
    # Populate the 2D array
    N_field[col_indices, row_indices] = nparticles
    T_field[col_indices, row_indices] = temps
    U_field[col_indices, row_indices] = Uvels
    V_field[col_indices, row_indices] = Vvels
    P_field[col_indices, row_indices] = Pressures
    # Create a mask for the valid data points
    valid_mask = ~np.isnan(N_field)
    # Find the bounds of the valid data
    valid_rows, valid_cols = np.where(valid_mask)
    min_row, max_row = valid_rows.min(), valid_rows.max()
    min_col, max_col = valid_cols.min(), valid_cols.max()
    # Extract the window with valid data
    N_data = N_field[min_row:max_row+1, min_col:max_col+1]
    T_data = T_field[min_row:max_row+1, min_col:max_col+1]
    U_data = U_field[min_row:max_row+1, min_col:max_col+1]
    V_data = V_field[min_row:max_row+1, min_col:max_col+1]
    P_data = P_field[min_row:max_row+1, min_col:max_col+1]
    DICT = {
    'N_data': N_data,
    'T_data': T_data,
    'U_data': U_data,
    'V_data': V_data,
    'P_data': P_data
    }
    return DICT

def read_snapshots(snapshot_folder, full_column_length):
    # Get list of all CSV files in the snapshot folder
    snapshot_files = sorted([f for f in os.listdir(snapshot_folder) if f.endswith('.csv')])
    n_snapshots = len(snapshot_files)
    first_snapshot = ReadSnapshot(os.path.join(snapshot_folder, snapshot_files[0]), full_column_length)['N_data']
    ysize, xsize = first_snapshot.shape
    N_snapshots = np.zeros((n_snapshots, ysize, xsize))
    T_snapshots = np.zeros((n_snapshots, ysize, xsize))
    U_snapshots = np.zeros((n_snapshots, ysize, xsize))
    V_snapshots = np.zeros((n_snapshots, ysize, xsize))
    P_snapshots = np.zeros((n_snapshots, ysize, xsize))

    for i, file in enumerate(snapshot_files):
        print(i)
        snapshotdict = ReadSnapshot(os.path.join(snapshot_folder, file), full_column_length)
        N_snapshots[i] = snapshotdict['N_data']
        T_snapshots[i] = snapshotdict['T_data']
        U_snapshots[i] = snapshotdict['U_data']
        V_snapshots[i] = snapshotdict['V_data']
        P_snapshots[i] = snapshotdict['P_data']

    N_snapshots = np.nan_to_num(N_snapshots)  # Replace NaN with 0
    # np.copyto(T_snapshots, 0, where=np.isnan(T_snapshots))
    T_snapshots = np.nan_to_num(T_snapshots)  # Replace NaN with 0
    # np.copyto(T_snapshots, 0, where=np.isnan(T_snapshots))
    U_snapshots = np.nan_to_num(U_snapshots)  # Replace NaN with 0
    # np.copyto(U_snapshots, 0, where=np.isnan(U_snapshots))
    V_snapshots = np.nan_to_num(V_snapshots)  # Replace NaN with 0
    # np.copyto(V_snapshots, 0, where=np.isnan(V_snapshots))
    P_snapshots = np.nan_to_num(P_snapshots)  # Replace NaN with 0
    # np.copyto(P_snapshots, 0, where=np.isnan(P_snapshots))
    return N_snapshots,T_snapshots,U_snapshots,V_snapshots,P_snapshots


xdom=60
ydom=2.5
snapshot_folder = "snapshots"
full_column_length = 75  # You may need to adjust this value

try:
    snapshotsNp = np.load('masterMS.npy')
    n_snapshots = snapshotsNp.shape[0]
    print(f"Snapshots array loaded from 'masterMS.npy' with shape {snapshotsNp.shape}.")
except FileNotFoundError:
    print("Snapshots file not found in '' directory. Reading data from .tec files...")
    N_snapshots,T_snapshots,U_snapshots,V_snapshots,P_snapshots = read_snapshots(snapshot_folder, full_column_length)
    n_snapshots = N_snapshots.shape[0] 
    np.save('N_snapshots.npy', N_snapshots)
    np.save('T_snapshots.npy', T_snapshots)
    np.save('U_snapshots.npy', U_snapshots)
    np.save('V_snapshots.npy', V_snapshots)
    np.save('P_snapshots.npy', P_snapshots)

    # Save all arrays in a single compressed .npz file
    np.savez_compressed('flowfield_snapshots_compressed.npz', 
                        N_snapshots=N_snapshots, 
                        T_snapshots=T_snapshots, 
                        U_snapshots=U_snapshots, 
                        V_snapshots=V_snapshots, 
                        P_snapshots=P_snapshots)
    print("Snapshots array saved as 'masterMS.npy'.")
 
# data = np.load('flowfield_snapshots_compressed.npz')
# N_snapshots = data['N_snapshots']
# T_snapshots = data['T_snapshots']
# U_snapshots = data['U_snapshots']
# V_snapshots = data['V_snapshots']
# P_snapshots = data['P_snapshots']

