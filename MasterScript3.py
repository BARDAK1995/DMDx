import scipy
import scipy.integrate
from matplotlib import animation
from IPython.display import HTML
from matplotlib import pyplot as plt
from pydmd import DMD
import os
import pandas as pd
from pydmd.plotter import plot_summary
import numpy as np
from dmd_functions import *
import glob


caseName = "flowfield_snapshots_ref/"
folderLocation="Datas/" + caseName 


VisualMode = True
# Get all .npy files in the folder
npy_files = glob.glob(os.path.join(folderLocation, '*.npy'))

# Load each file
data_dict = {}
filename_list=[]
for file_path in npy_files:
    # Get the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.join(folderLocation, file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")
    # filename_list.append(file_name)
    # Load the .npy file
    data = np.load(file_path)
    # Store the data in a dictionary with the file name as the key

    # NpyFileIndex=0
    snapshotsNp = np.load(file_path)[1500:,:,:]
    # snapshotsNp=data_dict[filename_list[NpyFileIndex]][1000:,:,:]

    xdom=60
    ydom=2.5
    full_column_length = 75  # You may need to adjust this value

    mean_field = snapshotsNp.mean(axis=0)  # Shape: (y_dim, x_dim)
    snapshotsNp = snapshotsNp - mean_field  # Shape: (n_timestep, y_dim, x_dim)

    snapshots = [snapshotsNp[i] for i in range(snapshotsNp.shape[0])]
    X_DIM = snapshots[0].shape[1]
    Y_DIM = snapshots[0].shape[0]
    x1 = np.linspace(0, xdom, X_DIM)  # 200 points for length 12
    x2 = np.linspace(0, ydom, Y_DIM)  # 50 points for height 3
    x1grid, x2grid = np.meshgrid(x1, x2)
    time = np.linspace(0, snapshotsNp.shape[0], snapshotsNp.shape[0]+1)#micrsosec
    dt = time[1]-time[0]
    # # # DATAGENERATE______________________________________________________________________________
    if VisualMode:
        total_snapshots = len(snapshots)
        interval = max(1, (total_snapshots - 1) // 15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            scipy.linalg.svdvals(
                np.array([snapshot.flatten() for snapshot in snapshots]).T
            ),
            "o",
        )
        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Values')
        ax.set_title('Singular Value Distribution')
        plt.tight_layout()
        svd_plot_path = os.path.join(folder_path, 'svd_distribution.png')
        plt.savefig(svd_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"SVD distribution plot saved to: {svd_plot_path}")
    # # PLOT SOME FOR DEBUG______________________________________________________________________________
    if VisualMode:
        fig = plt.figure(figsize=(18, 12))
        selected_snapshots = snapshots[::interval][:16]
        selected_snapshots += [None] * (16 - len(selected_snapshots))
        for id_subplot, snapshot in enumerate(selected_snapshots, start=1):
            plt.subplot(4, 4, id_subplot)
            if snapshot is not None:
                plt.pcolor(x1grid, x2grid, snapshot.real, vmin=-1, vmax=1)
            else:
                plt.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=plt.gca().transAxes)
            # Add timestamp to each subplot
            timestamp = time[id_subplot * interval - interval]
            plt.title(f"t = {timestamp:.2f}")
        plt.tight_layout()
        snapshots_plot_path = os.path.join(folder_path, 'selected_snapshots.png')
        plt.savefig(snapshots_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Selected snapshots plot saved to: {snapshots_plot_path}")
    # PLOT SOME FOR DEBUG______________________________________________________________________________
    svd_rank = 45
    dmd = DMD(svd_rank=svd_rank, tlsq_rank=0, exact=True, opt=True,forward_backward=True,sorted_eigs='abs')
    dmd.fit(snapshots)

    dmd.dmd_time['tend'] = time[-1]
    dmd.dmd_time['dt'] = dt
    dmd_states = [state.reshape(x1grid.shape) for state in dmd.reconstructed_data.T]
    aspect_ratio = x1grid.shape[1] / x1grid.shape[0]
    fig_width = 10
    fig_height = fig_width / aspect_ratio




    # # Compute IntegralofField_VERIFICCATIOn______________________________________________________________________________
    # compute_integral = scipy.integrate.trapezoid
    # OG_statesIntegrateNP = np.array(snapshots)[:,:,:]
    # OG_statesIntegrate = [OG_statesIntegrateNP[i] for i in range(OG_statesIntegrateNP.shape[0])]
    # original_int = [compute_integral(compute_integral(snapshot)).real for snapshot in OG_statesIntegrate]

    # dmd_statesIntegrateNP = np.array(dmd_states)[:,:,:]
    # dmd_statesIntegrate = [dmd_statesIntegrateNP[i] for i in range(dmd_statesIntegrateNP.shape[0])]
    # dmd_int = [compute_integral(compute_integral(state)).real for state in dmd_statesIntegrate]

    # figure = plt.figure(figsize=(18, 5))
    # plt.plot(dmd.original_timesteps, original_int, "bo", label="original snapshots")
    # plt.plot(dmd.dmd_timesteps, dmd_int, "r.", label="dmd states")
    # plt.ylabel("Integral")
    # plt.xlabel("Time")
    # plt.grid()
    # leg = plt.legend()
    # # Compute IntegralofField_VERIFICCATIOn______________________________________________________________________________

    plot_summary(
        dmd,
        x=x1,
        y=x2,
        t=dmd.dmd_timesteps,
        d=1,
        continuous=False,
        snapshots_shape=(X_DIM,Y_DIM),  # This is correct
        index_modes=(0, 1, 2),
        filename=folderLocation + file_name +"/" + "summary123.png",
        figsize=(20, 8),  # Adjusted for better visibility of the wide aspect ratio
        dpi=200,
        mode_cmap='RdBu_r',
        max_eig_ms=15,
        title_fontsize=16,
        label_fontsize=14,
        plot_semilogy=True,
        order='F'  # Try 'F' order if 'C' doesn't work
    )

    mode_order = np.argsort(-np.abs(dmd.amplitudes))
    # Reorder modes, eigenvalues, and dynamics
    sorted_modes = dmd.modes[:, mode_order]
    sorted_eigs = dmd.eigs[mode_order]
    sorted_dynamics = dmd.dynamics[mode_order]*(10**4) #khz
    sorted_amplitudes = dmd.amplitudes[mode_order]
    sorted_frequencies=dmd.frequency[mode_order]*(10**4) #khz
    
    total_frames = len(dmd_states)-1
    duration = 20 #seconds
    fps = total_frames / duration
    time_per_frame = 1e-7


    # Call the updated function
    plot_summaryNEW(
        dmd,
        x=x1,
        y=x2,
        t=dmd.dmd_timesteps,
        snapshots_shape=(Y_DIM, X_DIM),
        modes_per_plot=3,
        filename=folderLocation + file_name+"/" + "Detail.png",
        figsize=(20, 8),
        dpi=200,
        mode_cmap='RdBu_r',
        title_fontsize=16,
        label_fontsize=14,
        order='C',
        time_per_frame=time_per_frame  # Pass the time per frame
    )



    total_frames = len(dmd_states)-1
    duration = 20 #seconds
    fps = total_frames / duration
    time_per_frame = 1e-7



    video_output_dir = os.path.join(folderLocation, file_name, "videos")
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)

    if VisualMode:
        create_animation(
            data=dmd_states, 
            x1grid=x1grid, 
            x2grid=x2grid, 
            video_output_dir=video_output_dir,
            plot_name='DMD Reconstruction',
            update_func=update,
            display_video=True,
            save_video=True
        )

        # For original data
        create_animation(
            data=snapshots, 
            x1grid=x1grid, 
            x2grid=x2grid, 
            video_output_dir=video_output_dir,
            plot_name='Original Data',
            update_func=update_original,
            display_video=False,
            save_video=True
        )
        create_comparison_animation(
            snapshots=snapshots,
            dmd_states=dmd_states,
            x1grid=x1grid,
            x2grid=x2grid,
            video_output_dir=video_output_dir,
            plot_name='Ndensity',
            nModes=svd_rank,  # Assuming svd_rank is defined elsewhere in your code
            display_video=True,
            save_video=True
        )

