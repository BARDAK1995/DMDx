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
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_summaryNEW(
    dmd,
    *,
    x=None,
    y=None,
    t=None,
    d=1,
    continuous=False,
    snapshots_shape=None,
    index_modes=None,
    modes_per_plot=3,
    filename=None,
    order="C",
    figsize=(12, 8),
    dpi=200,
    tight_layout_kwargs=None,
    main_colors=("r", "b", "g"),
    mode_color="k",
    mode_cmap="bwr",
    dynamics_color="tab:blue",
    title_fontsize=14,
    label_fontsize=12,
    flip_continuous_axes=False,
    time_per_frame=None,  # Add time per frame in seconds
):
    """
    Generate multiple plots, each with 2 rows and `modes_per_plot` columns.
    Each plot contains:
    - The DMD modes specified by the `index_modes` parameter
    - The time dynamics that correspond with each plotted mode

    :param dmd: Fitted DMD instance.
    :type dmd: pydmd.DMDBase
    :param x: Points along the 1st spatial dimension where data has been collected.
    :type x: np.ndarray or iterable
    :param y: Points along the 2nd spatial dimension where data has been collected.
    :type y: np.ndarray or iterable
    :param t: The times of data collection, or the time-step between snapshots.
    :type t: {numpy.ndarray, iterable} or {int, float}
    :param d: Number of delays applied to the data passed to the DMD instance.
    :type d: int
    :param continuous: Whether or not the eigenvalues are continuous-time.
    :type continuous: bool
    :param snapshots_shape: Shape of the snapshots.
    :type snapshots_shape: iterable
    :param index_modes: Indices of the modes to plot after sorting by amplitude.
                        If None, all modes are plotted.
    :type index_modes: iterable
    :param modes_per_plot: Number of modes to plot per figure (default 3).
    :type modes_per_plot: int
    :param filename: If specified, the plots are saved with this base filename.
    :type filename: str
    :param order: The order to read/write elements ('C' or 'F').
    :type order: str
    :param figsize: Width, height in inches.
    :type figsize: iterable
    :param dpi: Figure resolution.
    :type dpi: int
    :param tight_layout_kwargs: Dictionary of `tight_layout` parameters.
    :type tight_layout_kwargs: dict
    :param main_colors: Colors used to denote eigenvalue, mode, dynamics associations.
    :type main_colors: iterable
    :param mode_color: Color used to plot the modes, if modes are 1-D.
    :type mode_color: str
    :param mode_cmap: Colormap used to plot the modes, if modes are 2-D.
    :type mode_cmap: str
    :param dynamics_color: Color used to plot the dynamics.
    :type dynamics_color: str
    :param title_fontsize: Fontsize used for subplot titles.
    :type title_fontsize: int
    :param label_fontsize: Fontsize used for axis labels.
    :type label_fontsize: int
    :param flip_continuous_axes: Whether or not to swap the real and imaginary axes on the continuous eigenvalues plot.
    :type flip_continuous_axes: bool
    :param time_per_frame: Time per frame in seconds.
    :type time_per_frame: float
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import os

    # Check that the DMD instance has been fitted.
    if dmd.modes is None:
        raise ValueError("You need to perform fit() first.")

    # By default, snapshots_shape is the flattened space dimension.
    if snapshots_shape is None:
        snapshots_shape = (len(dmd.snapshots) // d,)
    # If provided, snapshots_shape must contain 2 entries.
    elif len(snapshots_shape) != 2:
        raise ValueError("snapshots_shape must be None or 2-D.")

    # Get the actual rank used for the DMD fit.
    rank = len(dmd.eigs)

    # Sort eigenvalues, modes, and dynamics according to amplitude magnitude.
    mode_order = np.argsort(-np.abs(dmd.amplitudes))
    lead_eigs = dmd.eigs[mode_order]
    lead_modes = dmd.modes[:, mode_order]
    lead_dynamics = dmd.dynamics[mode_order]
    lead_amplitudes = dmd.amplitudes[mode_order]

    # Get frequencies of the modes
    frequencies = dmd.frequency[mode_order]  # Frequencies in Hz

    # Convert frequencies to kHz
    frequencies_kHz = frequencies * (1e+7)/(1000)#khz  # Assuming frequencies are in Hz

    # Only keep modes with positive frequencies
    positive_freq_indices = np.where(frequencies > 0)[0]
    if len(positive_freq_indices) == 0:
        print("No modes with positive frequencies found.")
        return

    # Filter modes with positive frequencies
    lead_eigs = lead_eigs[positive_freq_indices]
    lead_modes = lead_modes[:, positive_freq_indices]
    lead_dynamics = lead_dynamics[positive_freq_indices]
    lead_amplitudes = lead_amplitudes[positive_freq_indices]
    frequencies = frequencies[positive_freq_indices]
    frequencies_kHz = frequencies_kHz[positive_freq_indices]

    num_modes = len(lead_eigs)

    # If index_modes is None, plot all modes with positive frequencies
    if index_modes is None:
        index_modes = np.arange(num_modes)
    else:
        # Map the requested indices to the indices of the positive frequencies
        index_modes = np.array(index_modes)
        # Adjust index_modes to match the indices after filtering
        index_modes = np.intersect1d(index_modes, np.arange(num_modes))
        num_modes = len(index_modes)
        if num_modes == 0:
            print("No requested modes have positive frequencies.")
            return

    # Get time information
    if t is None:
        time = np.arange(lead_dynamics.shape[1])
    else:
        time = t

    # Convert time to seconds if time_per_frame is provided
    if time_per_frame is not None:
        time_in_seconds = time * time_per_frame
    else:
        time_in_seconds = time

    # Build the spatial grid for the mode plots.
    if x is None:
        x = np.arange(snapshots_shape[0])
    if len(snapshots_shape) == 2:
        if y is None:
            y = np.arange(snapshots_shape[1])
        xgrid, ygrid = np.meshgrid(x, y)

    # Determine the number of figures needed
    total_plots = int(np.ceil(num_modes / modes_per_plot))
    for plot_idx in range(total_plots):
        # Get the indices for this plot
        start_idx = plot_idx * modes_per_plot
        end_idx = min(start_idx + modes_per_plot, num_modes)
        modes_to_plot = index_modes[start_idx:end_idx]
        num_modes_in_plot = len(modes_to_plot)

        # Create the figure and axes
        fig, axes = plt.subplots(2, modes_per_plot, figsize=figsize, dpi=dpi)
        # If modes_per_plot == 1, axes might not be an array, so we ensure it is
        if modes_per_plot == 1:
            axes = axes.reshape(2, 1)

        # Plot modes in the first row
        for i, idx in enumerate(modes_to_plot):
            ax = axes[0, i]
            freq_kHz = frequencies_kHz[idx]
            ax.set_title(f"Mode {idx + 1}, Frequency: {freq_kHz:.2f} kHz", fontsize=title_fontsize)
            mode = lead_modes[:, idx]
            scale2Physical = np.mean(abs(lead_dynamics[idx].real))*2
            if len(snapshots_shape) == 1:
                # Plot modes in 1-D.
                ax.plot(x, mode.real, c=mode_color)
            else:
                # Plot modes in 2-D.
                mode_reshaped = mode.reshape(*snapshots_shape, order=order)
                vmax = np.abs(mode_reshaped.real).max()
                im = ax.pcolormesh(
                    xgrid,
                    ygrid,
                    mode_reshaped.real*scale2Physical ,
                    vmax=vmax*scale2Physical,
                    vmin=-vmax*scale2Physical,
                    cmap=mode_cmap,
                )
                # ax.set_aspect(4)
                # Align the colorbar with the plotted image.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                fig.colorbar(im, cax=cax)
            ax.set_xlabel('X-axis', fontsize=label_fontsize)
            ax.set_ylabel('Y-axis', fontsize=label_fontsize)
            plt.tight_layout()

        #  individual plots are created
        individual_fig, individual_ax = plt.subplots(figsize=(16, 3))  # Adjust figure size for a tighter plot
        plt.subplots_adjust(top=0.9, bottom=0.15, left=0.05, right=0.95)  # Adjust margins
        if len(snapshots_shape) == 1:
            individual_ax.plot(x, mode.real, c=mode_color)
        else:
            vmin = -vmax*scale2Physical
            vmax = vmax*scale2Physical
            im = individual_ax.pcolormesh(
                xgrid,
                ygrid,
                mode_reshaped.real*scale2Physical,
                vmin=vmin,
                vmax=vmax,
                cmap=mode_cmap,
            )
            individual_ax.set_aspect('auto')  # Change aspect ratio to 'auto'
            # Create a smaller colorbar with larger text
            divider = make_axes_locatable(individual_ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)  # Increase pad slightly
            cbar = individual_fig.colorbar(im, cax=cax, ticks=[vmin, vmax])
            cbar.ax.tick_params(labelsize=16)
            # cbar.set_label('Amplitude', fontsize=14, labelpad=10)

        individual_ax.set_title(f"Mode {idx + 1}, Frequency: {freq_kHz:.2f} kHz, Amplitude = +- {vmax}", fontsize=18, pad=10)
        individual_ax.set_xlabel('X (mm)', fontsize=16, labelpad=5)
        individual_ax.set_ylabel('Y (mm)', fontsize=16, labelpad=5)
        
        # Adjust layout to make it tighter
        plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        
        # Save individual plot in the same directory as summary plots
        if filename:
            base_dir = os.path.dirname(filename)
            individual_filename = os.path.join(base_dir, f"Mode_{idx + 1}_Freq_{freq_kHz:.2f}kHz.png")
        else:
            individual_filename = f"Mode_{idx + 1}_Freq_{freq_kHz:.2f}kHz.png"

        individual_fig.savefig(individual_filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(individual_fig)
        print(f"Saved individual mode plot: {individual_filename}")
        # Hide any unused subplots
        for i in range(num_modes_in_plot, modes_per_plot):
            axes[0, i].axis('off')

        # Plot dynamics in the second row
        for i, idx in enumerate(modes_to_plot):
            ax = axes[1, i]
            dynamics_data = lead_dynamics[idx].real * np.max(mode_reshaped.real)

            # Format the x-axis labels based on time magnitude
            if time_in_seconds[-1] < 1e-6:
                time_label = 'Time (ns)'
                time_data = time_in_seconds * 1e9
            elif time_in_seconds[-1] < 1e-3:
                time_label = 'Time (μs)'
                time_data = time_in_seconds * 1e6
            elif time_in_seconds[-1] < 1:
                time_label = 'Time (ms)'
                time_data = time_in_seconds * 1e3
            else:
                time_label = 'Time (s)'
                time_data = time_in_seconds

            ax.set_title("Mode Dynamics", fontsize=title_fontsize)
            ax.plot(time_data, dynamics_data, c=dynamics_color)
            ax.set_xlabel(time_label, fontsize=label_fontsize)
            ax.set_ylabel('Amplitude', fontsize=label_fontsize)

            # Re-adjust ylim if dynamics oscillations are extremely small.
            dynamics_range = dynamics_data.max() - dynamics_data.min()
            if dynamics_range != 0 and dynamics_range / np.abs(np.average(dynamics_data)) < 1e-4:
                ax.set_ylim(np.sort([0.0, 2 * np.average(dynamics_data)]))
        # Hide any unused subplots
        for i in range(num_modes_in_plot, modes_per_plot):
            axes[1, i].axis('off')

        # Adjust layout
        if tight_layout_kwargs is None:
            tight_layout_kwargs = {}
        plt.tight_layout(**tight_layout_kwargs)

        # Save plot if filename is provided
        if filename:
            # Modify filename to include plot index
            base, ext = os.path.splitext(filename)
            fname = f"{base}_{plot_idx}{ext}"
            plt.savefig(fname)
            plt.close(fig)
            print(f"Saved figure {fname}")
        else:
            plt.show()

def update(frame_num, plot_name, dmd_states, x1grid, x2grid, vDMD_min, vDMD_max, time_per_frame=1e-7):
    plt.clf()
    idx = frame_num % len(dmd_states)
    
    # Calculate time in seconds
    time_in_seconds = frame_num * time_per_frame
    
    # Format time label
    if time_in_seconds < 1e-6:
        time_label = f'{time_in_seconds * 1e9:.2f} ns'
    elif time_in_seconds < 1e-3:
        time_label = f'{time_in_seconds * 1e6:.2f} μs'
    elif time_in_seconds < 1:
        time_label = f'{time_in_seconds * 1e3:.2f} ms'
    else:
        time_label = f'{time_in_seconds:.2f} s'
    
    # Create the plot
    plt.pcolormesh(x1grid, x2grid, dmd_states[idx].real, vmin=vDMD_min, vmax=vDMD_max, shading='auto')
    plt.title(f'{plot_name} - DMD Reconstruction - Time: {time_label}')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.tight_layout()
    return plt.gcf()

def update_original(frame_num, plot_name, snapshots, x1grid, x2grid, vOG_min, vOG_max, time_per_frame=1e-7):
    plt.clf()
    idx = frame_num % len(snapshots)
    
    # Calculate time in seconds
    time_in_seconds = frame_num * time_per_frame
    
    # Format time label
    if time_in_seconds < 1e-6:
        time_label = f'{time_in_seconds * 1e9:.2f} ns'
    elif time_in_seconds < 1e-3:
        time_label = f'{time_in_seconds * 1e6:.2f} μs'
    elif time_in_seconds < 1:
        time_label = f'{time_in_seconds * 1e3:.2f} ms'
    else:
        time_label = f'{time_in_seconds:.2f} s'
    
    # Create the plot
    plt.pcolormesh(x1grid, x2grid, snapshots[idx].real, vmin=vOG_min, vmax=vOG_max, shading='auto')
    plt.title(f'{plot_name} - Original Data - Time: {time_label}')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.tight_layout()
    return plt.gcf()

def update_comparison(frame_num, plot_name, im1, im2, axs, snapshots, dmd_states, time_per_frame, nModes):
    idx = frame_num % len(snapshots)
    
    # Calculate time in seconds
    time_in_seconds = idx * time_per_frame
    
    # Format time label
    if time_in_seconds < 1e-6:
        time_label = f'{time_in_seconds * 1e9:.2f} ns'
    elif time_in_seconds < 1e-3:
        time_label = f'{time_in_seconds * 1e6:.2f} μs'
    elif time_in_seconds < 1:
        time_label = f'{time_in_seconds * 1e3:.2f} ms'
    else:
        time_label = f'{time_in_seconds:.2f} s'
    
    # Update Original Data
    im1.set_data(snapshots[idx].real)
    axs[0].set_title(f'{plot_name} - Original Data - Time: {time_label}')
    
    # Update DMD Reconstruction
    im2.set_data(dmd_states[idx].real)
    axs[1].set_title(f'{plot_name} - DMD Reconstruction with {nModes} MODES - Time: {time_label}')
    
    return im1, im2

def create_animation(data, x1grid, x2grid, video_output_dir, plot_name, update_func, 
                     time_per_frame=1e-7, duration=20, figsize=(20, 2), display_video=True, 
                     save_video=True, writer=None):
    vmin = np.array(data).real.min()
    vmax = np.array(data).real.max()

    # Create the figure
    fig = plt.figure(figsize=figsize)

    # Set up the animation parameters
    total_frames = len(data)
    fps = total_frames / duration

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        update_func,
        frames=total_frames,
        interval=1000 / fps,
        fargs=(plot_name, data, x1grid, x2grid, vmin, vmax, time_per_frame),
        blit=False
    )

    # Display the video in the notebook if requested
    if display_video:
        video_html = anim.to_jshtml()
        display(HTML(f"<h3>{plot_name}</h3>"))
        display(HTML(video_html))

    # Save the animation if requested
    if save_video:
        if writer is None:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=10000)
        
        video_path = os.path.join(video_output_dir, f'{plot_name.lower().replace(" ", "_")}.mp4')
        anim.save(video_path, writer=writer)
        print(f"{plot_name} video saved to {video_path}")

    plt.close(fig)
    return anim


def save_animation(anim_dmd, dmd_video_path, writer, fig_dmd):
    try:
        anim_dmd.save(dmd_video_path, writer=writer)
        print(f"DMD reconstruction video saved to {dmd_video_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
    finally:
        plt.close(fig_dmd)



def create_comparison_animation(snapshots, dmd_states, x1grid, x2grid, video_output_dir, plot_name, nModes,
                                time_per_frame=1e-7, duration=20, figsize=(20, 5), 
                                display_video=True, save_video=True, writer=None):
    vmin = min(np.array(snapshots).real.min(), np.array(dmd_states).real.min())
    vmax = max(np.array(snapshots).real.max(), np.array(dmd_states).real.max())
    extent = (x1grid.min(), x1grid.max(), x2grid.min(), x2grid.max())

    fig, axs = plt.subplots(2, 1, figsize=figsize)
    im1 = axs[0].imshow(snapshots[0].real, vmin=vmin, vmax=vmax, extent=extent, aspect='auto', origin='lower')
    im2 = axs[1].imshow(dmd_states[0].real, vmin=vmin, vmax=vmax, extent=extent, aspect='auto', origin='lower')

    for ax in axs:
        ax.set_xlabel('X (mm)')
    axs[0].set_ylabel('Y (mm)')
    axs[1].set_ylabel('Y (mm)')
    
    cbar1 = plt.colorbar(im1, ax=axs[0], label='Value')
    cbar2 = plt.colorbar(im2, ax=axs[1], label='Value')
    plt.tight_layout()

    def update(frame_num):
        idx = frame_num % len(snapshots)
        time_in_seconds = idx * time_per_frame
        
        if time_in_seconds < 1e-6:
            time_label = f'{time_in_seconds * 1e9:.2f} ns'
        elif time_in_seconds < 1e-3:
            time_label = f'{time_in_seconds * 1e6:.2f} μs'
        elif time_in_seconds < 1:
            time_label = f'{time_in_seconds * 1e3:.2f} ms'
        else:
            time_label = f'{time_in_seconds:.2f} s'
        
        im1.set_array(snapshots[idx].real)
        im2.set_array(dmd_states[idx].real)
        axs[0].set_title(f'{plot_name} - Original Data - Time: {time_label}')
        axs[1].set_title(f'{plot_name} - DMD Reconstruction ({nModes} modes) - Time: {time_label}')
        return im1, im2

    total_frames = min(len(snapshots), len(dmd_states))
    fps = total_frames / duration

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)

    if display_video:
        video_html = anim.to_jshtml()
        display(HTML(f"<h3>{plot_name} Comparison</h3>"))
        display(HTML(video_html))

    if save_video:
        if writer is None:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=10000)
        
        video_path = os.path.join(video_output_dir, f'{plot_name.lower().replace(" ", "_")}_comparison.mp4')
        anim.save(video_path, writer=writer)
        print(f"{plot_name} comparison video saved to {video_path}")

    plt.close(fig)
    return anim


def square_wave(x1grid, x2grid, t, frequency, amplitude, size_x=4, size_y=1, decay_factor=1.0):
    square = np.zeros_like(x1grid, dtype=complex)
    square[(x1grid >= -size_x/2) & (x1grid <= size_x/2) & 
           (x2grid >= -size_y/2) & (x2grid <= size_y/2)] = 1
    return amplitude * square * (decay_factor ** -t) * np.exp(1j * frequency * t)

def saveSnapshots(snapshots,x1grid,x2grid,output_dir):
    x_flat = x1grid.flatten()
    y_flat = x2grid.flatten()
    for i, snapshot in enumerate(snapshots):
        # Flatten the snapshot data
        data_flat = snapshot.real.flatten()
        # Create a DataFrame
        df = pd.DataFrame({
            'x_loc': x_flat,
            'y_loc': y_flat,
            'value': data_flat
        })
        filename = os.path.join(output_dir, f'snapshot{i:03d}.csv')
        df.to_csv(filename, index=False)
        print(f"Saved snapshot {i} to {filename}")

def reconstruct_with_modes(dmd, num_modes):
    # Sort modes by amplitude
    idx = np.argsort(np.abs(dmd.amplitudes))[::-1]
    
    # Select the top num_modes
    modes = dmd.modes[:, idx[:num_modes]]
    eigs = dmd.eigs[idx[:num_modes]]
    amplitudes = dmd.amplitudes[idx[:num_modes]]
    
    # Reconstruct
    time_grid = np.outer(dmd.dmd_timesteps, np.ones(num_modes))
    temp_matrix = np.exp(np.outer(dmd.dmd_timesteps, eigs))
    reconstructed = modes.dot(np.diag(amplitudes)).dot(temp_matrix.T)
    
    return reconstructed.T

def update_2_modes(frame_num):
    plt.clf()
    idx = frame_num % len(dmd_states_2_modes)
    plt.pcolormesh(x1grid, x2grid, dmd_states_2_modes[idx].real, vmin=-1, vmax=1, shading='auto')
    plt.title(f'DMD Reconstruction (2 modes) - Frame {idx}')
    plt.colorbar()
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    return plt.gcf()

def ReadSnapshot(snapshot_file, full_column_length):
    df = pd.read_csv(snapshot_file)
    # Extract probe_index and Nparticles
    probe_index = df['probe_index'].values
    nparticles = df['nparticles'].values
    temps = df['temperature'].values
    # Calculate the dimensions of the full field
    num_probes = len(probe_index)
    max_probe_index = np.max(probe_index)
    full_row_length = (max_probe_index // full_column_length) + 1
    
    # Create a 2D array to hold the full field data
    full_field = np.full((full_column_length, full_row_length), np.nan)
    
    # Calculate 2D indices from probe_index (column-major order)
    col_indices = probe_index % full_column_length
    row_indices = probe_index // full_column_length
    
    # Populate the 2D array
    full_field[col_indices, row_indices] = nparticles
    # full_field[col_indices, row_indices] = temps

    # Create a mask for the valid data points
    valid_mask = ~np.isnan(full_field)
    
    # Find the bounds of the valid data
    valid_rows, valid_cols = np.where(valid_mask)
    min_row, max_row = valid_rows.min(), valid_rows.max()
    min_col, max_col = valid_cols.min(), valid_cols.max()
    
    # Extract the window with valid data
    window_data = full_field[min_row:max_row+1, min_col:max_col+1]
    return window_data

def read_snapshots(snapshot_folder, full_column_length):
    # Get list of all CSV files in the snapshot folder
    snapshot_files = sorted([f for f in os.listdir(snapshot_folder) if f.endswith('.csv')])
    n_snapshots = len(snapshot_files)
    first_snapshot = ReadSnapshot(os.path.join(snapshot_folder, snapshot_files[0]), full_column_length)
    ysize, xsize = first_snapshot.shape
    snapshots = np.zeros((n_snapshots, ysize, xsize))

    for i, file in enumerate(snapshot_files):
        print(i)
        snapshot = ReadSnapshot(os.path.join(snapshot_folder, file), full_column_length)
        snapshots[i] = snapshot
    snapshots = np.nan_to_num(snapshots, nan=0)  # Replace NaN with 0
    return snapshots, n_snapshots