import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def visualize_sine_trajectory(trajectory, num_time_points=5, filename=None):
    """Visualizes sine waves at different time points in the trajectory.
    
    Args:
        trajectory: 3D tensor of shape (time_steps, num_waves, signal_length)
        num_time_points: Number of time points to display
        filename: Optional filename to save the plot
    """
    # Convert tensor to numpy if needed
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()
    
    time_indices = np.linspace(0, len(trajectory)-1, num_time_points, dtype=int)
    total_time = time_indices[-1]
    num_waves = trajectory.shape[1]
    
    fig, axes = plt.subplots(len(time_indices), num_waves, 
                          figsize=(3*num_waves, 2*len(time_indices)))
    
    # Handle case when there's only one wave
    if num_waves == 1:
        axes = axes[:, np.newaxis]
    
    for row, t_idx in enumerate(time_indices):
        for col in range(num_waves):
            ax = axes[row, col]
            ax.plot(trajectory[t_idx, col])
            ax.set_title(f'Wave {col+1}, Time {t_idx/total_time}')
            ax.grid(True)
            ax.set_ylim(trajectory.min(), trajectory.max())
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

def sine_trajectory_animation(trajectory, interval=50):
    """Creates an animation of sine wave evolution over time.
    
    Args:
        trajectory: 3D tensor of shape (time_steps, num_waves, signal_length)
        interval: Delay between frames in milliseconds
    
    Returns:
        matplotlib.animation.FuncAnimation object
    """
    # Convert tensor to numpy if needed
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()
    
    num_waves = trajectory.shape[1]
    fig, axes = plt.subplots(1, num_waves, figsize=(4*num_waves, 4))
    
    # Handle case when there's only one wave
    if num_waves == 1:
        axes = [axes]
    
    # Initialize lines and set up each subplot
    lines = []
    for col in range(num_waves):
        line, = axes[col].plot([], [])
        lines.append(line)
        axes[col].set_xlim(0, trajectory.shape[2])
        axes[col].set_ylim(trajectory.min(), trajectory.max())
        axes[col].grid(True)
        axes[col].set_title(f"Wave {col+1}")
    
    def update(frame):
        for col in range(num_waves):
            lines[col].set_data(np.arange(trajectory.shape[2]), 
                              trajectory[frame, col])
            axes[col].set_xlabel(f"t={frame/(len(trajectory)-1):.2f}")
        return lines
    
    anim = FuncAnimation(fig, update, frames=len(trajectory), 
                       interval=interval, blit=True)
    plt.close()
    return anim