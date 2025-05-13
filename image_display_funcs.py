import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import torch
import torch.nn.functional as F

def tensor_imshow(tensor, ax=None, **kwargs):
    """Display a tensor as an image, automatically handling channel ordering and normalization.
    
    Args:
        tensor: Input tensor (C,H,W) or (H,W) format
        ax: Optional matplotlib axis
        **kwargs: Additional arguments to pass to imshow
    """
    if ax is None:
        ax = plt.gca()
    
    # Move tensor to CPU and convert to numpy
    img = tensor.cpu().detach().float()
    
    # Handle different input formats
    if img.dim() == 4:  # Batch dimension [B,C,H,W]
        img = img[0]  # Take first image
    
    if img.dim() == 3:  # Channel dimension [C,H,W]
        if img.shape[0] == 1:  # Grayscale
            img = img.squeeze(0)  # [H,W]
            cmap = 'gray'
        else:  # Color (RGB/RGBA)
            img = img.permute(1, 2, 0)  # [H,W,C]
            cmap = None
    else:  # Already 2D [H,W]
        cmap = 'gray'
    
    # Normalize to [0,1] range
    if img.min() < 0 or img.max() > 1:  # normalized to [-1,1]
        img = (img - img.min()) / (img.max() - img.min())
    else:  # Handle constant images
        img = torch.clamp(img, 0, 1)
    
    # Display image
    ax.imshow(img.numpy(), cmap=cmap, vmin=0, vmax=1, **kwargs)
    ax.axis('off')
    return ax

    ## Handle normalization
    #img = img.float()
    #if img.min() < 0 or img.max() > 1:  # Likely normalized to [-1,1]
    #    img = (img + 1) / 2  # Scale to [0,1]

def plot_labelled_images(images, n_samples=4, title=None, labels=None):
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4))
    
    for i in range(n_samples):
        tensor_imshow(images[i], ax=axes[i])
        if labels is not None:
            axes[i].set_title(f"{title}\nLabel: {labels[i].item()}")
    
    plt.tight_layout()
    plt.show()

def plot_image(images, n_samples=4):
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4))
    
    for i in range(n_samples):
        tensor_imshow(images[i], ax=axes[i])
    
    plt.tight_layout()
    plt.show()

def plot_save_image(image, figsize, filename, title=None):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    tensor_imshow(image)
    axes.set_title(f"{title}")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

def visualize_image_trajectory(trajectory, num_time_points=5, filename=None):
    """Visualizes images at different time points in the trajectory."""
    time_indices = np.linspace(0, len(trajectory)-1, num_time_points, dtype=int)
    fig, axes = plt.subplots(len(time_indices), trajectory.shape[1], 
                         figsize=(15, 2*len(time_indices)))
    
    for row, t_idx in enumerate(time_indices):
        for col in range(trajectory.shape[1]):
            tensor_imshow(trajectory[t_idx, col], ax=axes[row, col])
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

def image_trajectory_animation(trajectory):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        ax.clear()
        tensor_imshow(trajectory[frame, 0], ax=ax)
        ax.set_title(f"t={frame/(len(trajectory)-1):.2f}", fontsize=20)
    
    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=50)
    plt.close()
    return anim

def find_closest_matches(generated_images, target_dataset, device='cpu'):
    """Finds closest matches in target_dataset for each generated image using MSE."""
    generated_images = generated_images.to(device)
    min_mse_values = []
    closest_indices = []
    
    for gen_img in generated_images:
        mse_values = torch.stack([
            F.mse_loss(gen_img, target_img.to(device), reduction='mean')
            for target_img in target_dataset
        ])
        min_mse, min_idx = torch.min(mse_values, dim=0)
        min_mse_values.append(min_mse.item())
        closest_indices.append(min_idx.item())
    
    return torch.tensor(min_mse_values), closest_indices

def display_comparisons(generated_images, target_dataset, 
                      closest_indices,
                      num_display=5, figsize=(12, 6),
                      filename=None):
    """Displays side-by-side comparisons of generated images and their closest matches."""
    num_display = min(num_display, len(generated_images))
    fig, axs = plt.subplots(num_display, 2, figsize=figsize)
    
    if num_display == 1:
        axs = axs.reshape(1, 2)
    
    for i in range(num_display):
        tensor_imshow(generated_images[i], ax=axs[i, 0])
        axs[i, 0].set_title('Generated Image')
        
        closest_img = target_dataset[closest_indices[i]]
        if isinstance(closest_img, tuple):
            closest_img = closest_img[0]
        tensor_imshow(closest_img, ax=axs[i, 1])
        axs[i, 1].set_title('Closest Match')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
