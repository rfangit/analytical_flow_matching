import torch

def compute_linear_velocity_batch_time(
    current_points: torch.Tensor,  # Shape [M, *dims]
    data: torch.Tensor,            # Shape [N, *dims]
    t: torch.Tensor,               # Shape [M] (batch of time values)
    sigma_i: float,
    coefficients=None              # Shape [N] (weights of the data points)
) -> torch.Tensor:
    """
    Computes velocity for batched inputs with time as tensor.    
    Returns: velocities: [M, *dims] computed velocities
    """
    # Reshaping time values for broadcasting
    # We need to multiply target data by time [M, 1, *dims]
    # We need to multiply the differences between target and data [M, N]
    # We need to multiply the final vector which requires [M, *dims]
    t_reshaped = t.view(-1, *([1]*(data.dim())))  # [M, 1, *dims]
    t_reshaped_2 = t.view(-1, *([1]*(data.dim() - 1)))  # [M, *dims]
    t_reshaped_3 = t[:, None]  # [M, 1]
    
    beta = t_reshaped       # [M, 1, *dims]
    alpha = 1 - t_reshaped  # [M, 1, *dims]
    alpha2 = 1 - t_reshaped_2 # [M, *dims]
    alpha3 = 1 - t_reshaped_3 # [M, 1]

    data_exp = data.unsqueeze(0)                    # Reshape data to [1, N, *dims]
    data_scaled = beta * data_exp               # [M, 1, *dims] * [1, N, *dims] => [M, N, *dims] 
    current_expanded = current_points.unsqueeze(1)  # [M, 1, *dims]
    
    # Compute distances
    diff = current_expanded - data_scaled           # [M, N, *dims]
    squared_dist = torch.sum(diff**2, dim=tuple(range(2, diff.dim())))  # [M, N]

    # Compute softmax weights [M, N]
    logits = -0.5 * squared_dist/(alpha3**2 * sigma_i)
    if coefficients is not None:
        logits = logits + torch.log(coefficients).unsqueeze(0)
    weights = torch.softmax(logits, dim=1)  # [M, N]
    
    # Compute weighted sum [M, *dims]
    weighted_sum = torch.einsum('mn,n...->m...', weights, data)  # [M, *dims]
    
    # Compute velocities [M, *dims]
    velocities = (weighted_sum - current_points) / alpha2
    return velocities

def compute_linear_velocity_batch(
    current_points: torch.Tensor,  # Shape [M, *dims]
    data: torch.Tensor,            # Shape [N, *dims]
    t: float,                      
    sigma_i: float,
    coefficients=None              # Shape [N] (weights of the data points)
) -> torch.Tensor:
    """
    Computes velocity for batched inputs with time as a fixed scalar for all entries.    
    Returns: velocities: [M, *dims] computed velocities
    """
    M = current_points.shape[0]
    tensor_t = torch.full((M,), t, dtype=torch.float32, device=current_points.device)
    velocities = compute_linear_velocity_batch_time(current_points = current_points,  # Shape [M, *dims]
                                                    data = data,            # Shape [N, *dims]
                                                    t = tensor_t,               # Shape [M] (batch of time values)
                                                    sigma_i = sigma_i,
                                                    coefficients=coefficients              # Shape [N] (weights of the data points)
                                                    )
    return velocities

def forward_euler_integration_analytic_linear(
    initial_points: torch.Tensor,  # Initial positions [m, d]
    data: torch.Tensor,            # Data points [n, d]
    t_start: float,                # Start time
    t_end: float,                  # End time
    num_steps: int,                # Number of time steps
    sigma_i: float,                # Initial variance
    coefficients=None              # Optional coefficients [n]
) -> torch.Tensor:
    """
    Forward Euler integration of the velocity field via linear scheduling
    
    Args:
        initial_points: Starting positions [m, d]
        data: Fixed data points [n, d]
        t_start: Initial time (typically 0)
        t_end: Final time (typically close to 1)
        num_steps: Number of integration steps
        sigma_i: Initial variance parameter
        coefficients: Optional weights for data points
        
    Returns:
        Tensor of shape [num_steps+1, m, d] containing the trajectory at each time step
    """
    dt = (t_end - t_start) / num_steps
    trajectory = torch.zeros(num_steps + 1, *initial_points.shape, device=initial_points.device)
    trajectory[0] = initial_points.clone()
    
    current_points = initial_points.clone()
    current_time = t_start
    
    for step in range(1, num_steps + 1):
        # Compute velocity at current position and time
        velocity = compute_linear_velocity_batch(
            current_points, data, current_time, sigma_i, coefficients
        )
        # Forward Euler update
        current_points = current_points + velocity * dt
        current_time = current_time + dt
        # Store current positions
        trajectory[step] = current_points.clone()
    return trajectory
    
def compute_predictions(trajectory, num_steps):
    """
    Extrapolate flow-matching predictions for the final answer
    
    Args:
        trajectory: Array of shape [t+1, N, D] where t is time, N is batch size, D is dimension
        num_steps: Number of time steps in the original integration
        
    Returns:
        predictions: Array of shape [t+1, N, D] with linear extrapolation predictions
                     (last prediction is just the last point in trajectory)
    """
    # Calculate time step size
    t_start = 0.0
    t_end = 1.0
    dt = (t_end - t_start) / num_steps
    
    # Compute forward differences
    delta = trajectory[1:] - trajectory[:-1]  # Shape [t, N, D]
    
    # Pad delta with zeros to match original trajectory shape
    zero_pad = torch.zeros_like(trajectory[:1])  # Shape [1, N, D]
    delta_padded = torch.cat([delta, zero_pad], dim=0)  # Shape [t+1, N, D]

    # Create array of time remaining for each step [t+1, 1, 1]
    time_indices = torch.arange(len(trajectory), device=trajectory.device)  # 0 to t
    time_remaining = t_end - (time_indices * dt)  # Shape [t+1]

    # Reshape for broadcasting
    time_remaining_reshaped = time_remaining.view(-1, *(1,)*(delta.dim()-1))

    # Compute predictions (current point + delta * (time_remaining/dt))
    predictions = trajectory + delta_padded * (time_remaining_reshaped / dt)
    
    return predictions

def compute_velocity_batch(
    current_points,           # Target points (x), shape (m, n_dims)
    data,                     # Data points (μ_j), shape (n_points, n_dims)
    sigma_i,                  # Initial variance (scalar)
    sigma_f,                  # Final variance (scalar)
    alpha,                    # Alpha value (scalar)
    beta,                     # Beta value (scalar)
    alpha_prime,              # dα/dt (scalar)
    beta_prime,               # dβ/dt (scalar)
    coefficients=None         # Optional coefficients (C_j), shape (n_points,)
):
    """
    Computes the velocity field for multiple points in batch.
    
    Args:
        current_points: [m, n_dims] tensor of target points
        data: [n_points, n_dims] tensor of source points
        Returns: [m, n_dims] tensor of velocities
    """
    # Reshape tensors for broadcasting [M, N, *dims]
    current_expanded = current_points.unsqueeze(1)  # [M, 1, *dims]
    data_scaled = data * beta                      # [N, *dims]
    data_expanded = data_scaled.unsqueeze(0)       # [1, N, *dims]
    
    # Compute squared distances [M, N], summing over the other dimensions
    diff = current_expanded - data_expanded  # [M, N, *dims]
    squared_dist = torch.sum(diff**2, dim=tuple(range(2, diff.dim())))  # [M, N]
    
    # Compute softmax weights [m, n_points]
    denominator = (alpha**2 * sigma_i) + (beta**2 * sigma_f)
    logits = -0.5 * squared_dist / denominator
    if coefficients is not None:
        logits = logits + torch.log(coefficients).unsqueeze(0)
    softmax_probs = torch.softmax(logits, dim=1)  # [m, n_points]

    denominator = (alpha**2 * sigma_i) + (beta**2 * sigma_f)
    weighted_mean = torch.einsum('mn,nd->md', softmax_probs, data)  # [m, n_dims]
    
    term1 = current_points * (alpha_prime * alpha * sigma_i + beta_prime * beta * sigma_f)
    term2_coeff = alpha * sigma_i * (alpha_prime * beta - alpha * beta_prime)
    term2 = term2_coeff * weighted_mean
    
    velocity = (term1 - term2) / denominator
    return velocity

def forward_euler_integration_analytic(
    initial_points: torch.Tensor,  # Initial positions [m, d]
    data: torch.Tensor,            # Data points [n, d]
    t_start: float,                # Start time
    t_end: float,                  # End time
    num_steps: int,                # Number of time steps
    sigma_i: float,                # Initial variance
    sigma_f: float,                # Final variance
    alpha: callable,               # Function alpha(t)
    alpha_prime: callable,         # Derivative of alpha: α'(t)
    beta: callable,                # Function beta(t)
    beta_prime: callable,          # Derivative of beta: β'(t)
    coefficients=None              # Optional coefficients [n]
) -> torch.Tensor:
    """
    Forward Euler integration of the velocity field with general scheduling
    
    Args:
        initial_points: Starting positions [m, d]
        data: Fixed data points [n, d]
        t_start: Initial time (typically 0)
        t_end: Final time (typically close to 1)
        num_steps: Number of integration steps
        sigma_i: Initial variance parameter
        sigma_f: Final variance parameter
        alpha: Function α(t) for drift scheduling
        alpha_prime: Function α'(t) for drift derivative
        beta: Function β(t) for diffusion scheduling
        beta_prime: Function β'(t) for diffusion derivative
        coefficients: Optional weights for data points
        
    Returns:
        Tensor of shape [num_steps+1, m, d] containing the trajectory at each time step
    """
    dt = (t_end - t_start) / num_steps
    trajectory = torch.zeros(num_steps + 1, *initial_points.shape, device=initial_points.device)
    trajectory[0] = initial_points.clone()
    
    current_points = initial_points.clone()
    current_time = t_start
    
    for step in range(1, num_steps + 1):
        # Compute velocity at current position and time
        velocity = compute_velocity_batch(
            current_points=current_points,
            data=data,
            sigma_i=sigma_i,
            sigma_f=sigma_f,
            alpha=alpha(current_time),          # Evaluate α at current time
            beta=beta(current_time),             # Evaluate β at current time
            alpha_prime=alpha_prime(current_time), # Evaluate α' at current time
            beta_prime=beta_prime(current_time),  # Evaluate β' at current time
            coefficients=coefficients
        )
        
        # Forward Euler update
        current_points = current_points + velocity * dt
        current_time = current_time + dt
        
        # Store current positions
        trajectory[step] = current_points.clone()
    
    return trajectory