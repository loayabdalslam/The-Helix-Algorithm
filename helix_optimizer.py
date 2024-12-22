import numpy as np
import torch

class HelixSGOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, radius=1, depth_target=0, steps=100):
        defaults = dict(lr=lr, radius=radius, depth_target=depth_target, steps=steps)
        super(HelixSGOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            radius = group['radius']
            depth_target = group['depth_target']
            steps = group['steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('HelixSGOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['x0'] = p.data.clone()
                    state['y0'] = p.data.clone()
                    state['z0'] = p.data.clone()

                # Current state
                step = state['step']
                x0, y0, z0 = state['x0'], state['y0'], state['z0']

                # Compute the next point directly
                angle = step * np.pi / (step + 1)
                radius_step = radius * (1 - step / steps)
                depth_step = z0 - (step / steps) * (z0 - depth_target)

                x_next = x0 + radius_step * torch.cos(torch.tensor(angle, device=p.device, dtype=p.dtype))
                y_next = y0 + radius_step * torch.sin(torch.tensor(angle, device=p.device, dtype=p.dtype))
                z_next = depth_step

                # Update the parameter
                p.data.copy_(z_next)

                # Update state
                state['step'] += 1
                state['x0'] = x_next
                state['y0'] = y_next
                state['z0'] = z_next

        return loss
  

class HelixMotion:
    """
    Class to generate helix-like motion trajectories, inspired by the optimizer described in the paper.
    """

    def __init__(self, x0=0, y0=0, z0=0, steps=100, radius=1, depth_target=0):
        """
        Initialize the HelixMotion object with parameters for the trajectory.

        Args:
            x0 (float): Initial x-coordinate.
            y0 (float): Initial y-coordinate.
            z0 (float): Initial z-coordinate.
            steps (int): Number of steps in the trajectory.
            radius (float): Initial radius of the helix.
            depth_target (float): Target depth for the helix motion.
        """
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.steps = steps
        self.radius = radius
        self.depth_target = depth_target

    def standard_helix(self):
        """
        Generate a standard helix motion trajectory.

        Returns:
            np.ndarray: Array of (x, y, z) coordinates along the trajectory.
        """
        x, y, z = self.x0, self.y0, self.z0
        trajectory = []

        for step in range(self.steps):
            angle = step * np.pi / 10  # Fixed increment for angle
            radius_step = self.radius * (1 - step / self.steps)  # Shrink radius
            depth_step = self.z0 - (step / self.steps) * (self.z0 - self.depth_target)  # Adjust depth

            # Update coordinates
            x = self.x0 + radius_step * np.cos(angle)
            y = self.y0 + radius_step * np.sin(angle)
            z = depth_step

            # Store point
            trajectory.append((x, y, z))

        return np.array(trajectory)

    def modified_helix(self):
        """
        Generate a modified helix motion trajectory with dynamic angle increments.

        Returns:
            np.ndarray: Array of (x, y, z) coordinates along the trajectory.
        """
        x, y, z = self.x0, self.y0, self.z0
        trajectory = []

        for step in range(self.steps):
            angle = step * np.pi / (step + 1)  # Dynamic increment for angle
            radius_step = self.radius * (1 - step / self.steps)  # Shrink radius
            depth_step = self.z0 - (step / self.steps) * (self.z0 - self.depth_target)  # Adjust depth

            # Update coordinates
            x = self.x0 + radius_step * np.cos(angle)
            y = self.y0 + radius_step * np.sin(angle)
            z = depth_step

            # Store point
            trajectory.append((x, y, z))

        return np.array(trajectory)

# Helix optimizer function: Quick usage
def helix_optimizer(step, radius=1, depth_target=0):
    # Simulate a helix-like trajectory (simplified)
    t = step / 1000.0  # Normalize the step
    angle = 2 * np.pi * t  # Helix angle
    z = depth_target * np.sin(angle)  # Z coordinate as a sine function for depth
    x = radius * np.cos(angle)  # X coordinate
    y = radius * np.sin(angle)  # Y coordinate
    return np.array([x, y, z])  # Return the 3D helix trajectory