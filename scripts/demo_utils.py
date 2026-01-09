#usr/bin/env python3
"""
Utilities to produce or load demonstration trajectories and initialize policy params.
The paper uses least-squares regression to get initial Kp_i and X_i; here we provide
a simple, robust initializer:
 - Partition the demo trajectory into K segments and set X_i to the segment's mean.
 - Set Kp_i to diag(kp_diag) initially (user can later allow RL to learn off-diagonals).
This is intentionally simple so you can replace with a proper regression later.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def make_demo(duration=1.0, timesteps=200):
    t = np.linspace(0.0, duration, timesteps)
    tau = t / duration

    start = np.array([0.0, 0.0, 0.0])
    goal  = np.array([0.25, 0.1, 0.15])

    # Minimum-jerk time scaling: 10τ³ - 15τ⁴ + 6τ⁵
    s = 10*tau**3 - 15*tau**4 + 6*tau**5

    x = np.zeros((timesteps, 3))
    x[:, 0] = start[0] + s * (goal[0] - start[0])
    x[:, 1] = start[1] + s * (goal[1] - start[1])
    x[:, 2] = 0.15 + 0.15 * np.sin(np.pi * s)  # smooth vertical curve

    # analytic derivative of s (for optional correctness)
    ds_dt = (30*tau**2 - 60*tau**3 + 30*tau**4) / duration
    xdot = np.zeros_like(x)
    xdot[:, 0] = ds_dt * (goal[0] - start[0])
    xdot[:, 1] = ds_dt * (goal[1] - start[1])
    xdot[:, 2] = 0.15 * np.pi * np.cos(np.pi * s) * ds_dt

    xdot[0, :] = 0.0
    xdot[-1, :] = 0.0

    return {'t': t, 'x': x, 'xdot': xdot}

def make_demo_6D(duration=1.0, timesteps=200, curvature=0.0):
    """
    Create a 6D demonstration trajectory:
      - First 3 dims: position (same as the original 3D demo)
      - Last 3 dims: orientation (RPY), rotating 90 deg (π/2) about X-axis
    Returns dict with keys: 't', 'x', 'xdot'
    """
    t = np.linspace(0.0, duration, timesteps)
    tau = t / duration

    start = np.zeros(6)
    goal = np.array([0.5, 0.2, 0.3, 0.2, -0.1, 0.1])  # last three are orientation deltas

    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    x = np.zeros((timesteps, 6))
    for d in range(6):
        x[:, d] = start[d] + s * (goal[d] - start[d])

    # Add some curvature for the position components
    x[:, 2] = curvature * np.sin(np.pi * s)  # z oscillation

    # ---- Orientation: rotation about X over time ----
    start_rpy = np.zeros(3)
    goal_rpy  = np.array([np.pi/6, 0.0, 0.0])

    for d in range(3):
        x[:, 3+d] = start_rpy[d] + s * (goal_rpy[d] - start_rpy[d])

    # Derivatives
    xdot = np.gradient(x, axis=0) / (t[1] - t[0])

    return {'t': t, 'x': x, 'xdot': xdot}

def init_from_demo(dmp, demo, kp_diag=50.0):
    """
    Populate dmp.X and dmp.Kp from demo.
    - Set X_i to mean of equally partitioned segments of the demo x(t)
    - Set Kp_i to kp_diag * I
    """
    T = demo['x'].shape[0]
    seg_len = T // dmp.K
    for i in range(dmp.K):
        s = i * seg_len
        e = (i+1) * seg_len if i < dmp.K-1 else T
        seg_mean = np.mean(demo['x'][s:e], axis=0)
        dmp.X[i] = seg_mean
        # initialize with diagonal stiffness
        dmp.Kp[i] = kp_diag * np.eye(dmp.D)
    return dmp

def init_from_demo_straightline(dmp, demo, kp_diag=50.0):
    """
    Populate dmp.X and dmp.Kp from demo, assuming a straight-line demo.
    - Set all X_i to the demo's goal position
    - Set Kp_i to kp_diag * I
    """
    goal = demo['x'][-1]
    for i in range(dmp.K):
        dmp.X[i] = goal
        dmp.Kp[i] = kp_diag * np.eye(dmp.D)
    return dmp

def set_axes_equal(ax):
    """
    Make 3D plot axes equal.
    This ensures that spheres look like spheres and cubes look like cubes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_environment(ax, demo, obstacles, goal):
    ax.plot(demo['x'][:,0], demo['x'][:,1], demo['x'][:,2],
            'r-', linewidth=2, label='Demo')
    ax.scatter(goal[0], goal[1], goal[2], color='green', s=100, label='Goal')

    # draw all obstacles
    for _, ob in enumerate(obstacles):
        c, r = ob['center'], ob['radius']
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        x = c[0] + r * np.cos(u) * np.sin(v)
        y = c[1] + r * np.sin(u) * np.sin(v)
        z = c[2] + r * np.cos(v)
        ax.plot_surface(x, y, z, color='red', alpha=0.25, linewidth=0)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=10., azim=-20)
    set_axes_equal(ax)

def plot_environment_no_demo(ax, obstacles, goal):
    ax.scatter(goal[0], goal[1], goal[2], color='green', s=100, label='Goal')

    # draw all obstacles
    for _, ob in enumerate(obstacles):
        c, r = ob['center'], ob['radius']
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        x = c[0] + r * np.cos(u) * np.sin(v)
        y = c[1] + r * np.sin(u) * np.sin(v)
        z = c[2] + r * np.cos(v)
        ax.plot_surface(x, y, z, color='red', alpha=0.25, linewidth=0)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=10., azim=-20)
    set_axes_equal(ax)
        
def plot_orientations(ax, traj, step=20, scale=0.05, alpha=0.7):
    """
    Plot 3D position trajectory (first 3 dims of 6D x) and draw small orientation frames.
    The last 3 dims of x are interpreted as roll-pitch-yaw angles (in radians).
    """
    x = traj['x']
    pos = x[:, :3]

    # Draw local frames every `step` samples
    for i in range(0, len(pos), step):
        rpy = x[i, 3:6]
        rot = R.from_euler('xyz', rpy).as_matrix()
        origin = pos[i]
        # x,y,z axes of the local frame
        ax.quiver(origin[0], origin[1], origin[2],
                  rot[0,0]*scale, rot[1,0]*scale, rot[2,0]*scale,
                  color='r', alpha=alpha)
        ax.quiver(origin[0], origin[1], origin[2],
                  rot[0,1]*scale, rot[1,1]*scale, rot[2,1]*scale,
                  color='g', alpha=alpha)
        ax.quiver(origin[0], origin[1], origin[2],
                  rot[0,2]*scale, rot[1,2]*scale, rot[2,2]*scale,
                  color='b', alpha=alpha)

def plot_quaternions(ax, traj, step=20, scale=0.05):
    x = traj['x']
    pos = x[:, :3]
    quats = x[:, 3:7]

    for i in range(0, len(pos), step):
        rot = R.from_quat(quats[i]).as_matrix()
        origin = pos[i]
        ax.quiver(origin[0], origin[1], origin[2],
                  rot[0,0]*scale, rot[1,0]*scale, rot[2,0]*scale, color='r', alpha=0.7)
        ax.quiver(origin[0], origin[1], origin[2],
                  rot[0,1]*scale, rot[1,1]*scale, rot[2,1]*scale, color='g', alpha=0.7)
        ax.quiver(origin[0], origin[1], origin[2],
                  rot[0,2]*scale, rot[1,2]*scale, rot[2,2]*scale, color='b', alpha=0.7)
        
def make_exploration_std(D, K, sigma_pos=0.03, sigma_ori=0.01, sigma_kp=0.005):
    """
    Build a per-parameter exploration standard deviation vector for a MixturePD DMP.

    Args:
        dmp: MixturePD instance
        sigma_pos: std dev for positional components of attractors X_i
        sigma_ori: std dev for orientation components (last 3 or 4 dims)
        sigma_kp: std dev for Kp matrix elements
    Returns:
        np.ndarray of shape (dmp.n_params(),)
    """
    expl_std = []

    for _ in range(K):
        # Kp_i block (D*D entries)
        expl_std.extend([sigma_kp] * (D * D))

        # X_i block (D entries)
        if D == 3:
            expl_std.extend([sigma_pos] * 3)
        if D == 6:
            # roll-pitch-yaw (3 pos + 3 ori)
            expl_std.extend([sigma_pos] * 3 + [sigma_ori] * 3)
        elif D == 7:
            # quaternion (3 pos + 4 quat)
            expl_std.extend([sigma_pos] * 3 + [sigma_ori] * 4)
        else:
            # generic fallback
            expl_std.extend([sigma_pos] * D)

    return np.array(expl_std)