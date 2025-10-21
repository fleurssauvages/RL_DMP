import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

def resample_min_jerk(traj, N_new=None, duration=None):
    """
    Resample a spatial trajectory with a minimum-jerk time law.
    Keeps path geometry (same shape), allows arbitrary number of samples N_new.

    Args:
        traj: dict with 't' and 'x' (N,D)
        N_new: number of desired output samples (default = same as input)
        duration: total time (default = traj['t'][-1] - traj['t'][0])
    Returns:
        dict with 't', 'x', 'xdot'
    """
    t_orig = np.asarray(traj['t'])
    x_orig = np.asarray(traj['x'])
    N = len(t_orig)
    D = x_orig.shape[1]
    if N_new is None:
        N_new = N
    if duration is None:
        duration = float(t_orig[-1] - t_orig[0])

    # --- Arc-length parameter of original trajectory ---
    diffs = np.linalg.norm(np.diff(x_orig, axis=0), axis=1)
    s_orig = np.concatenate(([0.0], np.cumsum(diffs)))
    total_length = s_orig[-1] if s_orig[-1] > 0 else 1.0
    s_orig /= total_length

    # --- Define interpolation of the path as function of normalized s ---
    interp_fun = interp1d(s_orig, x_orig, axis=0, kind='cubic', fill_value="extrapolate")

    # --- Build new normalized time grid and min-jerk phase profile ---
    tau_new = np.linspace(0.0, 1.0, N_new)
    s_new = 10*tau_new**3 - 15*tau_new**4 + 6*tau_new**5   # min-jerk position profile
    t_new = tau_new * duration

    # --- Sample new positions along the path using s_new ---
    x_new = interp_fun(s_new)

    # --- Compute velocities by finite differences ---
    xdot_new = np.zeros_like(x_new)
    for i in range(1, N_new-1):
        dt = t_new[i+1] - t_new[i-1]
        xdot_new[i] = (x_new[i+1] - x_new[i-1]) / dt
    xdot_new[0] = (x_new[1] - x_new[0]) / (t_new[1] - t_new[0])
    xdot_new[-1] = (x_new[-1] - x_new[-2]) / (t_new[-1] - t_new[-2])

    return {'t': t_new, 'x': x_new, 'xdot': xdot_new}

def gaussian_velocity_retiming(traj, duration, dt, sigma=0.15):
    """
    Retimes a trajectory so velocity follows a Gaussian bell curve.
    Args:
        traj: (N, D) numpy array
        duration: total time [s]
        dt: sampling period [s]
        sigma: width of Gaussian (relative to [0,1])
    Returns:
        new_traj
    """
    N = traj.shape[0]
    
    # --- Compute cumulative distance along trajectory (for position only) ---
    dist = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(dist)])
    s /= s[-1]  # normalized path parameter s ∈ [0, 1]
    
    # --- Define Gaussian velocity profile over s ∈ [0,1] ---
    s_uniform = np.linspace(0, 1, 1000)
    v_profile = norm.pdf(s_uniform, 0.5, sigma)
    v_profile /= np.trapz(v_profile, s_uniform)  # normalize to unit area
    v_profile *= s[-1] / duration                # scale to total path / total time

    # --- Integrate velocity profile to get cumulative path fraction vs time ---
    s_of_t = np.cumsum(v_profile)
    s_of_t /= s_of_t[-1]  # ensure final = 1
    t_uniform = np.linspace(0, duration, len(s_uniform))
    
    # --- Interpolate mapping t -> s ---
    f_s = interp1d(t_uniform, s_of_t, fill_value="extrapolate", bounds_error=False)
    
    # --- Define new time samples ---
    t_new = np.arange(0, duration, dt)
    s_new = f_s(t_new)  # get new s progression (monotonic increasing)

    # --- Interpolate trajectory along s ---
    f_traj = interp1d(s, traj, axis=0, fill_value="extrapolate")
    traj_new = f_traj(s_new)

    traj = {'t': np.linspace(0, duration, N), 'x': traj_new, 'xdot': np.gradient(traj_new, axis=0)/dt}
    return traj
