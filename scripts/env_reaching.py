#usr/bin/env python3
"""
Simple 3D reaching environment with a spherical obstacle.
Simulates only end-effector kinematics controlled by mixture-of-PD DMP.
"""
import numpy as np
from scipy.interpolate import interp1d
from numba import njit, set_num_threads, prange
import math

class ReachingEnv:
    def __init__(self, dmp: object, dt=0.01,
                 obstacles=None,  # list of dicts [{'center': np.array, 'radius': float}, ...]
                 demo_traj=None, goal=None):
        """
        dmp: MixturePD instance
        obstacles: list of {'center': np.array, 'radius': float}
        demo_traj: dict with keys 't','x'
        goal: target point
        """
        self.dmp = dmp
        self.dt = dt
        self.timesteps = int(np.ceil(dmp.duration / dt))
        self.demo = demo_traj
        if demo_traj is not None:
            demo_t = demo_traj['t']
            demo_x = demo_traj['x']
            target_t = np.linspace(0.0, dmp.duration, self.timesteps)
            self.demo_resampled = interp1d(demo_t, demo_x, axis=0)(target_t)
        self.goal = np.array(goal) if goal is not None else np.zeros(dmp.D)

        # Normalize obstacle data
        self.obstacles = []
        if obstacles is not None:
            for ob in obstacles:
                c = np.array(ob['center'])
                r = float(ob['radius'])
                self.obstacles.append({'center': c, 'radius': r})  
    
    def simulate(self, params, start_x=None, start_xdot=None):
        """
        Run a rollout using flattened params for the DMP.
        Returns trajectory dict with 't','x','xdot','xddot', 'collided' boolean
        """
        self.dmp.set_flat_params(params)
        D = self.dmp.D
        # initial states
        if start_x is None:
            x = np.zeros(D)
        else:
            x = np.array(start_x)
        if start_xdot is None:
            x_dot = np.zeros(D)
        else:
            x_dot = np.array(start_xdot)

        xs = np.zeros((self.timesteps, len(x)))
        xdots = np.zeros_like(xs)
        xddots = np.zeros_like(xs)
        obs_centers = np.array([ob['center'] for ob in self.obstacles]) if self.obstacles else None
        obs_radii = np.array([ob['radius'] for ob in self.obstacles]) if self.obstacles else None
        
        collided = False
        t = 0.0
        for i in range(self.timesteps):
            acc = self.dmp.desired_acceleration(x, x_dot, t)
            x_dot = x_dot + acc * self.dt
            x = x + x_dot * self.dt

            xs[i] = x
            xdots[i] = x_dot
            xddots[i] = acc
            t += self.dt
            
        if obs_centers is not None:
            dists = np.linalg.norm(xs[:, None, :3] - obs_centers[None, :, :3], axis=2)
            collided = np.any(dists <= obs_radii)

        traj = {
            't': np.linspace(0.0, self.dmp.duration, self.timesteps),
            'x': np.array(xs),
            'xdot': np.array(xdots),
            'xddot': np.array(xddots),
            'collided': collided
        }
        return traj
    
    def simulate_numba(self, params, start_x=None, start_xdot=None):
        """
        Run a rollout using numba. First initialization is slower, then the loop is much faster.
        Returns trajectory dict with 't','x','xdot','xddot', 'collided' boolean
        """
        self.dmp.set_flat_params(params)
        D = self.dmp.D
        x0 = np.zeros(D) if start_x is None else np.array(start_x)
        v0 = np.zeros(D) if start_xdot is None else np.array(start_xdot)

        Kp = np.stack(self.dmp.Kp)
        X = np.stack(self.dmp.X)
        centers = self.dmp.centers
        sigmas = self.dmp.sigmas
        V = self.dmp.V

        timesteps = int(np.ceil(self.dmp.duration / self.dt))
        xs = np.zeros((timesteps, D))
        xdots = np.zeros_like(xs)
        xddots = np.zeros_like(xs)
        acc = np.zeros(D)
        if self.obstacles:
            obs_centers = np.array([ob['center'] for ob in self.obstacles], dtype=np.float32)
            obs_radii = np.array([ob['radius'] for ob in self.obstacles], dtype=np.float32)
        else:
            obs_centers = None
            obs_radii = None

        xs, xdots, xddots, collided = rollout_numba(Kp, X, centers, sigmas, V,
                                        self.dmp.duration, self.dt, x0, v0, timesteps, xs, xdots, xddots, acc, obs_centers, obs_radii)

        traj = {'t': np.linspace(0, self.dmp.duration, xs.shape[0]),
                'x': xs, 'xdot': xdots, 'xddot': xddots, 'collided': collided}
        return traj


    def rollout_return(self, traj, w_demo=0.5, w_goal=0.5, w_jerk=0.005, w_end_vel=0.01):
        """
        Reward similar to eq (4) in Kormushev:
         - w_demo term: match to demonstration along the path (averaged)
         - w_goal term: distance to goal at final time
        If collision occurs, penalize by reducing rewards
        """
        T = traj['x'].shape[0]
        # Path matching term: if demo provided, compute per-step distance
        path_score = 0.0
        if self.demo_resampled is not None:
            diffs = np.linalg.norm(traj['x'] - self.demo_resampled, axis=1)
            path_score = (1.0 / T) * np.sum(np.exp(-diffs))
            
        # Final goal term
        final_dist = np.linalg.norm(traj['x'][-1] - self.goal)
        goal_score = np.exp(-5*final_dist)
        R_total = w_demo * path_score + w_goal * goal_score
        
        # Collision penalty
        if traj['collided']:
            min_d = np.min([
                np.min(np.linalg.norm(traj['x'][:, :3] - ob['center'], axis=1)) - ob['radius']
                for ob in self.obstacles
            ])
            R_total -= 5.0 * np.clip(-min_d, 0, 0.2) # stronger penalty for deeper collisions
        
        # Jerk penalty to encourage smoothness
        dt = self.dt
        xddot = traj.get('xddot', None)
        if xddot is not None and len(xddot) > 2:
            jerk = np.gradient(xddot, axis=0) / dt
            jerk_mag = np.linalg.norm(jerk, axis=1)
            jerk_cost = np.mean(jerk_mag**2)
        else:
            jerk_cost = 0.0
        jerk_cost /= len(jerk_mag) # Normalize to trajectory length
        R_total -= w_jerk * jerk_cost
        
        # Initial & final velocity penalty
        vel_penalty = 0.0
        xdot = traj.get('xdot', None)
        if xdot is not None and xdot.shape[0] >= 1:
            v0 = xdot[0]
            vT = xdot[-1]
            # Squared-norm penalty (L2) scaled by timestep
            vel_penalty = np.dot(v0, v0) + np.dot(vT, vT)

        # subtract penalization (lambda_endvel controls strength)
        R_total -= w_end_vel * vel_penalty

        return R_total
    
    def rollout_return_using_numba(self, traj, w_demo=0.5, w_goal=0.5, w_jerk=0.005, w_end_vel=0.01):
        xs = traj['x']      # (T, D)
        xddot = traj.get('xddot', None)
        xdot = traj.get('xdot', None)
        collided = bool(traj.get('collided', False))

        # Ensure we always pass arrays (numba-friendly shapes)
        if xddot is None:
            xddot = np.empty((0, xs.shape[1]), dtype=xs.dtype)
        if xdot is None:
            xdot = np.empty((0, xs.shape[1]), dtype=xs.dtype)

        # demo_resampled: either (T, D) or 0-length
        if self.demo_resampled is None:
            demo_arr = np.empty((0, xs.shape[1]), dtype=xs.dtype)
        else:
            demo_arr = self.demo_resampled

        # obstacles → centers & radii arrays
        if self.obstacles:
            obs_centers = np.array([ob['center'] for ob in self.obstacles], dtype=xs.dtype)
            obs_radii = np.array([ob['radius'] for ob in self.obstacles], dtype=xs.dtype)
        else:
            obs_centers = np.empty((0, 3), dtype=xs.dtype)
            obs_radii = np.empty((0,), dtype=xs.dtype)

        R_total = rollout_return_numba(xs,
                                    xddot,
                                    xdot,
                                    demo_arr,
                                    self.goal.astype(xs.dtype),
                                    self.dt,
                                    collided,
                                    obs_centers,
                                    obs_radii,
                                    w_demo,
                                    w_goal,
                                    w_jerk,
                                    w_end_vel)
        return R_total
    
    def simulate_and_return_traj_numba(self, params,
                                   start_x=None,
                                   start_xdot=None,
                                   w_demo=0.5, w_goal=0.5,
                                   w_jerk=0.005, w_end_vel=0.01):

        self.dmp.set_flat_params(params)
        D = self.dmp.D

        x0 = np.zeros(D) if start_x is None else np.array(start_x)
        v0 = np.zeros(D) if start_xdot is None else np.array(start_xdot)

        Kp = np.stack(self.dmp.Kp)
        X = np.stack(self.dmp.X)
        centers = self.dmp.centers
        sigmas = self.dmp.sigmas
        V = self.dmp.V

        T = int(np.ceil(self.dmp.duration / self.dt))

        if self.demo_resampled is None:
            demo_arr = np.empty((0, D))
        else:
            demo_arr = self.demo_resampled

        if self.obstacles:
            obs_centers = np.array([o['center'] for o in self.obstacles])
            obs_radii = np.array([o['radius'] for o in self.obstacles])
        else:
            obs_centers = np.zeros((0, 3))
            obs_radii = np.zeros((0,))

        traj, R_total, collided = rollout_and_return_numba(
            Kp, X, centers, sigmas, V,
            self.dmp.duration, self.dt,
            x0, v0, T,
            obs_centers, obs_radii,
            demo_arr, self.goal,
            w_demo, w_goal, w_jerk, w_end_vel
        )

        return traj, float(R_total)

@njit(cache=True)
def rollout_numba(Kp, X, centers, sigmas, V, duration, dt,
                  start_x, start_xdot, timesteps,
                  xs, xdots, xddots, acc,
                  obs_centers, obs_radii):
    D = start_x.shape[0]
    K = centers.shape[0]

    x = start_x.copy()
    x_dot = start_xdot.copy()
    collided = False

    exps = np.empty(K)
    hs = np.empty(K)
    diff = np.empty((K, D))
    acc = np.empty(D)

    for i in range(timesteps):
        t = i * dt

        # compute exps & hs
        sum_exps = 0.0
        for k in range(K):
            num = t - centers[k]
            tmp = -0.5 * (num * num) / (sigmas[k] + 1e-12)
            exps[k] = np.exp(tmp)
            sum_exps += exps[k]
        denom = sum_exps + 1e-12
        for k in range(K):
            hs[k] = exps[k] / denom

        # diff = X - x
        for k in range(K):
            for d in range(D):
                diff[k, d] = X[k, d] - x[d]

        # acc = sum_k hs[k] * (Kp[k] @ diff[k] - V @ x_dot)
        for d in range(D):
            acc[d] = 0.0

        # precompute V @ x_dot once
        Vx = np.empty(D)
        for r in range(D):
            s = 0.0
            for c in range(D):
                s += V[r, c] * x_dot[c]
            Vx[r] = s

        for k in range(K):
            # Kp[k] @ diff[k]
            Kp_diff = np.empty(D)
            for r in range(D):
                s = 0.0
                for c in range(D):
                    s += Kp[k, r, c] * diff[k, c]
                Kp_diff[r] = s

            for d in range(D):
                acc[d] += hs[k] * (Kp_diff[d] - Vx[d])

        # integrate
        for d in range(D):
            x_dot[d] += acc[d] * dt
            x[d] += x_dot[d] * dt

        xs[i] = x
        xdots[i] = x_dot
        xddots[i] = acc

        # collision check (no sqrt)
        if obs_centers is not None and obs_radii is not None and not collided:
            n_obs = obs_centers.shape[0]
            for k in range(n_obs):
                dx = x[0] - obs_centers[k, 0]
                dy = x[1] - obs_centers[k, 1]
                dz = x[2] - obs_centers[k, 2]
                dist_sq = dx*dx + dy*dy + dz*dz
                r_sq = obs_radii[k] * obs_radii[k]
                if dist_sq <= r_sq:
                    collided = True
                    break

    return xs, xdots, xddots, collided

@njit(cache=True)
def rollout_return_numba(xs,
                        xddot,
                        xdot,
                        demo_resampled,
                        goal,
                        dt,
                        collided,
                        obs_centers,
                        obs_radii,
                        w_demo,
                        w_goal,
                        w_jerk,
                        w_end_vel):
    """
    Numba version of the reward. All inputs must be arrays/scalars, no dicts or objects.
    xs: (T, D)
    xddot: (T, D) or (0, D) if not available
    xdot: (T, D) or (0, D) if not available
    demo_resampled: (T, D) or (0, D) if no demo
    goal: (D,)
    obs_centers: (N_obs, 3) or (0, 3) if no obstacles
    obs_radii: (N_obs,)
    """
    T = xs.shape[0]

    # --- Path matching term ---
    path_score = 0.0
    if demo_resampled.shape[0] == T:
        acc = 0.0
        for t in range(T):
            # ||x_t - demo_t||
            diff_sq = 0.0
            for d in range(xs.shape[1]):
                tmp = xs[t, d] - demo_resampled[t, d]
                diff_sq += tmp * tmp
            diff = np.sqrt(diff_sq)
            acc += np.exp(-diff)
        path_score = acc / T

    # --- Final goal term ---
    D = xs.shape[1]
    final_dist_sq = 0.0
    for d in range(D):
        tmp = xs[T - 1, d] - goal[d]
        final_dist_sq += tmp * tmp
    final_dist = np.sqrt(final_dist_sq)
    goal_score = np.exp(-5.0 * final_dist)

    R_total = w_demo * path_score + w_goal * goal_score

    # --- Collision penalty ---
    if collided and obs_centers.shape[0] > 0:
        # min over obstacles of (min over time of (dist_to_surface))
        # dist_to_surface = ||x[:3] - center|| - radius
        min_d = 1e9
        for o in range(obs_centers.shape[0]):
            radius = obs_radii[o]
            # min over time
            for t in range(T):
                # only first 3 dims used for obstacle distance
                dist_sq = 0.0
                for d in range(3):
                    tmp = xs[t, d] - obs_centers[o, d]
                    dist_sq += tmp * tmp
                dist = np.sqrt(dist_sq)
                d_surface = dist - radius
                if d_surface < min_d:
                    min_d = d_surface

        # penalty = 5 * clip(-min_d, 0, 0.2)
        if min_d < 0.0:
            depth = -min_d
            if depth > 0.2:
                depth = 0.2
            R_total -= 5.0 * depth

    # --- Jerk penalty ---
    jerk_cost = 0.0
    n_xddot = xddot.shape[0]
    if n_xddot > 2:
        # simple finite-difference jerk along time
        # jerk[t] ~ (xddot[t+1] - xddot[t]) / dt   (you can also do higher-order)
        Tj = n_xddot - 1
        acc_cost = 0.0
        for t in range(Tj):
            mag_sq = 0.0
            for d in range(D):
                j = (xddot[t + 1, d] - xddot[t, d]) / dt
                mag_sq += j * j
            acc_cost += mag_sq
        jerk_cost = acc_cost / Tj  # mean over time
        # normalize to trajectory length (like your original intent)
        jerk_cost = jerk_cost / T

    R_total -= w_jerk * jerk_cost

    # --- Initial & final velocity penalty ---
    vel_penalty = 0.0
    if xdot.shape[0] >= 1:
        v0 = xdot[0]
        vT = xdot[xdot.shape[0] - 1]

        v0_sq = 0.0
        vT_sq = 0.0
        for d in range(D):
            v0_sq += v0[d] * v0[d]
            vT_sq += vT[d] * vT[d]

        vel_penalty = v0_sq + vT_sq

    R_total -= w_end_vel * vel_penalty

    return R_total

@njit(cache=True)
def rollout_and_return_numba(Kp, X, centers, sigmas, V, duration, dt,
                             start_x, start_xdot, timesteps,
                             obs_centers, obs_radii,
                             demo_resampled, goal,
                             w_demo, w_goal, w_jerk, w_end_vel):
    """
    Fused version: integrates dynamics AND accumulates reward in one pass.

    Parameters are the same as your old rollout + reward, but:
    - We accumulate all costs on the fly.
    - We only track x (position), x_dot (velocity), and acc (acceleration).
    """

    D = start_x.shape[0]
    K = centers.shape[0]

    xs     = np.zeros((timesteps, D))
    xdots  = np.zeros((timesteps, D))
    xddots = np.zeros((timesteps, D))

    x = start_x.copy()
    x_dot = start_xdot.copy()
    acc = np.zeros(D)

    x = start_x.copy()
    x_dot = start_xdot.copy()
    acc = np.zeros(D)

    collided = False

    # pre-allocate stuff you are already using in rollout_numba
    exps = np.empty(K)
    hs = np.empty(K)
    diff = np.empty((K, D))

    # --- reward accumulators ---
    # path matching
    path_acc = 0.0
    has_demo = demo_resampled.shape[0] == timesteps

    # obstacle distance / collision
    has_obs = obs_centers.shape[0] > 0
    min_d = 1e9  # min distance-to-surface

    # jerk
    jerk_acc_cost = 0.0
    have_prev_acc = False
    prev_acc = np.zeros(D)

    Kp_diff = np.empty(D)
    Vx = np.empty(D)

    # v0, vT (v0 is start_xdot, vT will be final x_dot)
    # v0 is known already; vT will be x_dot after the last step

    for i in range(timesteps):
        t = i * dt

        # compute exps & hs
        sum_exps = 0.0
        for k in range(K):
            num = t - centers[k]
            tmp = -0.5 * (num * num) / (sigmas[k] + 1e-12)
            exps[k] = np.exp(tmp)
            sum_exps += exps[k]
        denom = sum_exps + 1e-12
        for k in range(K):
            hs[k] = exps[k] / denom

        # diff = X - x
        for k in range(K):
            for d in range(D):
                diff[k, d] = X[k, d] - x[d]

        # acc = sum_k hs[k] * (Kp[k] @ diff[k] - V @ x_dot)
        for d in range(D):
            acc[d] = 0.0

        # precompute V @ x_dot once
        for r in range(D):
            s = 0.0
            for c in range(D):
                s += V[r, c] * x_dot[c]
            Vx[r] = s

        for k in range(K):
            # Kp[k] @ diff[k]
            for r in range(D):
                s = 0.0
                for c in range(D):
                    s += Kp[k, r, c] * diff[k, c]
                Kp_diff[r] = s

            for d in range(D):
                acc[d] += hs[k] * (Kp_diff[d] - Vx[d])

        # integrate
        for d in range(D):
            x_dot[d] += acc[d] * dt
            x[d] += x_dot[d] * dt

        # collision check (no sqrt)
        if obs_centers is not None and obs_radii is not None and not collided:
            n_obs = obs_centers.shape[0]
            for k in range(n_obs):
                dx = x[0] - obs_centers[k, 0]
                dy = x[1] - obs_centers[k, 1]
                dz = x[2] - obs_centers[k, 2]
                dist_sq = dx*dx + dy*dy + dz*dz
                r_sq = obs_radii[k] * obs_radii[k]
                if dist_sq <= r_sq:
                    collided = True
                    break

        # --- jerk accumulation from acceleration ---
        if have_prev_acc:
            mag_sq = 0.0
            for d in range(D):
                j = (acc[d] - prev_acc[d]) / dt
                mag_sq += j * j
            jerk_acc_cost += mag_sq
        else:
            have_prev_acc = True

        for d in range(D):
            prev_acc[d] = acc[d]

        # --- path matching term (demo) ---
        if has_demo:
            diff_sq = 0.0
            for d in range(D):
                tmp = x[d] - demo_resampled[i, d]
                diff_sq += tmp * tmp
            dist = math.sqrt(diff_sq)
            path_acc += math.exp(-5 * dist)

        # --- obstacle distance / collision ---
        if has_obs:
            for o in range(obs_centers.shape[0]):
                radius = obs_radii[o]
                dist_sq = 0.0
                # only first 3 dims used for obstacle distance
                for d in range(3):
                    tmp = x[d] - obs_centers[o, d]
                    dist_sq += tmp * tmp
                dist = math.sqrt(dist_sq)
                d_surface = dist - radius
                if d_surface < min_d:
                    min_d = d_surface
                if d_surface <= 0.0:
                    collided = True

        xs[i] = x
        xdots[i] = x_dot
        xddots[i] = acc

    # --- Path score ---
    if has_demo and timesteps > 0:
        path_score = path_acc / timesteps
    else:
        path_score = 0.0

    # --- Final goal term ---
    final_dist_sq = 0.0
    for d in range(D):
        tmp = x[d] - goal[d]
        final_dist_sq += tmp * tmp
    final_dist = math.sqrt(final_dist_sq)
    goal_score = math.exp(-5.0*final_dist)

    R_total = w_demo * path_score + w_goal * goal_score

    # --- Collision penalty (same as rollout_return_numba) ---
    if has_obs and min_d < 0.0:
        depth = -min_d
        if depth > 0.2:
            depth = 0.2
        R_total -= 5.0 * depth

    # --- Jerk penalty (same normalization as numba version) ---
    if timesteps > 1 and have_prev_acc:
        Tj = timesteps - 1
        jerk_cost = (jerk_acc_cost / Tj) / timesteps
    else:
        jerk_cost = 0.0

    R_total -= w_jerk * jerk_cost

    # --- Initial & final velocity penalty ---
    v0_sq = 0.0
    vT_sq = 0.0
    for d in range(D):
        v0_sq += start_xdot[d] * start_xdot[d]
        vT_sq += x_dot[d] * x_dot[d]
    vel_penalty = v0_sq + vT_sq

    R_total -= w_end_vel * vel_penalty

    return xs, R_total, collided