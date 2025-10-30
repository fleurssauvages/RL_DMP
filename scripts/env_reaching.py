#usr/bin/env python3
"""
Simple 3D reaching environment with a spherical obstacle.
Simulates only end-effector kinematics controlled by mixture-of-PD DMP.
"""
import numpy as np
from scipy.interpolate import interp1d
from numba import njit

@njit(cache=True)
def rollout_numba(Kp, X, centers, sigmas, V, duration, dt, start_x, start_xdot):
    D = start_x.shape[0]
    K = centers.shape[0]
    timesteps = int(np.ceil(duration / dt))
    xs = np.zeros((timesteps, D))
    xdots = np.zeros_like(xs)
    xddots = np.zeros_like(xs)
    
    x = start_x.copy()
    x_dot = start_xdot.copy()
    
    for i in range(timesteps):
        t = i * dt
        exps = np.exp(-0.5 * ((t - centers)**2) / (sigmas + 1e-12))
        hs = exps / (np.sum(exps) + 1e-12)
        
        # Vectorized acceleration calculation
        diff = X - x
        acc = np.zeros(D)
        for k in range(K):
            acc += hs[k] * (Kp[k] @ diff[k] - V @ x_dot)
        
        x_dot += acc * dt
        x += x_dot * dt
        xs[i] = x
        xdots[i] = x_dot
        xddots[i] = acc
        
    return xs, xdots, xddots      

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
        Run a rollout using numba. First initialization is slower, then the loop is x4 faster approx.
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

        xs, xdots, xddots = rollout_numba(Kp, X, centers, sigmas, V,
                                        self.dmp.duration, self.dt, x0, v0)

        collided = False
        if self.obstacles:
            obs_centers = np.array([ob['center'] for ob in self.obstacles])
            obs_radii = np.array([ob['radius'] for ob in self.obstacles])
            dists = np.linalg.norm(xs[:, None, :3] - obs_centers[None, :, :3], axis=2)
            collided = np.any(dists <= obs_radii)

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
