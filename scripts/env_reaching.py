#usr/bin/env python3
"""
Simple 3D reaching environment with a spherical obstacle.
Simulates only end-effector kinematics controlled by mixture-of-PD DMP.
"""
import numpy as np
from scipy.interpolate import interp1d

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

            # if not collided and obs_centers is not None:
            #     dists = np.linalg.norm(x[:3] - obs_centers[:, :3], axis=1)
            #     if np.any(dists <= obs_radii):
            #         collided = True

            # xs[i] = x
            # xdots[i] = x_dot
            xddots[i] = acc
            t += self.dt
            
        xdots = np.cumsum(xddots, axis=0) * self.dt
        xs = np.cumsum(xdots, axis=0) * self.dt
            
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
        if self.demo is not None:
            demo_x = self.demo['x']
            # Resample demo to T if not same length
            if demo_x.shape[0] != T:
                demo_t = np.linspace(0.0, self.dmp.duration, demo_x.shape[0])
                f = interp1d(demo_t, demo_x, axis=0)
                demo_x = f(np.linspace(0.0, self.dmp.duration, T))
            diffs = np.linalg.norm(traj['x'] - demo_x, axis=1)
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
