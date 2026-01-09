#usr/bin/env python3
"""
Demo using MultiAgentPowerRL for diverse exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from scripts.dmp import MixturePD
from scripts.env_reaching import ReachingEnv
from scripts.demo_utils import make_demo_6D, init_from_demo, plot_environment_no_demo, set_axes_equal, plot_orientations, make_exploration_std
from scripts.multiagent_power_rl import MultiAgentPowerRL
import pickle

def update_obstacles_sinusoid(obstacles, base_centers, amp_xyz, t, period_iters, dt):
    """Translate each obstacle center by the same sinusoidal offset (xyz).

    Parameters
    ----------
    obstacles : list[dict]
        Each dict must contain a 'center' (np.array shape (3,)) and optionally other fields.
    base_centers : list[np.ndarray]
        Rest positions for obstacle centers (same length as obstacles).
    it : int
        Current outer iteration.
    amp_xyz : np.ndarray
        Amplitude in meters for x/y/z, shape (3,).
    t : float
        Current time in seconds.
    """
    phase = 2.0 * np.pi * (t / (period_iters * dt))  # full oscillation every period_iters
    for obs, c0, amp in zip(obstacles, base_centers, amp_xyz):
        offset = amp * np.sin(phase)  # shape (3,)
        obs["center"] = c0 + offset
    return offset

def update_obstacles_circular(obstacles, base_centers, radius, it, period_iters):
    """Translate each obstacle center in a circular trajectory in the XY plane.

    Parameters
    ----------
    obstacles : list[dict]
        Each dict must contain a 'center' (np.array shape (3,)) and optionally other fields.
    base_centers : list[np.ndarray]
        Rest positions for obstacle centers (same length as obstacles).
    radius : float
        Radius of the circular trajectory (meters).
    it : int
        Current outer iteration.
    period_iters : int
        Number of iterations for a full circular motion.
    """
    angle = 2.0 * np.pi * (it % period_iters) / period_iters
    for obs, c0 in zip(obstacles, base_centers):
        offset = radius * np.array([0.0, np.cos(angle), np.sin(angle)])
        obs["center"] = c0 + offset
    return offset


def scale_exploration_by_obstacle_motion(
    base_std, delta_obs, motion_ref,
):
    """Scale exploration std based on obstacle motion magnitude.

    delta_obs : float
        Motion magnitude since previous iteration (meters).
    motion_ref : float
        Motion magnitude corresponding to ~1x scaling (meters).
    """
    scale = float(delta_obs) / float(motion_ref + 1e-12)
    return base_std * scale, scale

def main(seed=1, doPlot = True, circular_motion=False):
    np.random.seed(seed)
    
    # --- Parameters ---
    D, K = 6, 4
    duration, dt = 1.0, 0.02
    weight_demo, weight_goal = 0.05, 0.95
    weight_jerk, weight_end_vel = 0.005, 0.05
    n_iterations, rollouts_per_agent = 10, 8
    n_agents = 5
    exploration_std = make_exploration_std(D, K, sigma_pos=0.05, sigma_ori=0.1, sigma_kp=0.02)
    base_exploration_std = exploration_std.copy()
    decay = 0.98

    # --- DMP and environment setup ---
    dmp = MixturePD(D=D, K=K, duration=duration, kp_diag=160.0, vel_gain=12.0)
    demo = make_demo_6D(duration=duration, timesteps=int(duration / dt), curvature=0.01)
    dmp = init_from_demo(dmp, demo, kp_diag=80.0)
    goal = demo["x"][-1]

    center = (demo["x"][0, :3] + demo["x"][-1, :3]) / 2
    vect = (demo["x"][0, :3] + demo["x"][-1, :3])
    demo_len = np.linalg.norm(demo["x"][-1, :3] - demo["x"][0, :3])
    obstacles = [
        {'center': vect/2, 'radius': demo_len/6},
    ]
    env = ReachingEnv(dmp, dt=dt, obstacles=obstacles, demo_traj=demo, goal=goal)
    
    # Obstacle motion parameters
    base_centers = [obs['center'].copy() for obs in obstacles]   # keep "rest" positions
    amp_xyz = [
        np.array([0.0, 0.0, 0.1]), 
    ]  # meters: move obstacles along Y
    period_iters = 60                       # full oscillation every 60 outer iterations
    radius_circular = 0.1                   # meters for circular motion

    # --- Multi-agent RL system ---
    population = MultiAgentPowerRL(
        dmp.get_flat_params(),
        exploration_std=exploration_std,
        n_agents=n_agents,
        reuse_top_n=3,
        diversity_strength=0.1 * np.mean(exploration_std)
    )

    # --- Visualization setup ---
    if doPlot:
        fig = plt.figure(figsize=(22, 14))
        ax_traj = fig.add_subplot(111, projection='3d')
        fig.canvas.manager.set_window_title('Multi-Agent Trajectories')
        middle = (demo["x"][0, :3] + demo["x"][-1, :3]) / 2

        paused = False
        def on_key(event):
            nonlocal paused
            if event.key == " " or event.key == "space":
                paused = not paused
                print("Paused" if paused else "Resumed")

        fig.canvas.mpl_connect("key_press_event", on_key)

    # --- Main loop ---
    nb_points_simulation = 180
    times_per_simulation = np.zeros(nb_points_simulation)
    for i in range(nb_points_simulation):
        t = dt * i
        offset_prev = offset if i > 0 else np.zeros(3)
        if circular_motion:
            offset = update_obstacles_circular(obstacles, base_centers, radius_circular, i, period_iters)
        else:
            offset = update_obstacles_sinusoid(obstacles, base_centers, amp_xyz, t, period_iters, dt)
        env.obstacles = obstacles
        delta_obs = np.linalg.norm(offset - offset_prev)
        exploration_std, scale = scale_exploration_by_obstacle_motion(base_exploration_std, delta_obs, np.linalg.norm(amp_xyz))

        t0 = time.time()
        for it in range(n_iterations):
            population.reset_histories()
            def rollout_job(agent_id):
                agent_local = population.agents[agent_id]
                params_k = agent_local.sample_policy()
                traj, Rk = env.simulate_and_return_traj_numba(
                    params_k,
                    w_demo=weight_demo,
                    w_goal=weight_goal,
                    w_jerk=weight_jerk,
                    w_end_vel=weight_end_vel,
                )
                return agent_id, params_k, traj, Rk

            # Create one job per (agent, rollout)
            jobs = [agent_id for agent_id in range(n_agents) for _ in range(rollouts_per_agent)]

            # Run all rollouts
            results_all = [rollout_job(agent_id) for agent_id in jobs]
            
            # # Assign results per agent
            for agent_id, params_k, _, Rk in results_all:
                agent = population.agents[agent_id]
                agent.add_rollout(params_k, Rk)

            # Update each agent and apply diversity
            population.update_agents()
            population.apply_diversity_pressure(exploration_std=exploration_std*0.1)
            population.update_exploration(exploration_std * (decay ** it))
            population.update_diversity_strength(population.diversity_strength * decay)

        # --- Compute best traj/return per agent ---
        best_trajs_per_agent = []
        best_returns_per_agent = []

        for _, agent in enumerate(population.agents):
            if len(agent.history_returns) == 0:
                best_trajs_per_agent.append(None)
                best_returns_per_agent.append(-np.inf)
                continue

            history_R = np.array(agent.history_returns)
            best_idx_local = np.argmax(history_R)
            best_params = agent.history_params[best_idx_local]
            best_returns_per_agent.append(history_R[best_idx_local])

            best_traj = env.simulate_numba(best_params)
            best_trajs_per_agent.append(best_traj)

        t1 = time.time()
        times_per_simulation[i] = t1 - t0
        print(f"Iteration {i+1}/{nb_points_simulation} completed in {times_per_simulation[i]:.4f}s")

        # # --- Visualization ---
        if doPlot:
            azim = ax_traj.azim
            elev = ax_traj.elev
            ax_traj.cla()

            plot_environment_no_demo(ax_traj, obstacles, goal)

            # Plot all agent rollouts (faint, colored)
            cmap = plt.cm.get_cmap("nipy_spectral", n_agents)
            # for agent_id, params_k, traj, Rk in results_all:
            #     xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
            #     ax_traj.plot(xs, ys, zs, color=cmap(agent_id), alpha=0.3)
                
            # === plot best trajectory per agent ===
            for i, traj_best_local in enumerate(best_trajs_per_agent):
                xs, ys, zs = traj_best_local['x'][:, 0], traj_best_local['x'][:, 1], traj_best_local['x'][:, 2]
                collided = traj_best_local["collided"]
                ax_traj.plot(xs, ys, zs, color=cmap(i), linewidth=2.5, label=f"Agent {i} best (R={best_returns_per_agent[i]:.3f}, {'collided' if collided else 'free'})")
                # plot_orientations(ax_traj, traj_best_local, step=10, scale=0.02, alpha=0.4)

            # Finalize plot
            set_axes_equal(ax_traj)
            ax_traj.set_xlim([middle[0] - demo_len/2, middle[0] + demo_len/2])
            ax_traj.set_ylim([middle[1] - demo_len/2, middle[1] + demo_len/2])
            ax_traj.set_zlim([middle[2] - demo_len/2, middle[2] + demo_len/2])
            ax_traj.legend()
            fig.tight_layout()
            ax_traj.view_init(elev=elev, azim=azim)
            plt.pause(0.001)

            while paused:
                plt.pause(0.05)

    print(f"Average time per iteration over simulation: {np.mean(times_per_simulation):.4f}s")
    print(f"Average frame rate: {1.0/np.mean(times_per_simulation):.2f} Hz")


if __name__ == "__main__":
    main(seed=6, doPlot=True, circular_motion=True)