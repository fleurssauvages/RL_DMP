#usr/bin/env python3
"""
Demo using MultiAgentPowerRL for diverse exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

from scripts.dmp import MixturePD
from scripts.env_reaching import ReachingEnv
from scripts.demo_utils import make_demo_6D, init_from_demo, plot_environment, set_axes_equal, plot_orientations, make_exploration_std
from scripts.multiagent_power_rl import MultiAgentPowerRL
import pickle

def main(seed=1, doAnimation=True, export=False):
    np.random.seed(seed)
    
    # --- Parameters ---
    D, K = 6, 4
    duration, dt = 1.0, 0.01
    weight_demo, weight_goal = 0.05, 0.95
    weight_jerk, weight_end_vel = 0.005, 0.05
    n_iterations, rollouts_per_agent = 120, 4
    n_agents = 5
    exploration_std = make_exploration_std(D, K, sigma_pos=0.05, sigma_ori=0.1, sigma_kp=0.02)
    decay = 0.98

    # --- DMP and environment setup ---
    dmp = MixturePD(D=D, K=K, duration=duration, kp_diag=160.0, vel_gain=12.0)
    demo = make_demo_6D(duration=duration, timesteps=int(duration / dt))
    dmp = init_from_demo(dmp, demo, kp_diag=80.0)
    goal = demo["x"][-1]

    center = demo["x"][int(len(demo["x"]) / 2), :3]
    demo_len = np.linalg.norm(demo["x"][-1, :3] - demo["x"][0, :3])
    obstacles = [
        {'center': center, 'radius': demo_len/6},
        {'center': center + np.array([0, 1, 0]) * demo_len/4, 'radius': demo_len/6},
        {'center': center - np.array([0, 1, 0]) * demo_len/4, 'radius': demo_len/6},
        {'center': center + np.array([0, 0, 1]) * demo_len/4, 'radius': demo_len/6},
        {'center': center - np.array([0, 0, 1]) * demo_len/4, 'radius': demo_len/6},
    ]
    env = ReachingEnv(dmp, dt=dt, obstacles=obstacles, demo_traj=demo, goal=goal)

    # --- Multi-agent RL system ---
    population = MultiAgentPowerRL(
        dmp.get_flat_params(),
        exploration_std=exploration_std,
        n_agents=n_agents,
        reuse_top_n=3,
        diversity_strength=0.1 * np.mean(exploration_std)
    )

    all_returns = []
    best_traj = None
    best_R = -np.inf

    # --- Visualization setup ---
    if doAnimation:
        fig = plt.figure(figsize=(22, 14))
        ax_traj = fig.add_subplot(121, projection='3d')
        ax_ret = fig.add_subplot(122)
        fig.canvas.manager.set_window_title('Multi-Agent Trajectories')

    # --- Main loop ---
    iter_times = np.zeros(n_iterations)
    for it in range(n_iterations):
        t0 = time.time()
        population.reset_histories()

        results_all = []
        best_trajs_per_agent = []
        best_returns_per_agent = []

        # --- Parallel rollout for all agents at once ---
        def rollout_job(agent_id):
            agent_local = population.agents[agent_id]
            params_k = agent_local.sample_policy()
            traj = env.simulate_numba(params_k)
            Rk = env.rollout_return(traj, w_demo=weight_demo, w_goal=weight_goal,
                                    w_jerk=weight_jerk, w_end_vel=weight_end_vel)
            return agent_id, params_k, traj, Rk

        # Create one job per (agent, rollout)
        jobs = [agent_id for agent_id in range(n_agents) for _ in range(rollouts_per_agent)]

        # Run all rollouts
        results_all = [rollout_job(agent_id) for agent_id in jobs]

        # Prepare collections
        rollouts_data = []  # for plotting

        # Assign results per agent
        for agent_id, params_k, traj, Rk in results_all:
            agent = population.agents[agent_id]
            agent.history_params.append(params_k)
            agent.history_returns.append(Rk)
            rollouts_data.append((agent_id, traj, Rk))
            all_returns.append(Rk)

        # Update each agent and apply diversity
        population.update_agents()
        population.apply_diversity_pressure()
        population.update_exploration(exploration_std * (decay ** it))
        population.update_diversity_strength(population.diversity_strength * decay)

        # Find best-performing agent and trajectory
        best_idx, best_agent = population.best_agent()
        best_R_iter = np.max(best_agent.history_returns)
        if best_R_iter > best_R:
            best_R = best_R_iter
            best_traj = env.simulate_numba(best_agent.theta)
            dmp.set_flat_params(best_agent.theta)

        t1 = time.time()
        iter_times[it] = time.time() - t0
        if doAnimation:
            print(f"Iteration {it+1}/{n_iterations} | Best Agent {best_idx} | Best Return {best_R:.3f}")
            print(f"Iteration time: {t1 - t0:.3f}s")
        
        # --- Compute best traj/return per agent ---
        best_trajs_per_agent = []
        best_returns_per_agent = []

        for agent_id, agent in enumerate(population.agents):
            if len(agent.history_returns) == 0:
                # safety fallback (should not happen)
                best_trajs_per_agent.append(None)
                best_returns_per_agent.append(-np.inf)
                continue

            best_idx_local = np.argmax(agent.history_returns)
            best_params = agent.history_params[best_idx_local]
            best_returns_per_agent.append(agent.history_returns[best_idx_local])

            # Re-simulate to get best trajectory (fast with numba)
            best_traj = env.simulate_numba(best_params)
            best_trajs_per_agent.append(best_traj)

        # --- Visualization ---
        if doAnimation:
            fig.clf()
            ax_traj = fig.add_subplot(121, projection='3d')
            ax_ret = fig.add_subplot(122)

            # Plot environment and demo
            plot_environment(ax_traj, demo, obstacles, goal)
            plot_orientations(ax_traj, demo, step=10, scale=0.05)

            # Plot all agent rollouts (faint, colored)
            cmap = plt.cm.get_cmap("nipy_spectral", n_agents)
            for agent_id, params_k, traj, Rk in results_all:
                xs, ys, zs = traj['x'][:, 0], traj['x'][:, 1], traj['x'][:, 2]
                ax_traj.plot(xs, ys, zs, color=cmap(agent_id), alpha=0.3)
                
            # === plot best trajectory per agent ===
            for i, traj_best_local in enumerate(best_trajs_per_agent):
                xs, ys, zs = traj_best_local['x'][:, 0], traj_best_local['x'][:, 1], traj_best_local['x'][:, 2]
                ax_traj.plot(xs, ys, zs, color=cmap(i), linewidth=2.5, label=f"Agent {i} best (R={best_returns_per_agent[i]:.3f})")
                plot_orientations(ax_traj, traj_best_local, step=10, scale=0.02, alpha=0.4)

            # === Returns plot (vectorized, color-coded per agent) ===
            rollouts_per_iter_total = n_agents * rollouts_per_agent
            num_iters_so_far = len(all_returns) // rollouts_per_iter_total

            returns_matrix = np.reshape(
                all_returns[:num_iters_so_far * rollouts_per_iter_total],
                (num_iters_so_far, n_agents, rollouts_per_agent)
            )

            colors = np.array([cmap(i) for i in range(n_agents)])

            # x positions repeated for each rollout
            x_vals = np.repeat(np.arange(num_iters_so_far), n_agents * rollouts_per_agent)

            # Flattened returns, color per agent
            flat_returns = returns_matrix.reshape(-1)
            agent_ids = np.tile(np.repeat(np.arange(n_agents), rollouts_per_agent), num_iters_so_far)
            color_values = colors[agent_ids]

            ax_ret.scatter(
                x_vals, flat_returns,
                color=color_values,
                alpha=0.5, s=12
            )

            # --- Plot summary curves ---
            mean_returns = np.mean(returns_matrix.reshape(num_iters_so_far, -1), axis=1)
            best_per_agent = np.max(returns_matrix, axis=2)  # shape (iters, n_agents)

            # Mean
            ax_ret.plot(np.arange(num_iters_so_far), mean_returns, 'k-', linewidth=1.5, label='Mean per iter')

            # Optional: per-agent best curves (thin, colored)
            for i in range(n_agents):
                ax_ret.plot(np.arange(num_iters_so_far), best_per_agent[:, i],
                            color=cmap(i), linewidth=1, alpha=0.6)

            # Axes setup
            ax_ret.set_title("Returns per iteration (color = agent)")
            ax_ret.set_xlabel("Iteration")
            ax_ret.set_ylabel("Return")
            ax_ret.grid(True)
            ax_ret.set_xlim(0, n_iterations)
            ax_ret.set_ylim(-0.5, 1.0)
            ax_ret.legend(fontsize='small')

            # Finalize plot
            ax_traj.set_title(f"Iteration {it+1}: Best Return {best_R:.3f}")
            ax_traj.legend()
            set_axes_equal(ax_traj)
            fig.tight_layout()
            plt.pause(0.01)

    print("\n Multi-agent training complete.")
    print(f"Median iteration time: {np.median(iter_times):.4f}s")
    print(f"Median framerate: {1.0/np.median(iter_times):.2f} it/s")
    if doAnimation:
        plt.show()
    
    if export:
        for i in range(n_agents):
            best_traj_i = best_trajs_per_agent[i]
            data = {
                "trajectory": best_traj_i,
                "duration": duration,
                "dt": dt
                }
            filename = f"records/trajectory" + str(i) + ".pkl"
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved trajectory to {filename}")


if __name__ == "__main__":
    main(seed=1, doAnimation=False, export=False)