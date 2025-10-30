# main.py
"""
Example experiment:
 - Build a demo
 - Initialize DMP from demo
 - Create environment with an obstacle
 - Create PowerRL and run several iterations of sampling + updating
 - Plot returns vs rollout
 - Resample the obtain trajectory for minimal jerk
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from scripts.dmp import MixturePD
from scripts.power_rl import PowerRL
from scripts.env_reaching import ReachingEnv
from scripts.demo_utils import make_demo_6D, init_from_demo, plot_environment, set_axes_equal, plot_orientations, make_exploration_std
from scripts.resample import resample_min_jerk

def main(seed=1, doAnimation=False, export=False):
    print(f"\n🚀 Starting experiment with seed={seed}")
    np.random.seed(seed)

    # ----- Parameters and Setup -----
    D, K = 6, 4
    duration, dt = 1.0, 0.01
    weight_demo, weight_goal = 0.2, 0.8
    weight_jerk = 0.005
    weight_end_vel = 0.05
    n_iterations, rollouts_per_iter, reused_rollouts = 120, 6, 3
    exploration_std = make_exploration_std(D, K, sigma_pos=0.05, sigma_ori=0.1, sigma_kp=0.02)
    decay = 0.98  # decay per iteration

    # Create DMP + demo
    dmp = MixturePD(D=D, K=K, duration=duration, kp_diag=160.0, vel_gain=12.0)
    demo = make_demo_6D(duration=duration, timesteps=int(duration / dt))
    dmp = init_from_demo(dmp, demo, kp_diag=80.0)
    goal = demo['x'][-1]

    # Create environment with obstacles
    center_of_obstacles = np.array(demo['x'][int(len(demo['x'])/2), :3])
    demo_length = np.linalg.norm(demo['x'][-1, :3] - demo['x'][0, :3])
    obstacles = [
        {'center': center_of_obstacles, 'radius': demo_length/6},
        {'center': center_of_obstacles + np.array([0, 1, 0]) * demo_length/4, 'radius': demo_length/6},
        {'center': center_of_obstacles - np.array([0, 1, 0]) * demo_length/4, 'radius': demo_length/6},
        {'center': center_of_obstacles + np.array([0, 0, 1]) * demo_length/4, 'radius': demo_length/6},
        {'center': center_of_obstacles - np.array([0, 0, 1]) * demo_length/4, 'radius': demo_length/6},
    ]
    env = ReachingEnv(dmp, dt=dt, obstacles=obstacles, demo_traj=demo, goal=goal)
    
    # Create PoWER RL agent
    agent = PowerRL(dmp.get_flat_params(), exploration_std=exploration_std, reuse_top_n=reused_rollouts)
    all_returns = []
    best_traj_prev, best_R_prev = {'x': np.zeros((int(duration/dt), 6)), 'xdot': np.zeros((int(duration/dt), 6)), 't': np.zeros(int(duration/dt))}, -100

    # ----- Visualization of trajectories -----
    fig_traj = plt.figure(figsize=(22,14))
    ax_traj = fig_traj.add_subplot(121, projection='3d')
    plot_environment(ax_traj, demo, obstacles, goal)
    plot_orientations(ax_traj, demo, step=10, scale=0.05)
    
    ax_vel = fig_traj.add_subplot(143)
    v = demo['xdot'][:, :3]
    t = demo['t']
    v_mag = np.linalg.norm(v, axis=1)
    ax_vel.plot(t, v_mag, color= "blue", label=f'Demo')
    ax_vel.legend(loc='upper right', fontsize='small')
    ax_vel.grid(True)
    ax_vel.set_title("Velocity magnitude (best rollout per iteration)")
    ax_vel.set_xlabel("Time [s]")
    ax_vel.set_ylabel("|v| [m/s]")
    ax_vel.set_xlim(0, duration)
    ax_vel.set_ylim(0, 1.6)
    
    if doAnimation:
        plt.show(block=False)
        plt.pause(0.01)
    
    fig_traj.canvas.manager.set_window_title('Trajectory')

    def rollout_job():
        params_k = agent.sample_policy()
        traj = env.simulate_numba(params_k)
        Rk = env.rollout_return(traj,  w_demo=weight_demo, w_goal=weight_goal, w_jerk=weight_jerk, w_end_vel=weight_end_vel)
        return params_k, traj, Rk
    
    # ----- RL Loop -----
    iter_times = np.zeros(n_iterations)
    for it in range(n_iterations):
        print(f"\n=== Iteration {it+1}/{n_iterations} ===")
        t0 = time.time()
        agent.reset_history()
        iter_returns = []
        rollouts = []

        results = [rollout_job() for _ in range(rollouts_per_iter)]
        
        params_list, traj_list, R_list = zip(*results)
        agent.history_params.extend(params_list)
        agent.history_returns.extend(map(float, R_list))
        rollouts.extend(zip(traj_list, R_list))
        iter_returns.extend(R_list)
        all_returns.extend(R_list)

        # Update policy
        new_theta = agent.update()
        dmp.set_flat_params(new_theta)
        agent.update_exploration(exploration_std * (decay ** it))

        # Find best trajectory of this iteration
        best_idx = np.argmax(iter_returns)
        best_traj, best_R = rollouts[best_idx]
        if best_R > best_R_prev:
            best_traj_prev, best_R_prev = best_traj, best_R
        else:
            best_traj, best_R = best_traj_prev, best_R_prev
            
        t1 = time.time()
        print(f"Iteration time: {t1 - t0:.4f}s")
        iter_times[it] = t1 - t0

        # # ----- Visualization of rollouts -----
        if doAnimation:
            fig_traj.clf()
            ax_traj = fig_traj.add_subplot(121, projection='3d')
            ax_vel = fig_traj.add_subplot(143)
            plot_environment(ax_traj, demo, obstacles, goal)
            plot_orientations(ax_traj, demo, step=10, scale=0.05)
            
            v = demo['xdot'][:, :3]
            t = demo['t']
            v_mag = np.linalg.norm(v, axis=1)
            ax_vel.plot(t, v_mag, color= "blue", label=f'Demo')
            
            # plot all rollouts (faint)
            for traj, R in rollouts:
                xs, ys, zs = traj['x'][:, 0], traj['x'][:, 1], traj['x'][:, 2]
                ax_traj.plot(xs, ys, zs, color='gray', alpha=0.3)
                
                v = traj['xdot'][:, :3]
                t = traj['t']
                v_mag = np.linalg.norm(v, axis=1)
                ax_vel.plot(t, v_mag, color= "gray", alpha=0.3)
                
            # highlight best trajectory
            xs, ys, zs = best_traj['x'][:, 0], best_traj['x'][:, 1], best_traj['x'][:, 2]
            ax_traj.plot(xs, ys, zs, 'g-', linewidth=3, label=f'Best Traj (R={best_R:.3f})')
            plot_orientations(ax_traj, best_traj, step=10, scale=0.05)
            
            v = best_traj['xdot'][:, :3]
            t = best_traj['t']
            v_mag = np.linalg.norm(v, axis=1)
            ax_vel.plot(t, v_mag, color= "green", label=f'Best Traj')
            ax_vel.legend(loc='upper right', fontsize='small')
            ax_vel.grid(True)
            ax_vel.set_title("Velocity magnitude (best rollout per iteration)")
            ax_vel.set_xlabel("Time [s]")
            ax_vel.set_ylabel("|v| [m/s]")
            ax_vel.set_xlim(0, duration)
            ax_vel.set_ylim(0, 1.6)

            ax_traj.set_title(f"Iteration {it+1}: Best Return {best_R:.3f}")
            ax_traj.set_xlabel('X'); ax_traj.set_ylabel('Y'); ax_traj.set_zlabel('Z')
            ax_traj.legend()
            set_axes_equal(ax_traj)

            ax_returns = fig_traj.add_subplot(144)
            all_returns_plot = np.reshape(all_returns, (it+1, rollouts_per_iter))
            indexes = np.reshape(np.repeat(np.arange(it+1), rollouts_per_iter), (it+1, rollouts_per_iter))
            ax_returns.scatter(indexes.flatten(), all_returns_plot.flatten())
            ax_returns.plot(np.max(all_returns_plot, axis=1), 'g--')
            ax_returns.set_xlabel('Rollout index')
            ax_returns.set_ylabel('Return')
            ax_returns.set_xlim([1, n_iterations])
            ax_returns.set_ylim([-0.25, 1.0])
            ax_returns.set_title('Returns over rollouts')
            ax_returns.grid(True)
            
            fig_traj.canvas.draw()
            fig_traj.canvas.flush_events()

    print(f"Median iteration time: {np.median(iter_times):.4f}s")
    print(f"Median framerate: {1.0/np.median(iter_times):.2f} it/s")
    traj_mj = resample_min_jerk(best_traj, duration=duration, N_new=int(duration/dt))
    
    xs, ys, zs = traj_mj['x'][:, 0], traj_mj['x'][:, 1], traj_mj['x'][:, 2]
    ax_traj.plot(xs, ys, zs, 'g-.', linewidth=3, label=f'Resampled')
    plot_orientations(ax_traj, traj_mj, step=10, scale=0.05)
    ax_traj.legend()
    set_axes_equal(ax_traj)
    
    v = traj_mj['xdot'][:, :3]
    t = traj_mj['t']
    v_mag = np.linalg.norm(v, axis=1)
    ax_vel.plot(t, v_mag, label=f'Resampled', color= "green", linestyle='-.')
    ax_vel.legend(loc='upper right', fontsize='small')

    fig_traj.canvas.draw()
    fig_traj.canvas.flush_events()
    plt.show()
    
    if export:
        data = {
            "trajectory": traj_mj,
            "duration": duration,
            "dt": dt
        }
        filename = f"records/trajectory.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved trajectory to {filename}")


if __name__ == "__main__":
    main(seed=1, doAnimation=False, export=False)