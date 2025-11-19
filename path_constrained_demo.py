#!/usr/bin/env python3
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyspacemouse 
from scripts.demo_utils import make_demo_6D, set_axes_equal
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import threading

# -------------------------------------------------------------------------
# Utility functions: resampling, cylinder radius, smoothing
# -------------------------------------------------------------------------
def resample_traj(x, n_points=1000):
    """
    Resample trajectory x (N x D) to n_points using linear interpolation
    over normalized time [0, 1].
    """
    N = x.shape[0]
    t_orig = np.linspace(0.0, 1.0, N)
    t_new = np.linspace(0.0, 1.0, n_points)
    x_new = np.empty((n_points, x.shape[1]))
    for d in range(x.shape[1]):
        x_new[:, d] = np.interp(t_new, t_orig, x[:, d])
    return x_new

def compute_tube_radius(points, obstacles, r_max=3.0, r_min=1e-2):
    """
    For each 3D point, compute the largest radius (<= r_max) such that
    no spherical obstacle intersects the ball around the point.

    obstacles: list of dicts with keys 'center' (3D array) and 'radius' (float)
    points: (N, 3)
    returns: radii, shape (N,)
    """
    radii = np.zeros(points.shape[0])
    if len(obstacles) == 0:
        radii[:] = r_max
        return radii

    for i, p in enumerate(points):
        # distance to surface of each obstacle
        clearances = []
        for obs in obstacles:
            dc = np.linalg.norm(p - obs['center'])
            d_surface = dc - obs['radius']
            clearances.append(d_surface)
        min_clearance = min(clearances)
        r = max(min(min_clearance, r_max), r_min)
        radii[i] = r
    return radii

def smooth_radius(r, window=101, passes=2):
    """
    Heavy low-pass via moving-average convolution, optionally repeated.
    Ensures window is odd and not larger than the signal.
    """
    if len(r) < 3:
        return r.copy()

    # ensure odd window and <= len(r)
    window = min(window, len(r) // 2 * 2 + 1)
    if window < 3:
        return r.copy()

    kernel = np.ones(window, dtype=float) / window
    r_smooth = r.copy()
    for _ in range(passes):
        r_padded = np.pad(r_smooth, (window // 2, window // 2), mode='edge')
        r_smooth = np.convolve(r_padded, kernel, mode='valid')
    return r_smooth

# -------------------------------------------------------------------------
# Load trajectories and reconstruct obstacles (same logic as RL demo)
# -------------------------------------------------------------------------

def load_trajectories(folder="records", n_traj=5):
    traj_list = []
    duration = None
    dt = None
    for i in range(n_traj):
        filename = os.path.join(folder, f"trajectory{i}.pkl")
        with open(filename, "rb") as f:
            data = pickle.load(f)
        traj_list.append(data["trajectory"])  # dict with 'x'
        duration = data["duration"]
        dt = data["dt"]
    return traj_list, duration, dt


def make_obstacles(duration, dt):
    """
    Reconstruct the same obstacle layout as in demo_multiagent_6D.py,
    in the DMP/demo coordinate frame (no Panda transform).
    """
    demo = make_demo_6D(duration=duration, timesteps=int(duration / dt))
    center = demo["x"][int(len(demo["x"]) / 2), :3]
    demo_len = np.linalg.norm(demo["x"][-1, :3] - demo["x"][0, :3])

    obstacles = [
        {'center': center, 'radius': demo_len / 6},
        {'center': center + np.array([0, 1, 0]) * demo_len / 4, 'radius': demo_len / 6},
        {'center': center - np.array([0, 1, 0]) * demo_len / 4, 'radius': demo_len / 6},
        {'center': center + np.array([0, 0, 1]) * demo_len / 4, 'radius': demo_len / 6},
        {'center': center - np.array([0, 0, 1]) * demo_len / 4, 'radius': demo_len / 6},
    ]
    return obstacles

# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------

def plot_obstacles(ax, obstacles, color='red', alpha=0.15):
    """
    Plot spherical obstacles as translucent surfaces.
    """
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    for obs in obstacles:
        c = obs['center']
        R = obs['radius']
        x = R * np.outer(np.cos(u), np.sin(v)) + c[0]
        y = R * np.outer(np.sin(u), np.sin(v)) + c[1]
        z = R * np.outer(np.ones_like(u), np.cos(v)) + c[2]
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def build_tubes(traj_list, obstacles, n_samples=1000, r_max=3.0):
    """
    For each trajectory, resample to n_samples, compute radii and smooth them.
    Returns list of dicts with keys:
        'centers' : (n_samples, 3)
        'radii'   : (n_samples,)
    """
    tubes = []
    for traj in traj_list:
        x = traj['x']  # (N, 6) -> positions in first 3 dims
        positions = x[:, :3]
        centers = resample_traj(positions, n_points=n_samples)[:, :3]
        radii = compute_tube_radius(centers, obstacles, r_max=r_max)
        radii = smooth_radius(radii, window=101, passes=3)
        tubes.append({'centers': centers, 'radii': radii})
    return tubes

# -------------------------------------------------------------------------
# Main viewer with SpaceMouse
# -------------------------------------------------------------------------

def draw_sphere(ax, center, r, color, alpha=0.2, res=12):
    u = np.linspace(0, 2*np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
    return ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

def draw_disc3d(ax, center, radius, normal=(0,0,1), color='blue', alpha=0.2, res=40):
    center = np.asarray(center)
    normal = np.asarray(normal)
    normal = normal / np.linalg.norm(normal)

    # ---- build circle in YZ plane (normal = X axis) ----
    theta = np.linspace(0, 2*np.pi, res)
    
    circle_yz = np.c_[
        np.zeros(res),             # X
        radius * np.cos(theta),    # Y
        radius * np.sin(theta)     # Z
    ]

    # ---- find rotation matrix from (1,0,0) to the target normal ----
    x = np.array([1,0,0], float)
    v = np.cross(x, normal)
    c = np.dot(x, normal)
    s = np.linalg.norm(v)

    if s < 1e-8:    # already aligned
        R = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx@vx*((1-c)/(s**2))

    # ---- rotate the disc into correct orientation ----
    circle_rot = circle_yz @ R.T

    # ---- translate to center point ----
    verts = center + circle_rot

    # ---- create polygon ----
    poly = Poly3DCollection([verts], alpha=alpha)
    poly.set_facecolor(color)
    poly.set_edgecolor(color)
    ax.add_collection3d(poly)

    return poly

def draw_cube(ax, center, size=0.05, color='k', alpha=0.3):
    cx, cy, cz = center
    s = size / 2.0

    # 8 vertices of the cube
    verts = np.array([
        [cx - s, cy - s, cz - s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s],
        [cx - s, cy + s, cz + s],
    ])

    # 6 cube faces
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[4], verts[7], verts[3], verts[0]],
    ]

    poly = Poly3DCollection(faces, alpha=alpha)
    poly.set_facecolor(color)
    poly.set_edgecolor(color)
    ax.add_collection3d(poly)
    return poly

def update_cube(poly, center, R, size=0.05):
    cx, cy, cz = center
    s = size / 2.0

    # base cube vertices centered at origin
    verts0 = np.array([
        [-s, -s, -s],
        [ s, -s, -s],
        [ s,  s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
        [ s, -s,  s],
        [ s,  s,  s],
        [-s,  s,  s],
    ])

    # rotate
    verts = verts0 @ R.T

    # translate
    verts += np.array(center)

    # faces
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[4], verts[7], verts[3], verts[0]],
    ]

    poly.set_verts(faces)

def apply_deadzone(v, dtrans=0.05, drot=0.05):
    """
    Apply deadzone to each component of a vector.
    v : numpy array
    dz : deadzone threshold (0–1, SpaceMouse axes are in [-1,1])
    """
    out = v.copy()
    for i in range(len(out)):
        if i < 3:  # translation components
            dz = dtrans
        else:      # rotation components
            dz = drot

        if abs(out[i]) < dz:
            out[i] = 0.0
        else:
            out[i] += -np.sign(out[i]) * dz
    return out

def main():
    foldername = "records"
    n_traj = 2

    traj_list, duration, dt = load_trajectories(folder=foldername, n_traj=n_traj)
    print(f"Loaded {n_traj} trajectories, duration={duration:.3f}s, dt={dt:.3f}s")

    obstacles = make_obstacles(duration, dt)

    # Precompute tubes (centerline + radius)
    print("Precomputing trajectory cylinders...")
    tubes = build_tubes(traj_list, obstacles, n_samples=1000, r_max=3.0)

    # --- Matplotlib 3D figure ---
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title("Interactive Trajectories + Cylinders + SpaceMouse")

    # Colormap for trajectories (no strong orange)
    cmap = plt.cm.get_cmap("viridis", n_traj)

    # Plot obstacles
    plot_obstacles(ax, obstacles, color='red', alpha=0.2)

    # Plot cylinders: centerline + sparse radius markers
    for k, tube in enumerate(tubes):
        centers = tube['centers']
        radii = tube['radii']

        # centerline
        ax.plot(
            centers[:, 0], centers[:, 1], centers[:, 2],
            color=cmap(k), linewidth=1.5, alpha=0.7, label=f"traj {k}"
        )

        # visualize radius by scatter size every few points
        step = 20
        for idx in range(0, len(centers), step):
            normal = centers[idx]
            draw_disc3d(
                ax,
                center=centers[idx],
                radius=radii[idx],     # data-unit radius
                normal=(1,0,0),        # disc facing "up"
                color=cmap(k),
                alpha=0.1
            )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Trajectories with obstacle-aware cylinders")
    set_axes_equal(ax)
    ax.legend(loc="upper right")

    # Initial current position: start of first tube
    current_pos = tubes[0]['centers'][0].copy()
    cube_R = np.eye(3) # no rotation
    current_cube = draw_cube(ax, current_pos, size=0.03, color='k')
    ax.legend(loc="upper right")
    fig.canvas.draw()
    plt.pause(0.05)

    # --- SpaceMouse setup ---
    print("Opening SpaceMouse (pyspacemouse)...")
    success = pyspacemouse.open()
    global latest_v 
    latest_v = np.zeros(6)
    lock = threading.Lock()
    global run_input 
    run_input= True

    def mouse_reader():
        global latest_v, run_input
        while run_input:
            state = pyspacemouse.read()
            if state is not None:
                with lock:
                    latest_v[:] = [
                        state.x, state.y, state.z,      # translation
                        -state.pitch, state.roll, -state.yaw    # rotation
                    ]
                    latest_v = apply_deadzone(latest_v, dtrans=0.05, drot=0.2)
            time.sleep(0.0001)   # 10 kHz polling

    threading.Thread(target=mouse_reader, daemon=True).start()
    if not success:
        print("Could not open SpaceMouse. Check connection/permissions.")
        print("You can still move the point manually by editing code, but SpaceMouse input won't work.")
    else:
        print("SpaceMouse connected. Move the cap to translate the black point in 3D.")

    # Motion gains
    trans_gain = 0.1  # [m/s] per full deflection
    rot_gain = 1.5   # [rad/s] per full deflection
    loop_dt = 0.05    # [s] loop period

    # Main loop: update current position based on joystick, update plot
    try:
        last_time = time.time()
        while plt.fignum_exists(fig.number):
            now = time.time()
            dt_loop = now - last_time
            last_time = now

            with lock:
                v = latest_v.copy()
            
            t_v = v[:3]   # translational velocity
            r_v = v[3:]   # rotational velocity
            
            current_pos += trans_gain * t_v * dt_loop
            rx, ry, rz = r_v * rot_gain * dt_loop
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx),  np.cos(rx)]
            ])

            Ry = np.array([
                [ np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ])

            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz),  np.cos(rz), 0],
                [0, 0, 1]
            ])

            # Update cube orientation
            cube_R = cube_R @ Rz @ Ry @ Rx

            # Constrain to a tube
            i = np.argmin(np.sum((centers - current_pos)**2, axis=1))
            tube_center = centers[i]
            tube_radius = radii[i]

            d = current_pos - tube_center
            dist = np.linalg.norm(d)

            if dist > tube_radius:
                current_pos = tube_center + d/dist * tube_radius

            # Update scatter position in 3D
            update_cube(current_cube, current_pos, cube_R, size=0.03)

            fig.canvas.draw_idle()
            plt.pause(loop_dt)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if success:
            run_input = False
            pyspacemouse.close()
        print("Exiting viewer.")


if __name__ == "__main__":
    main()
