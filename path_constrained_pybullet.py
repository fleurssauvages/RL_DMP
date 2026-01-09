#!/usr/bin/env python3
import os
import pickle
import time
import threading

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import spatialmath as sm
import roboticstoolbox as rtb
import pyspacemouse

from MPC.QP_solver import QPController
from MPC.LMPC_solver import LinearMPCController
from scripts.demo_utils import make_demo_6D

# ============================================================
#  Utility functions: trajectories, tubes, obstacles
# ============================================================

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

    for i, pnt in enumerate(points):
        clearances = []
        for obs in obstacles:
            dc = np.linalg.norm(pnt - obs['center'])
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

    window = min(window, len(r) // 2 * 2 + 1)
    if window < 3:
        return r.copy()

    kernel = np.ones(window, dtype=float) / window
    r_smooth = r.copy()
    for _ in range(passes):
        r_padded = np.pad(r_smooth, (window // 2, window // 2), mode='edge')
        r_smooth = np.convolve(r_padded, kernel, mode='valid')
    return r_smooth


def build_tubes(traj_list, obstacles, n_samples=1000, r_max=3.0):
    """
    For each trajectory, resample to n_samples, compute radii and smooth them.
    Returns list of dicts with keys:
        'centers' : (n_samples, 3)
        'radii'   : (n_samples,)
    All in the SAME FRAME as 'obstacles'.
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


def make_obstacles_dmp(duration, dt):
    """
    Same logic as in shared_controller_demo: obstacles in DMP/demo frame.
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
    return obstacles, demo


def project_point_to_all_tubes(current_pos, tubes_world):
    """
    Project point p onto the surface of the *closest tube* across all tubes.
    tubes_world: list of dicts with:
        'centers': (N,3)
        'radii'  : (N,)
    """

    # --- Find closest point across ALL tubes ---
    best_dist_per_tube = []
    best_idx_per_tube = []

    for tube in tubes_world:
        centers_t = tube['centers']
        d = np.sum((centers_t - current_pos)**2, axis=1)
        i = np.argmin(d)
        best_idx_per_tube.append(i)
        best_dist_per_tube.append(np.sqrt(d[i]))

    isInside = np.any([tubes_world[t]['radii'][best_idx_per_tube[t]] >= best_dist_per_tube[t] for t in range(len(tubes_world))])

    best_tube = np.argmin(best_dist_per_tube)
    best_i    = best_idx_per_tube[best_tube]
    tube_center = tubes_world[best_tube]['centers'][best_i]
    tube_radius = tubes_world[best_tube]['radii'][best_i]
    d = current_pos - tube_center
    dist = np.linalg.norm(d)
    
    if not isInside:
        current_pos = tube_center + d/dist * tube_radius

    return current_pos


# ============================================================
#  Async SpaceMouse 6-DOF input with deadzone
# ============================================================

latest_v = np.zeros(6)
_spacemouse_running = False
_spacemouse_lock = threading.Lock()


def apply_deadzone(v, dtrans=0.05, drot=0.2):
    """
    Apply component-wise deadzone.
    First 3 components = translation, last 3 = rotation.
    """
    out = v.copy()
    for i in range(len(out)):
        dz = dtrans if i < 3 else drot
        if abs(out[i]) < dz:
            out[i] = 0.0
        else:
            out[i] = np.sign(out[i]) * (abs(out[i]) - dz)
    return out


def start_spacemouse():
    global _spacemouse_running, latest_v

    if _spacemouse_running:
        return

    success = pyspacemouse.open()
    if not success:
        print("Could not open SpaceMouse. Check connection/permissions.")
        return

    print("SpaceMouse connected.")
    _spacemouse_running = True

    def reader():
        global latest_v, _spacemouse_running
        while _spacemouse_running:
            state = pyspacemouse.read()
            if state is not None:
                raw = np.array([
                    state.x, state.y, state.z,
                    -state.pitch, state.roll, -state.yaw
                ])
                filtered = apply_deadzone(raw, dtrans=0.05, drot=0.2)
                with _spacemouse_lock:
                    latest_v[:] = filtered
            time.sleep(0.0001)  # ~10 kHz polling

    threading.Thread(target=reader, daemon=True).start()


def stop_spacemouse():
    global _spacemouse_running
    _spacemouse_running = False
    try:
        pyspacemouse.close()
    except Exception:
        pass


def get_spacemouse_6d():
    with _spacemouse_lock:
        return latest_v.copy()

def rotmat_to_quat(R):
    """Convert rotation matrix to quaternion (x,y,z,w) for PyBullet."""
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4*qw)
    qy = (R[0,2] - R[2,0]) / (4*qw)
    qz = (R[1,0] - R[0,1]) / (4*qw)
    return [qx, qy, qz, qw]

def draw_cylinder_segment(p1, p2, radius, rgba=[0,1,0,0.2]):
    import numpy as np
    import pybullet as p

    p1 = np.array(p1)
    p2 = np.array(p2)
    mid = (p1 + p2) / 2
    vec = p2 - p1
    length = np.linalg.norm(vec)

    if length < 1e-6:
        return None

    # cylinder in pybullet is aligned with +Z
    z = vec / length
    # generate orthonormal basis
    up = np.array([0, 0, 1])
    if abs(np.dot(z, up)) > 0.999:
        up = np.array([0, 1, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z]).T

    # convert to quaternion
    orn = p.getQuaternionFromEuler(sm.SO3(R).eul())

    visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=rgba
    )

    body = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=mid.tolist(),
        baseOrientation=orn
    )

    return body

# ============================================================
#  Main: PyBullet teleoperation
# ============================================================

def main():
    foldername = "records"
    n_traj = 2

    # ----------------- load trajectories in DMP frame -----------------
    traj_list = []
    duration = None
    dt = None
    for i in range(n_traj):
        filename = os.path.join(foldername, f"trajectory{i}.pkl")
        with open(filename, "rb") as f:
            data = pickle.load(f)
        traj_list.append(data["trajectory"])  # dict with 'x'
        duration = data["duration"]
        dt = data["dt"]

    print(f"Loaded {n_traj} trajectories, duration={duration:.3f}s, dt={dt:.3f}s")

    # ----------------- build obstacles + tubes in DMP frame -----------------
    obstacles_dmp, demo = make_obstacles_dmp(duration, dt)
    tubes_dmp = build_tubes(traj_list, obstacles_dmp, n_samples=1000, r_max=3.0)
    tube0_dmp = tubes_dmp[0]
    centers_dmp = tube0_dmp["centers"]          # (N,3) in DMP frame
    radii_tube = tube0_dmp["radii"]             # (N,)

    # ----------------- PyBullet setup -----------------
    physicsClient = p.connect(p.GUI)
    p.setTimeStep(dt)
    p.setRealTimeSimulation(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5],
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

    p.loadURDF("plane.urdf")
    panda_urdf_path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
    with open(panda_urdf_path, "r") as f:
        print("Loading URDF from:", panda_urdf_path)
        _ = f.read()
    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        [0, 0, 0],
        p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
    )

    # get joint indices
    num_joints = p.getNumJoints(panda_id)
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(panda_id, i)[2] == p.JOINT_REVOLUTE]

    # create toolbox Panda and set qr
    panda = rtb.models.Panda()
    panda.qr[1] = -1.7
    panda.qr[3] = -3.1
    panda.q = panda.qr
    for j in joint_indices:
        p.resetJointState(panda_id, j, panda.qr[j])

    # initial EE frame
    Tini = panda.fkine(panda.qr)

    # ----------------- obstacles + tubes in WORLD frame -----------------
    # convert obstacles from DMP to world (same layout as shared_controller, but in world)
    obstacles_world = []
    for obs in obstacles_dmp:
        c_dmp = obs["center"]
        c_world = (sm.SE3.Trans(c_dmp) * Tini).t  # 3x1
        obstacles_world.append({
            "center": np.array(c_world).reshape(3,),
            "radius": obs["radius"],
        })

    # draw obstacles as PyBullet spheres
    for obs in obstacles_world:
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=obs["radius"],
            rgbaColor=[1, 0, 0, 0.6]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=obs["center"].tolist()
        )

    # convert tube centers from DMP to world
    tube_bodies = []

    # Loop over all tubes (one tube per trajectory)
    cmap = plt.cm.get_cmap("viridis", n_traj)

    for k, tube in enumerate(tubes_dmp):
        centers_dmp = tube["centers"]    # (1000,3) in DMP frame
        radii_tube  = tube["radii"]      # (1000,)
        traj_color = list(cmap(k)[0:3]) 
        rgba = traj_color + [0.1]

        # Convert centers to world frame
        centers_world = []
        for c in centers_dmp:
            cw = (sm.SE3.Trans(c) * Tini).t
            centers_world.append(np.array(cw).reshape(3,))
        centers_world = np.vstack(centers_world)

        # Draw cylinders every N samples (20 is ok)
        for i in range(0, len(centers_world) - 1, 30):
            r = radii_tube[i]
            p1 = centers_world[i]
            p2 = centers_world[i+1]
            tube_bodies.append(
                draw_cylinder_segment(p1, p2, r, rgba=rgba)
            )

    tubes_world = []
    for tube in tubes_dmp:
        centers_dmp = tube["centers"]
        radii_dmp   = tube["radii"]

        centers_world = []
        for c in centers_dmp:
            cw = (sm.SE3.Trans(c) * Tini).t
            centers_world.append(np.array(cw).reshape(3,))
        centers_world = np.vstack(centers_world)

        tubes_world.append({
            "centers": centers_world,
            "radii"  : radii_dmp
        })

    # draw all recorded trajectories in world
    for k, traj in enumerate(traj_list):
        for i in range(len(traj["x"]) - 1):
            t1 = (sm.SE3.Trans(traj["x"][i, :3]) * Tini).t
            t2 = (sm.SE3.Trans(traj["x"][i + 1, :3]) * Tini).t
            p.addUserDebugLine(
                t1,
                t2,
                cmap(k)[0:3],
                lineWidth=2
            )

    # ----------------- Robot controllers -----------------
    lmpc_solver = LinearMPCController(
        horizon=25,
        dt=dt,
        gamma=0.1,
        u_min=np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]),
        u_max=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    )

    qp_solver = QPController(panda)
    qp_solver.solve(np.zeros((6, 1)))

    # ----------------- Teleop cursor (cube) -----------------
    cube_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.02, 0.02, 0.02],
        rgbaColor=[0, 0, 0, 0.5]
    )
    cube_body = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=cube_visual,
        basePosition=centers_world[0].tolist()
    )

    # ----------------- SpaceMouse -----------------
    start_spacemouse()

    trans_gain = 0.3   # [m/s] per full deflection
    rot_gain = 1.5     # [rad/s] per full deflection
    loop_dt = dt       # controller loop time

    current_pos = centers_world[0].copy()
    R_ee = panda.fkine(panda.q).R   # end-effector orientation (world frame)
    cube_quat = rotmat_to_quat(R_ee)

    try:
        Uopt = np.zeros((6 * lmpc_solver.horizon,))
        while True:
            v6 = get_spacemouse_6d()
            t_v = v6[:3] * trans_gain
            r_v = v6[3:] * rot_gain

            # --- integrate translation ---
            current_pos = current_pos + t_v * loop_dt
            # project onto tube surface
            current_pos = project_point_to_all_tubes(current_pos, tubes_world)

            # --- integrate rotation (roll-pitch-yaw) ---
            rx, ry, rz = r_v * loop_dt
            ry, rz = -ry, -rz  # adjust for PyBullet frame
            if abs(rx) + abs(ry) + abs(rz) > 1e-9:
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
                R_ee = R_ee @ Rz @ Ry @ Rx
                # re-orthogonalize
                U, _, Vt = np.linalg.svd(R_ee)
                R_ee = U @ Vt
                cube_quat = rotmat_to_quat(R_ee)

            # update cube in PyBullet
            # (orientation visualized as identity; we could also convert R_ee to quaternion and use it)
            p.resetBasePositionAndOrientation(
                cube_body,
                current_pos.tolist(),
                cube_quat
            )

            # --- build desired SE3 for MPC/IK ---
            T_mat = np.eye(4)
            T_mat[:3, :3] = R_ee
            T_mat[:3, 3] = current_pos
            T_des = sm.SE3(np.array(T_mat, dtype=float))

            # current robot pose
            T_current = panda.fkine(panda.q)

            # MPC: desired twist sequence
            Uopt, Xopt, poses = lmpc_solver.solve(T_current, T_des, xi0=Uopt[0:6])

            qp_solver.update_robot_state(panda)
            qp_solver.add_local_tangent_plane_constraints(obstacles_world, margin=0.02)
            qp_solver.solve(Uopt[0:6], alpha=0.02, beta=0.01)

            panda.qd = qp_solver.solution

            p.setJointMotorControlArray(
                panda_id,
                jointIndices=joint_indices,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=panda.qd,
            )

            # sync toolbox model with PyBullet
            q = []
            for j in range(7):
                state = p.getJointState(panda_id, j)
                q.append(state[0])
            panda.q = np.array(q)

            time.sleep(loop_dt)

    except KeyboardInterrupt:
        print("Teleop stopped by user.")

    finally:
        stop_spacemouse()
        p.disconnect()


if __name__ == "__main__":
    main()
