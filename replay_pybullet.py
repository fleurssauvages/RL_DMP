#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import time
import roboticstoolbox as rtb
import os
import pickle
import numpy as np

from MPC.QP_solver import QPController
from MPC.LMPC_solver import LinearMPCController
import spatialmath as sm
from scripts.demo_utils import make_demo_6D

filename = "records/trajectory.pkl"  # <-- replace with your filename
with open(filename, 'rb') as f:
    data = pickle.load(f)

traj = data["trajectory"]  # Nx6 array [x, y, z, roll, pitch, yaw]
duration = data["duration"]
dt = data["dt"]
print(f"Loaded samples of duration {duration:.2f}s, dt={dt:.3f}s")

# Start PyBullet in GUI mode
physicsClient = p.connect(p.GUI)
p.setTimeStep(dt)
p.setRealTimeSimulation(1)

# Set up environment
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Place camera closer to the Panda base
p.resetDebugVisualizerCamera(
    cameraDistance=1.0,          # zoom in (smaller = closer)
    cameraYaw=45,                # rotate around z-axis
    cameraPitch=-30,             # look down
    cameraTargetPosition=[0, 0, 0.5]  # focus on robot base
)

# Hide the side panel (mini camera views and GUI widgets)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

# Load plane and robot
p.loadURDF("plane.urdf")
with open(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"), "r") as f:
    print("Loading URDF from:", os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"))
    urdf = f.read()
panda_id = p.loadURDF("franka_panda/panda.urdf",
                        [0, 0, 0],
                        p.getQuaternionFromEuler([0, 0, 0]),
                        useFixedBase=True)

# Get controllable revolute joints (7 DOF arm)
num_joints = p.getNumJoints(panda_id)
joint_indices = [i for i in range(num_joints) if p.getJointInfo(panda_id, i)[2] == p.JOINT_REVOLUTE]
sim_time = 0.0

# Reset to qr
panda = rtb.models.Panda()
panda.qr[1] = -1.7  # avoid singularity
panda.qr[3] = -3.1
panda.q = panda.qr
for j in joint_indices:
    p.resetJointState(panda_id, j, panda.qr[j])
    
# Robot Controller
lmpc_solver = LinearMPCController(horizon=25, dt=dt, gamma = 0.1,
                                    u_min=np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]),
                                    u_max=np.array([ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5]))

# Init QP solver for IK with safety
qp_solver = QPController(panda)
qp_solver.solve(np.zeros((6,1)))
Tini = panda.fkine(panda.qr)
    
# Plot obstacles
demo = make_demo_6D(duration=duration, timesteps=int(duration / dt))
for i in range(len(demo['x'])-1):
    t1 = (sm.SE3.Trans(demo['x'][i, :3]) * Tini).t
    t2 = (sm.SE3.Trans(demo['x'][i+1, :3]) * Tini).t
    p.addUserDebugLine(t1, t2, [1, 0, 0], lineWidth=3)
    
center_of_obstacles = (sm.SE3.Trans(demo['x'][int(len(demo['x'])/2), :3]) * Tini).t
demo_length = np.linalg.norm(demo['x'][-1, :3] - demo['x'][0, :3])
obstacles = [
    {'center': center_of_obstacles, 'radius': demo_length/8},
    {'center': center_of_obstacles + np.array([0, 1, 0]) * demo_length/4, 'radius': demo_length/8},
    {'center': center_of_obstacles - np.array([0, 1, 0]) * demo_length/4, 'radius': demo_length/8},
    {'center': center_of_obstacles + np.array([0, 0, 1]) * demo_length/4, 'radius': demo_length/8},
    {'center': center_of_obstacles - np.array([0, 0, 1]) * demo_length/4, 'radius': demo_length/8},
]

for obs in obstacles:
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=obs['radius'],
        rgbaColor=[1, 0, 0, 0.7]  # red, semi-transparent
    )
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=obs['radius']
    )
    p.createMultiBody(
        baseMass=0,  # static
        baseVisualShapeIndex=visual_shape_id,
        baseCollisionShapeIndex=collision_shape_id,
        basePosition=obs['center'].tolist()
    )
    
# Plot trajectory
for i in range(len(traj['x'])-1):
    t1 = (sm.SE3.Trans(traj['x'][i, :3]) * Tini).t
    t2 = (sm.SE3.Trans(traj['x'][i+1, :3]) * Tini).t
    p.addUserDebugLine(t1, t2, [0, 1, 0], lineWidth=3)

try:
    Uopt = np.zeros((6 * lmpc_solver.horizon,))
    while True:
        if sim_time < duration:
            T_des = sm.SE3.Trans(traj['x'][int(sim_time/dt), :3]) * Tini
        else:
            T_des = sm.SE3.Trans(traj['x'][-1, :3]) * Tini
        # Compute desired pose from trajectory
        T_current = panda.fkine(panda.q)
        Uopt, Xopt, poses = lmpc_solver.solve(T_current, T_des, xi0=Uopt[0:6])

        #Solve QP
        qp_solver.update_robot_state(panda)
        qp_solver.add_local_tangent_plane_constraints(obstacles, margin = 0.0)
        qp_solver.solve(Uopt[0:6], alpha=0.02, beta=0.01)
        
        panda.qd = qp_solver.solution
        
        # Apply low level velocity control
        p.setJointMotorControlArray(
            panda_id,
            jointIndices=joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities= qp_solver.solution,
        )
        
        # Step simulation
        q, qd = [], []
        for j in range(7):
            state = p.getJointState(panda_id, j)
            q.append(state[0])
            qd.append(state[1])
        panda.q = np.array(q)
        
        sim_time += dt
        time.sleep(dt)
        
        if sim_time > duration + 2.0:
            p.disconnect()
            break

except KeyboardInterrupt:
    print("Simulation stopped by user.")
    p.disconnect()