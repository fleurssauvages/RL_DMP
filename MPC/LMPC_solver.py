import numpy as np
from scipy.spatial.transform import Rotation as R
from qpsolvers import solve_qp
import scipy.sparse as sp
from scipy.linalg import block_diag

""" ------------------------
MPC problem assembly based from 
Alberto, Nicolas Torres, et al. "Linear Model Predictive Control in SE (3) for online trajectory planning in dynamic workspaces." (2022).
https://hal.science/hal-03790059/document
------------------------"""
class LinearMPCController:
    def __init__(self, horizon=10, dt=0.05, gamma=1e-3, u_min=None, u_max=None):
        self.n = 6
        self.horizon = horizon
        self.dt = dt
        self.gamma = gamma
        self.u_min = u_min
        self.u_max = u_max
        self.H = np.zeros((self.n*self.horizon, self.n*self.horizon)) # Hessian
        self.g = np.zeros((self.n*self.horizon, self.n*self.horizon)) # Gradient
        self.A = np.zeros((0, self.n*self.horizon))  # Inequality onstraints
        self.b = np.zeros(0)  # Inequality constraint bounds
        self.eqA = np.zeros((0, self.n*self.horizon))  # Equality constraints
        self.eqb = np.zeros(0) # Equality constraint bounds
        self.lb = -np.ones(self.n) * np.inf  # Lower bounds
        self.ub = np.ones(self.n) * np.inf   # Upper bounds
        self.dt = dt # Time step of the controller loop / simulation
        self.solution = None
        pass
    
    def solve(self, ini_pose, des_pose):
        """
        Solve finite-horizon linear MPC:
        minimize ||X - X_des||^2 + gamma ||U||^2
        s.t. X = A_big x0 + B_big U
        u_min <= U <= u_max (component-wise)
        x0: current state in tangent space (6,)
        xi_des: target in tangent space that is repeated along horizon (6,)
        A,B: system (6x6), here A = I6, B = dlog * dt
        h: horizon length (# time-steps)
        dt: time step (for info only)
        gamma: weight on input norm
        u_min/u_max: (6,) or arrays of length 6*h
        Returns optimal U (stacked) and the predicted X sequence.
        """
        ini_pose = np.array(ini_pose)
        des_pose = np.array(des_pose)
        
        # Linearize the poses in the tangent space with dlog
        Xc_inv = np.linalg.inv(ini_pose)
        xi_current = se3_log(np.eye(4))
        xi_des = se3_log(Xc_inv @ des_pose)
        dlog = compute_dlog_approx(xi_current) # Linearization dlog at xi_current

        # Build big matrices, A = I6, B = dlog * dt
        A_big = np.eye(self.n * self.horizon)
        singleB = compute_dlog_approx(xi_current) * self.dt  # (6x6)
        B_big = np.kron(np.tril(np.ones((self.horizon, self.horizon))), singleB)  # (6h x 6h)
        
        Xd = np.tile(xi_des, self.horizon)
        Xprev = np.tile(xi_current, self.horizon)
        
        # Cost weights
        Q = np.eye(6)       # weight for pose error
        Q_big = block_diag(*[Q for _ in range(self.horizon-1)])  # (12h x 12h)
        Q_big = block_diag(Q_big, np.eye(6) * 150)  # Final state also considered, strong weight on final pose and velocity

        # Cost: (A_big x0 + B_big U - Xd)^T (A_big x0 + B_big U - Xd) + gamma * U^T U
        # Variable is stacked X_pred(= A_big @ np.tile(x0, h) + B_big @ U) and U
        # Formulation according to QP, see QP for more details
        self.H = 2 * (B_big.T @ Q_big @ B_big + self.gamma * np.eye(self.n*self.horizon))
        self.g = 2 * (B_big.T @ Q_big @ (A_big @ Xprev - Xd))

        # Constraints on U
        if self.u_min is not None:
            self.lb = np.tile(self.u_min, self.horizon)
        if self.u_max is not None:
            self.ub = np.tile(self.u_max, self.horizon)
            
        # Solve QP
        Uopt = solve_qp(sp.csc_matrix(self.H), self.g, G=sp.csc_matrix(self.A), h=self.b, \
            A=sp.csc_matrix(self.eqA), b=self.eqb, lb=self.lb, ub=self.ub, solver="osqp", initvals=self.solution)
        self.solution = Uopt
        
        Xopt = (A_big @ Xprev + B_big @ Uopt).reshape(self.horizon, self.n)
        
        # Convert predicted x sequence back to poses
        T = ini_pose
        poses = [T]
        for i in range(self.horizon):
            u_i = Uopt[self.n*i:self.n*(i+1)]
            T = T @ se3_exp(dlog @ u_i * self.dt)
            poses.append(T)
        return Uopt, Xopt, poses

# ------------------------
# Example
# ------------------------

def example():
    # Current pose: identity slightly translated/rotated
    cur_pos = np.array([0.4, 0.0, 0.4])
    cur_quat = np.array([0.0, 0.0, 0.0, 1.0])  # x,y,z,w
    T_ini = pose_to_matrix(cur_pos, cur_quat)
    p, q = matrix_to_pose(T_ini)
    print(f" Initial: pos = {p}, quat = {q}")

    # Target pose: rotated 45deg around z, translated
    tgt_pos = np.array([0.4, 0.0, 1.5])
    tgt_quat = R.from_euler('z', 45, degrees=True).as_quat()
    T_des = pose_to_matrix(tgt_pos, tgt_quat)
    p, q = matrix_to_pose(T_des)
    print(f" Desired: pos = {p}, quat = {q}")

    # run one MPC step
    lmpc_solver = LinearMPCController(horizon=100, dt=0.05, gamma = 0.001,
                                    u_min=np.array([-0.5, -0.5, -0.5, -1.0, -1.0, -1.0]),
                                    u_max=np.array([ 0.5,  0.5,  0.5,  1.0,  1.0,  1.0]))
    Uopt, Xopt, poses = lmpc_solver.solve(T_ini, T_des)
    
    print("\nPredicted poses:")
    for i, T in enumerate(poses):
        p, q = matrix_to_pose(T)
        print(f" step {i}: pos = {p}, quat = {q}")

# ------------------------
# Utils
# ------------------------

def skew(v):
    """Return 3x3 skew-symmetric matrix for vector v (3,)."""
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]], dtype=float)

def so3_left_jacobian(phi):
    """
    Left Jacobian J(φ) of SO(3) (3x3).
    phi: vector (3,) rotation vector (axis * angle)
    """
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * skew(phi) + (1.0/12.0) * (skew(phi) @ skew(phi))
    axis_hat = skew(phi / theta)
    J = (np.eye(3)
         + (1 - np.cos(theta)) / (theta**2) * axis_hat
         + (theta - np.sin(theta)) / (theta**3) * (axis_hat @ axis_hat))
    return J

def so3_left_jacobian_inv(phi):
    """Inverse of the left Jacobian J^{-1}(phi)."""
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        # series expansion
        return np.eye(3) - 0.5 * skew(phi) + (1.0/12.0) * (skew(phi) @ skew(phi))
    axis = phi / theta
    A = 0.5 * skew(axis)
    cot_term = (1 / theta - 0.5 / np.tan(theta / 2))
    return np.eye(3) - 0.5 * skew(phi) + cot_term * (skew(phi) @ skew(phi)) / (theta**2)

def se3_exp(xi):
    """
    Exponential map from se(3) (6-vector) to SE(3) homogeneous matrix (4x4).
    xi = [v (3,), omega (3,)] where omega is rotation vector (axis*angle).
    """
    v = xi[:3]
    omega = xi[3:]
    theta = np.linalg.norm(omega)
    Rm = R.from_rotvec(omega).as_matrix()
    if theta < 1e-8:
        J = np.eye(3) + 0.5 * skew(omega) + (1.0/6.0) * (skew(omega) @ skew(omega))
    else:
        J = so3_left_jacobian(omega)  # V matrix
    p = J @ v
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = p
    return T

def se3_log(T):
    """
    Log map from SE(3) (4x4 matrix) to se(3) 6-vector [v, omega].
    Uses: omega = log(R) (rotvec), v = J^{-1}(omega) * p
    """
    Rm = T[:3, :3]
    p = T[:3, 3]
    rot = R.from_matrix(Rm)
    omega = rot.as_rotvec()
    J_inv = so3_left_jacobian_inv(omega)
    v = J_inv @ p
    xi = np.concatenate([v, omega])
    return xi

def pose_to_matrix(position, quaternion):
    """Return 4x4 homogeneous matrix from position (3,) and quaternion (x,y,z,w)."""
    Rm = R.from_quat(quaternion).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = position
    return T

def matrix_to_pose(T):
    """Return (pos, quat) from 4x4 matrix. quat as (x,y,z,w)."""
    pos = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat

def compute_dlog_approx(xi):
    """
    Approximate dlog at xi (6,) by block-diagonal of J_inv(omega)
    where xi = [v, omega].
    This yields a 6x6 matrix approximating mapping such that x_{k+1} = x_k + dlog * u * dt
    """
    omega = xi[3:]
    J_inv = so3_left_jacobian_inv(omega)
    # Block diag: translational & rotational
    dlog = np.zeros((6,6))
    dlog[:3, :3] = J_inv
    dlog[3:, 3:] = J_inv
    return dlog

if __name__ == "__main__":
    example()
