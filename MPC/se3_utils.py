import numpy as np
from scipy.spatial.transform import Rotation as R

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