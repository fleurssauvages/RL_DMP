import numpy as np
from qpsolvers import solve_qp
import scipy.sparse as sp
import spatialmath as sm

class QPController:
    def __init__(self, robot, dt=0.05):
        self.n_dof = robot.n
        self.robot = robot
        self.joint_positions = robot.q
        self.joint_velocities = robot.qd
        self.joints_limits = np.vstack([robot.qlim[0, :] + 0.01,   # lower limits
                                        robot.qlim[1, :] - 0.01]) # Add some margin to joint limits
        self.joints_velocities_limits = robot.qdlim
        self.H = np.eye(robot.n)  # Hessian
        self.g = np.zeros(robot.n)  # Gradient
        self.A = np.zeros((0, robot.n))  # Inequality constraints
        self.b = np.zeros(0)  # Inequality constraint bounds
        self.eqA = np.zeros((0, robot.n))  # Equality constraints
        self.eqb = np.zeros(0) # Equality constraint bounds
        self.lb = -np.ones(robot.n) * np.inf  # Lower bounds
        self.ub = np.ones(robot.n) * np.inf   # Upper bounds
        self.dt = dt # Time step of the controller loop / simulation
        self.solution = None

    def solve(self, xdot, alpha=0.02, beta=0.01, W = np.diag([1.0, 1.0, 2.0, 0.1, 0.1, 0.1])):
        """
        Solve the quadratic programming problem using previous solution as initial value
        Minimize the cost function ||J qdot - xdot||^2 + alpha ||N qdot||^2 - beta * manipulability_gradient * qdot
        where N is the nullspace projector of J
        xdot the desired end-effector velocity (6D vector)
        alpha the weight on the secondary task (minimize joint velocities)
        beta the weight on maximizing manipulability
        The weight matrix W can be used to prioritize translation over rotation or vice-versa,
        if translations are prioritize, set higher values on the first 3 diagonal elements
        """
        self.update_IK_problem(xdot, alpha=alpha, beta=beta, W = W)
        self.add_floor_constraint(z_floor=0.0, margin=0.02)
        self.update_joints_limits(self.joints_limits)
        x = solve_qp(sp.csc_matrix(self.H), self.g, G=sp.csc_matrix(self.A), h=self.b, A=sp.csc_matrix(self.eqA), b=self.eqb, lb=self.lb, ub=self.ub, solver="osqp", initvals=self.solution)
        self.solution = x
        self.reset_constraints()
        pass
    
    def update_robot_state(self, robot):
        self.robot = robot
        self.joint_positions = robot.q
        self.joint_velocities = robot.qd
        pass
    
    def update_IK_problem(self, xdot, alpha=0.02, beta=0.01, W=np.eye(6), damping=1e-6):
        """
        Update the IK problem parameters based on desired end-effector velocity (6D vector) and current joint positions
        xdot: np.array of shape (6,)
        joint_pos: np.array of shape (n_dof,)
        The cost-function solved is ||J qdot - xdot||^2 + alpha ||N qdot||^2
        Nullspace solved with secondary task of minimizing joint velocities and keeping elbow at 0 position (gain alpha)
        And maximizing manipulability (gain beta)
        The weight matrix W can be used to prioritize translation over rotation or vice-versa
        """
        def weighted_damped_pinv(J, W, damping=1e-6):
            # Solve min || W (J dq - xdot) ||^2  => dq = (J^T W^T W J + damping I)^{-1} J^T W^T W xdot
            JT_WTW_J = J.T @ (W.T @ W) @ J
            inv = np.linalg.inv(JT_WTW_J + damping * np.eye(J.shape[1]))
            J_pinv_w = inv @ J.T @ (W.T @ W)
            return J_pinv_w

        I = np.eye(self.n_dof)
        J = self.robot.jacobe(self.joint_positions)
        Jpinv = weighted_damped_pinv(J, W, damping=damping) # Damped pseudo-inverse of J with weight W, damping to avoid singularities
        
        N = I - Jpinv @ J
        qdot_des = np.zeros(self.n_dof)
        qdot_des[1] = -self.joint_positions[1] / self.dt # Secondary task to keep elbow at 0 position
        
        def manipulability_index(q):
            J = self.robot.jacobe(q)
            return np.sqrt(np.linalg.det(J @ J.T))

        def manipulability_gradient(q, delta=1e-1):
            w0 = manipulability_index(q)
            grad = np.zeros_like(q)
            for i in range(len(q)):
                dq = np.zeros_like(q)
                dq[i] = delta
                Jqdq = self.robot.jacobe(q + dq)
                w1 = manipulability_index(Jqdq)
                grad[i] = (w1 - w0) / delta
            return grad
        
        self.g = -2 * xdot.T @ (W.T @ W) @ J - 2 * qdot_des.T @ (N.T @ N) * alpha - 2 * manipulability_gradient(self.joint_positions).T @ (N.T @ N) * beta
        self.H = 2 * (N.T @ N * alpha + J.T @ (W.T @ W) @ J)
        pass
        
    def add_constraint(self, A, b):
        """
        Add position constraints to the QP problem
        A: np.array of shape (m, n_dof)
        b: np.array of shape (m,)
        """
        self.A = np.vstack((self.A, A))
        self.b = np.hstack((self.b, b))
        pass
    
    def reset_constraints(self):
        """
        Reset all position constraints
        """
        self.A = np.zeros((0, self.n_dof))
        self.b = np.zeros(0)
        pass
    
    def update_joints_limits(self, limits):
        """
        Add joint limits constraints to the QP problem
        limits: np.array of shape (2, n_dof) with min and max limits for each joint
        The joint limits are converted to velocity limits based on current position and dt
        """
        self.lb = 0.1*(limits[0, :] - self.joint_positions) / self.dt
        self.ub = 0.1*(limits[1, :] - self.joint_positions) / self.dt
        self.lb = np.maximum(self.lb, -self.joints_velocities_limits[0:self.n_dof])
        self.ub = np.minimum(self.ub, self.joints_velocities_limits[0:self.n_dof])
        pass
    
    def add_floor_constraint(self, z_floor=0.0, margin=0.02):
        """ Add a constraint to avoid the end-effector going below z_floor + margin in the next time step """
        z = self.robot.fkine(self.joint_positions).t[2]
        
        J = self.robot.jacobe(self.joint_positions)  # expected shape (6, n)
        Jz = np.atleast_2d(np.asarray(J[2, :], dtype=float))  # shape (1, n)

        required = (z_floor + margin - z) / float(self.dt)  # scalar

        G = Jz
        h = np.array([-required], dtype=float)

        self.add_constraint(G, h)
        pass

    def add_local_tangent_plane_constraints(self, obstacles, margin = 0.05):
        """
        For each spherical obstacle, create a local tangent plane toward each
        considered point/joint and add a constraint preventing that joint from crossing the plane.
        A similar technique can be found in
        A Quadratic Programming Approach to Manipulation in Real-Time Using Modular Robots, Chao Liu and Mark Yim, 2021

        Parameters
        ----------
        obstacles : list of dict
            [{'center': np.array(3), 'radius': float}, ...]
        """
        
        franka_contact_points = [
            # --- fingertips when gripper closed ---
            np.array([0.00, 0.00, 0.0]),   # center tips

            # # # --- palm corners (front face, outer square) ---
            np.array([+0.0, -0.12, -0.08]), # top-right
            np.array([+0.0, 0.12, -0.08]), # top-left
            np.array([+0.0, -0.12, -0.04]), # bottom-right
            np.array([+0.0, 0.12, -0.04]), # bottom-left
        ]
        
        for obs in obstacles:
            sphere_center = np.array(obs['center'], dtype=float)
            sphere_radius = float(obs['radius']) + abs(margin)
                        
            T_ee = self.robot.fkine(self.joint_positions)
            R_ee, p_ee = T_ee.R, T_ee.t
            J = self.robot.jacob0(self.joint_positions)
            Jv = J[0:3, :]     # linear part
            Jw = J[3:6, :]     # angular part
            
            for p_local in franka_contact_points:
                # Global position of the contact point
                p_world = p_ee + R_ee @ p_local

                # Spatial Jacobian at this point: Jp = Jv + ω×r
                # The linear velocity part for an offset point is:
                # v_p = v + ω × r  => Jv_p = Jv + [ω]× * Jw = Jv - skew(r) * Jw
                def skew(v):
                    return np.array([
                        [0, -v[2],  v[1]],
                        [v[2], 0,  -v[0]],
                        [-v[1], v[0],  0]
                    ])

                r = p_world - p_ee
                Jp = Jv - skew(r) @ Jw

                # Direction vector from point to obstacle center
                vec = sphere_center - p_world
                dist = np.linalg.norm(vec)
                if dist < 1e-9:
                    s = np.array([1.0, 0.0, 0.0])
                else:
                    s = vec / dist
                    
                # Tangent plane: obstacle surface offset
                o_prime = sphere_center - sphere_radius * s

                # Linear constraint: (s^T * Jp + (p × s)^T * Jw) * qdot <= ||o' - p|| / dt
                A_row = s.T @ Jp  # shape (n_dof,)
                b_scalar = (dist - sphere_radius) / (self.dt * 2)

                self.add_constraint(A_row, b_scalar)