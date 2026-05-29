#!/usr/bin/env python3
import math
import numpy as np
from dataclasses import dataclass, field
import cvxpy
from scipy.linalg import block_diag
from scipy.sparse import block_diag as sp_block_diag, csc_matrix, diags
from scipy.spatial import transform
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from mpc_pkg.utils import nearest_point


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = [acceleration, steering_speed]
    TK: int = 8  # finite time horizon length - kinematic

    # ---------------------------------------------------
    # Cost matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 5.0])
    )  # input cost [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 5.0])
    )  # input difference cost
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])  # levine sim
    )  # state error cost (for each step)
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])  # levine sim
    )  # final state error cost
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.2  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # minimum steering angle [rad]
    MAX_STEER: float = 0.4189   # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 14.0   # maximum speed [m/s]
    MIN_SPEED: float = 0.0    # minimum speed [m/s]
    MAX_ACCEL: float = 3.0    # maximum acceleration [m/s^2]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        # use the MPC as a tracker (similar to pure pursuit)
        self.is_real = False
        self.map_name = 'YasMarina_fast'

        # create ROS subscribers and publishers
        pose_topic = "/pf/viz/inferred_pose" if self.is_real else "/ego_racecar/odom"
        drive_topic = "/drive"
        vis_ref_traj_topic = "/ref_traj_marker"
        vis_waypoints_topic = "/waypoints_marker"
        vis_pred_path_topic = "/pred_path_marker"

        self.pose_sub = self.create_subscription(
            PoseStamped if self.is_real else Odometry,
            pose_topic,
            self.pose_callback,
            1,
        )
        self.pose_sub  # prevent unused variable warning

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.drive_msg = AckermannDriveStamped()
        # Initialize drive command to safe values
        self.drive_msg.drive.speed = 0.0
        self.drive_msg.drive.steering_angle = 0.0

        self.vis_waypoints_pub = self.create_publisher(Marker, vis_waypoints_topic, 1)
        self.vis_waypoints_msg = Marker()
        self.vis_ref_traj_pub = self.create_publisher(Marker, vis_ref_traj_topic, 1)
        self.vis_ref_traj_msg = Marker()
        self.vis_pred_path_pub = self.create_publisher(Marker, vis_pred_path_topic, 1)
        self.vis_pred_path_msg = Marker()

        map_path = os.path.abspath(os.path.join('src/mpc_ws/src', 'csv_data'))
        self.waypoints = np.loadtxt(
            map_path + '/' + self.map_name + '.csv',
            delimiter=';',
            skiprows=0
        )  # csv data

        if self.map_name == 'YasMarina_fast':
            self.waypoints[:, 3] += math.pi / 2
        self.visualize_waypoints_in_rviz()

        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # initialize MPC problem
        self.mpc_prob_init()

        # init state - avoid unknown variables for scan callback
        self.curr_pos = np.array([0.0, 0.0, 0.0])
        self.rot_mat = np.identity(3)

    def pose_callback(self, pose_msg):
        # extract pose from ROS msg
        self.update_rotation_matrix(pose_msg)
        vehicle_state = self.update_vehicle_state(pose_msg)

        if self.is_real:
            vehicle_state.v = -1 * vehicle_state.v  # negate the monitoring speed

        # 🔹 Clip current speed into [MIN_SPEED, MAX_SPEED]
        vehicle_state.v = np.clip(
            vehicle_state.v,
            self.config.MIN_SPEED,
            self.config.MAX_SPEED,
        )

        # Calculate the next reference trajectory for the next T steps
        # ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        ref_path = self.calc_ref_trajectory(
            vehicle_state,
            self.waypoints[:, 1],
            self.waypoints[:, 2],
            self.waypoints[:, 3],
            self.waypoints[:, 5] * 1.172,
        )
        self.visualize_ref_traj_in_rviz(ref_path)
        
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # solve the MPC control problem
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)

        # publish drive message.
        steer_output = self.odelta_v[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

        # 🔹 Clip command speed into [MIN_SPEED, MAX_SPEED]
        speed_output = np.clip(
            speed_output,
            self.config.MIN_SPEED,
            self.config.MAX_SPEED,
        )

        self.drive_msg.drive.steering_angle = steer_output
        self.drive_msg.drive.speed = (-1.0 if self.is_real else 1.0) * speed_output
        self.drive_pub.publish(self.drive_msg)
        print(
            "steering ={}, speed ={}".format(
                self.drive_msg.drive.steering_angle,
                self.drive_msg.drive.speed,
            )
        )

        self.vis_waypoints_pub.publish(self.vis_waypoints_msg)

    # toolkits
    def update_rotation_matrix(self, pose_msg):
        # get rotation matrix from the car frame to the world frame
        curr_orien = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        quat = [curr_orien.x, curr_orien.y, curr_orien.z, curr_orien.w]
        self.rot_mat = (transform.Rotation.from_quat(quat)).as_matrix()
        # print("rotation matrix = {}".format(self.rot_mat))

    def update_vehicle_state(self, pose_msg):
        """
        written by Derek, not from the template, != update state
        """
        vehicle_state = State()
        vehicle_state.x = pose_msg.pose.position.x if self.is_real else pose_msg.pose.pose.position.x
        vehicle_state.y = pose_msg.pose.position.y if self.is_real else pose_msg.pose.pose.position.y
        # Use last commanded speed as current speed estimate
        vehicle_state.v = self.drive_msg.drive.speed

        curr_orien = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        q = [curr_orien.x, curr_orien.y, curr_orien.z, curr_orien.w]
        vehicle_state.yaw = math.atan2(
            2 * (q[3] * q[2] + q[0] * q[1]),
            1 - 2 * (q[1] ** 2 + q[2] ** 2),
        )
        # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternion_%E2%86%92_Euler_angles_(z-y%E2%80%B2-x%E2%80%B3_intrinsic)

        return vehicle_state

    # mpc functions
    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)  # 4 x 9
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)  # 2 x 8
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))  # 4
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))  # 4 x 9
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = sp_block_diag(tuple([self.config.Rk] * self.config.TK))  # (2*8)x(2*8)

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = sp_block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))  # (2*7)x(2*7)

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = sp_block_diag(tuple(Q_block))  # (4*9)x(4*9)

        # Objective part 1: Influence of the control inputs
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

        # Objective part 2: Deviation from reference trajectory (includes final step)
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # Objective part 3: Smooth changes in control inputs
        objective += cvxpy.quad_form(
            cvxpy.vec(cvxpy.diff(self.uk, axis=1)),
            Rd_block
        )

        # Constraints: Vehicle dynamics over the horizon
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))  # 4 x 9
        for t in range(self.config.TK):  # 8
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0  # reference steering angle is zero
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = sp_block_diag(tuple(A_block))  # 32 x 32
        B_block = sp_block_diag(tuple(B_block))  # 32 x 16
        C_block = np.array(C_block)  # 32 x 1

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        m, n = A_block.shape  # 32, 32
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        self.Annz_k.value = A_block.data
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A for B
        m, n = B_block.shape  # 32, 16
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))

        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # C as dense parameter
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # Dynamics constraints (flattened)
        flatten_prev_xk = cvxpy.vec(self.xk[:, :-1])
        flatten_next_xk = cvxpy.vec(self.xk[:, 1:])
        c1 = flatten_next_xk == self.Ak_ @ flatten_prev_xk + self.Bk_ @ cvxpy.vec(self.uk) + self.Ck_
        constraints.append(c1)
        
        # Steering rate constraints
        dsteering = cvxpy.diff(self.uk[1, :])
        c2_lower = -self.config.MAX_DSTEER * self.config.DTK <= dsteering
        c2_upper = dsteering <= self.config.MAX_DSTEER * self.config.DTK
        constraints.append(c2_lower)
        constraints.append(c2_upper)
        
        # Initial state constraint
        c3 = self.xk[:, 0] == self.x0k
        constraints.append(c3)

        # State constraints (speed bounds)
        speed = self.xk[2, :]
        c4_lower = self.config.MIN_SPEED <= speed
        c4_upper = speed <= self.config.MAX_SPEED
        constraints.append(c4_lower)
        constraints.append(c4_upper)

        # Input constraints: steering bounds
        steering = self.uk[1, :]
        c5_lower = self.config.MIN_STEER <= steering
        c5_upper = steering <= self.config.MAX_STEER
        constraints.append(c5_lower)
        constraints.append(c5_upper)

        # Input constraints: acceleration upper bound
        acc = self.uk[0, :]
        c6 = acc <= self.config.MAX_ACCEL
        constraints.append(c6)

        # Create the optimization problem in CVXPY and setup the workspace
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc reference trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        """
        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(
            np.array([state.x, state.y]),
            np.array([cx, cy]).T
        )

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        if dind < 1.0:
            dind = 1.0
            
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)),
            0,
            0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]

        angle_thres = 4.5

        for i in range(len(cyaw)):
            if cyaw[i] - state.yaw > angle_thres:
                cyaw[i] -= 2 * np.pi
            if state.yaw - cyaw[i] > angle_thres:
                cyaw[i] += 2 * np.pi

        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= self.config.MIN_STEER:
            delta = self.config.MIN_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: X_{k+1} = A X_k + B u_k + C
        State vector: x=[x, y, v, yaw]
        """
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = sp_block_diag(tuple(A_block))
        B_block = sp_block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        try:
            self.MPC_prob.solve(
                solver=cvxpy.OSQP,
                verbose=False,
                warm_start=True
            )
        except cvxpy.error.SolverError as e:
            print(f"Error: MPC solver failed: {e}")
            oa = odelta = ox = oy = oyaw = ov = None
            return oa, odelta, ox, oy, oyaw, ov

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()
        else:
            print(f"Error: Cannot solve mpc.. Status: {self.MPC_prob.status}")
            oa = odelta = ox = oy = oyaw = ov = None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC control with updating operational point iteratively
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Predict the vehicle motion for T steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        self.visualize_pred_path_in_rviz(path_predict)

        # Run the MPC optimization
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        # If solve failed, fall back to previous controls / predicted path
        if mpc_a is None or mpc_delta is None:
            return (
                np.array(oa),
                np.array(od),
                path_predict[0, :],
                path_predict[1, :],
                path_predict[3, :],
                path_predict[2, :],
                path_predict,
            )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    # visualization
    def visualize_waypoints_in_rviz(self):
        self.vis_waypoints_msg.points = []
        self.vis_waypoints_msg.header.frame_id = '/map'
        self.vis_waypoints_msg.type = Marker.POINTS
        self.vis_waypoints_msg.color.g = 0.75
        self.vis_waypoints_msg.color.a = 1.0
        self.vis_waypoints_msg.scale.x = 0.05
        self.vis_waypoints_msg.scale.y = 0.05
        self.vis_waypoints_msg.id = 0
        for i in range(self.waypoints.shape[0]):
            point = Point(
                x=self.waypoints[i, 1],
                y=self.waypoints[i, 2],
                z=0.1
            )
            self.vis_waypoints_msg.points.append(point)
        
        # self.vis_waypoints_pub.publish(self.vis_waypoints_msg)

    def visualize_ref_traj_in_rviz(self, ref_traj):
        # visualize the path data in the world frame
        self.vis_ref_traj_msg.points = []
        self.vis_ref_traj_msg.header.frame_id = '/map'
        self.vis_ref_traj_msg.type = Marker.LINE_STRIP
        self.vis_ref_traj_msg.color.b = 0.75
        self.vis_ref_traj_msg.color.a = 1.0
        self.vis_ref_traj_msg.scale.x = 0.08
        self.vis_ref_traj_msg.scale.y = 0.08
        self.vis_ref_traj_msg.id = 0
        for i in range(ref_traj.shape[1]):
            point = Point(
                x=ref_traj[0, i],
                y=ref_traj[1, i],
                z=0.2
            )
            self.vis_ref_traj_msg.points.append(point)
        
        self.vis_ref_traj_pub.publish(self.vis_ref_traj_msg)

    def visualize_pred_path_in_rviz(self, path_predict):
        # visualize the path data in the world frame
        self.vis_pred_path_msg.points = []
        self.vis_pred_path_msg.header.frame_id = '/map'
        self.vis_pred_path_msg.type = Marker.LINE_STRIP
        self.vis_pred_path_msg.color.r = 0.75
        self.vis_pred_path_msg.color.a = 1.0
        self.vis_pred_path_msg.scale.x = 0.08
        self.vis_pred_path_msg.scale.y = 0.08
        self.vis_pred_path_msg.id = 0
        for i in range(path_predict.shape[1]):
            point = Point(
                x=path_predict[0, i],
                y=path_predict[1, i],
                z=0.2
            )
            self.vis_pred_path_msg.points.append(point)
        
        self.vis_pred_path_pub.publish(self.vis_pred_path_msg)


def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
