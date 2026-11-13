"""A tiny stand-in for a real arm, to test the ROS recording -> training -> deployment path.

Simulates 3 revolute joints + a gripper (velocity-limited tracking of the latest command),
an object at a random position, and an optional scripted "teleoperator".

Topics
  out: /joint_states (50 Hz), /object_pose (10 Hz)
  in : /arm_controller/joint_trajectory   (what policy_runner publishes)
  out: /teleop/joint_command               (what demo_recorder records as the action, teleop mode)
Services (std_srvs/Trigger)
  ~/reset        new random object, arm back home
  ~/teleop       run one scripted teleop episode (reach object, close gripper, lift)
  ~/check        success? = gripper closed while within `tol` rad of the grasp configuration

The "grasp configuration" for an object at (x, y, z) is a smooth made-up inverse kinematics:
  q* = [atan2(y, x), 1.2 * r - 0.3, -0.8 + 1.5 * z]       r = |(x, y)|
so a policy can only succeed if it actually uses the object pose.
"""
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

JOINTS = ['joint1', 'joint2', 'joint3', 'gripper']
HOME = np.array([0.0, 0.0, 0.0, 0.04], np.float32)      # gripper 0.04 = open, 0.0 = closed
OPEN, CLOSED = 0.04, 0.0


def grasp_config(obj):
    x, y, z = obj
    return np.array([math.atan2(y, x), 1.2 * math.hypot(x, y) - 0.3, -0.8 + 1.5 * z], np.float32)


class FakeArm(Node):
    def __init__(self):
        super().__init__('fake_arm')
        self.declare_parameter('max_vel', 1.0)       # rad/s per joint
        self.declare_parameter('tol', 0.08)          # rad, success tolerance
        self.declare_parameter('seed', 0)
        # 'state': teleop reacts to the arm's state (default). 'timer': fixed-duration waypoints,
        # kept to reproduce the README's "inconsistent demonstrations" failure.
        self.declare_parameter('teleop_mode', 'state')
        self.teleop_mode = self.get_parameter('teleop_mode').value
        self.max_vel = self.get_parameter('max_vel').value
        self.tol = self.get_parameter('tol').value
        self.rng = np.random.default_rng(self.get_parameter('seed').value)
        self.lock = threading.Lock()
        self.q = HOME.copy()
        self.target = HOME.copy()
        self.obj = self._random_obj()
        self.success = False
        cb = ReentrantCallbackGroup()
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.obj_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        self.teleop_pub = self.create_publisher(JointState, '/teleop/joint_command', 10)
        self.create_subscription(JointTrajectory, '/arm_controller/joint_trajectory', self._on_cmd, 10)
        self.create_service(Trigger, '~/reset', self._reset, callback_group=cb)
        self.create_service(Trigger, '~/teleop', self._teleop, callback_group=cb)
        self.create_service(Trigger, '~/check', self._check, callback_group=cb)
        self.dt = 0.02
        self.create_timer(self.dt, self._step)
        self.create_timer(0.1, self._pub_obj)

    def _random_obj(self):
        return np.array([self.rng.uniform(0.3, 0.6), self.rng.uniform(-0.3, 0.3), self.rng.uniform(0.0, 0.2)],
                        np.float32)

    # simulation -------------------------------------------------------------------
    def _step(self):
        with self.lock:
            d = np.clip(self.target - self.q, -self.max_vel * self.dt, self.max_vel * self.dt)
            d[3] = np.clip(self.target[3] - self.q[3], -0.2 * self.dt, 0.2 * self.dt)   # gripper speed
            self.q = self.q + d
            # success latch: gripper closed while at the grasp configuration
            if self.q[3] < 0.01 and np.max(np.abs(self.q[:3] - grasp_config(self.obj))) < self.tol:
                self.success = True
            q = self.q.copy()
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = JOINTS
        m.position = [float(v) for v in q]
        self.js_pub.publish(m)

    def _pub_obj(self):
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        with self.lock:
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in self.obj)
        m.pose.orientation.w = 1.0
        self.obj_pub.publish(m)

    def _on_cmd(self, msg: JointTrajectory):
        if not msg.points:
            return
        idx = {n: i for i, n in enumerate(msg.joint_names)}
        if all(j in idx for j in JOINTS):
            with self.lock:
                self.target = np.array([msg.points[0].positions[idx[j]] for j in JOINTS], np.float32)

    # services ----------------------------------------------------------------------
    def _reset(self, req, resp):
        with self.lock:
            self.q = HOME.copy()
            self.target = HOME.copy()
            self.obj = self._random_obj()
            self.success = False
        resp.success, resp.message = True, f'object at {np.round(self.obj, 3).tolist()}'
        return resp

    def _teleop(self, req, resp):
        if self.teleop_mode == 'timer':
            return self._teleop_timer(resp)
        return self._teleop_state(resp)

    def _teleop_timer(self, resp):
        """Clock-driven operator: each waypoint is held for a fixed time, whatever the arm is doing."""
        with self.lock:
            g = grasp_config(self.obj)
        above = g + np.array([0.0, 0.0, 0.3], np.float32)
        for wp, dur in [(np.r_[above, OPEN], 1.4), (np.r_[g, OPEN], 0.8), (np.r_[g, CLOSED], 0.5),
                        (np.r_[above, CLOSED], 0.8)]:
            t_end = time.monotonic() + dur
            while time.monotonic() < t_end:
                with self.lock:
                    self.target = wp.astype(np.float32)
                m = JointState()
                m.header.stamp = self.get_clock().now().to_msg()
                m.name = JOINTS
                m.position = [float(v) for v in wp]
                self.teleop_pub.publish(m)
                time.sleep(1 / 15)
        resp.success, resp.message = True, f'teleop done, success={self.success}'
        return resp

    def _teleop_state(self, resp):
        """Scripted operator: reach above the grasp, descend, close the gripper, lift.

        Each command is chosen from the arm's *current state* (like a human watching the arm),
        not from a clock. A clock-driven script gives the same observation different labels
        (arm hovering: "stay" for a while, then "descend"), which behaviour cloning averages
        into a policy that hovers forever - see the README.
        """
        with self.lock:
            g = grasp_config(self.obj)
        above = g + np.array([0.0, 0.0, 0.3], np.float32)
        near = lambda a, b: np.max(np.abs(a - b)) < 0.03  # noqa: E731
        t_end = time.monotonic() + 8.0
        lifted = False
        while time.monotonic() < t_end and not lifted:
            with self.lock:
                q = self.q.copy()
            aligned = np.max(np.abs(q[:2] - g[:2])) < 0.03          # joints 1-2 over the object
            if q[3] < 0.005:                                         # holding -> lift
                tgt = np.r_[above, CLOSED]
                lifted = near(q[:3], above)
            elif near(q[:3], g):                                     # at the grasp -> close
                tgt = np.r_[g, CLOSED]
            elif aligned and q[2] <= above[2] + 0.03:                # above / descending -> go down
                tgt = np.r_[g, OPEN]
            else:                                                    # otherwise -> move above it
                tgt = np.r_[above, OPEN]
            with self.lock:
                self.target = tgt.astype(np.float32)
            m = JointState()
            m.header.stamp = self.get_clock().now().to_msg()
            m.name = JOINTS
            m.position = [float(v) for v in tgt]
            self.teleop_pub.publish(m)
            time.sleep(1 / 15)
        resp.success, resp.message = True, f'teleop done, success={self.success}'
        return resp

    def _check(self, req, resp):
        with self.lock:
            err = float(np.max(np.abs(self.q[:3] - grasp_config(self.obj))))
            resp.success = bool(self.success)
        resp.message = f'success={resp.success} final_err={err:.3f}'
        return resp


def main():
    rclpy.init()
    node = FakeArm()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
