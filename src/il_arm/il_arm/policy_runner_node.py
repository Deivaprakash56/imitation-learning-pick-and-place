"""Run a behaviour-cloning policy on a real (or simulated ROS) arm.

Mirrors demo_recorder: the observation is the same 'state' vector the recorder stored
(joint positions in `joint_names` order, optionally followed by the object position from
`object_pose_topic`), and the policy outputs the next joint targets (as recorded with
`action_is_next_state:=true` or from a commanded-joint topic).

Safety:
  * nothing moves until `~/start` is called; `~/stop` holds the current position
  * each command is clamped to `max_joint_step` rad from the current measured position
  * stops if joint_states (or the object pose, when used) go stale

  ros2 run il_arm policy_runner --ros-args -p policy:=runs/bc_real/best.pt \
      -p joint_names:="[joint1,joint2,joint3,joint4,joint5,joint6,gripper]" \
      -p command_topic:=/arm_controller/joint_trajectory -p object_pose_topic:=/object_pose
  ros2 service call /policy_runner/start std_srvs/srv/Trigger
"""
import threading

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from il_arm.bc_policy import BCPolicy, ChunkExecutor


class PolicyRunner(Node):
    def __init__(self):
        super().__init__('policy_runner')
        p = self.declare_parameter
        p('policy', '')
        p('joint_names', [''])
        p('joint_states_topic', '/joint_states')
        p('object_pose_topic', '')
        p('command_topic', '/arm_controller/joint_trajectory')
        p('rate_hz', 15.0)                 # must match the recorder's rate_hz
        p('max_joint_step', 0.05)          # rad per control step
        p('stale_after_s', 0.5)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.joints = [j for j in gp('joint_names') if j]
        if not gp('policy') or not self.joints:
            raise RuntimeError('set policy and joint_names')
        self.policy = BCPolicy.load(gp('policy'))
        self.exec = ChunkExecutor(self.policy)
        self.use_object = bool(gp('object_pose_topic'))
        expected = len(self.joints) + (3 if self.use_object else 0)
        if self.policy.obs_dim != expected:
            raise RuntimeError(f'policy expects {self.policy.obs_dim}-d obs, config gives {expected} '
                               f'({len(self.joints)} joints{" + object xyz" if self.use_object else ""})')
        self.period = 1.0 / float(gp('rate_hz'))
        self.max_step = float(gp('max_joint_step'))
        self.stale = float(gp('stale_after_s'))

        self._lock = threading.Lock()
        self._q = self._q_stamp = self._obj = self._obj_stamp = None
        self._running = False
        self.create_subscription(JointState, gp('joint_states_topic'), self._on_js, qos_profile_sensor_data)
        if self.use_object:
            self.create_subscription(PoseStamped, gp('object_pose_topic'), self._on_obj, 10)
        self.pub = self.create_publisher(JointTrajectory, gp('command_topic'), 10)
        self.create_service(Trigger, '~/start', self._start)
        self.create_service(Trigger, '~/stop', self._stop)
        self.create_timer(self.period, self._tick)
        self.get_logger().info(f'policy loaded ({self.policy.obs_dim}-d obs -> {self.policy.act_dim}-d action, '
                               f'chunk {self.policy.chunk}); call ~/start to run')

    def _on_js(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(j in idx for j in self.joints):
            with self._lock:
                self._q = np.array([msg.position[idx[j]] for j in self.joints], np.float32)
                self._q_stamp = self.get_clock().now()

    def _on_obj(self, msg):
        with self._lock:
            p = msg.pose.position
            self._obj = np.array([p.x, p.y, p.z], np.float32)
            self._obj_stamp = self.get_clock().now()

    def _age(self, stamp):
        return float('inf') if stamp is None else (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def _send(self, target):
        msg = JointTrajectory()
        msg.joint_names = self.joints
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in target]
        sec = self.period * 1.5
        pt.time_from_start = DurationMsg(sec=int(sec), nanosec=int((sec % 1) * 1e9))
        msg.points = [pt]
        self.pub.publish(msg)

    def _start(self, req, resp):
        self.exec.reset()
        self._running = True
        resp.success, resp.message = True, 'policy running'
        self.get_logger().info(resp.message)
        return resp

    def _stop(self, req, resp):
        self._running = False
        with self._lock:
            q = self._q
        if q is not None:
            self._send(q)        # hold position
        resp.success, resp.message = True, 'stopped, holding position'
        self.get_logger().info(resp.message)
        return resp

    def _tick(self):
        if not self._running:
            return
        with self._lock:
            q, q_age = self._q, self._age(self._q_stamp)
            obj, obj_age = self._obj, self._age(self._obj_stamp)
        if q is None or q_age > self.stale or (self.use_object and (obj is None or obj_age > self.stale)):
            self.get_logger().warn('stale joint_states / object pose - holding', throttle_duration_sec=1.0)
            if q is not None:
                self._send(q)
            return
        obs = np.concatenate([q, obj]) if self.use_object else q
        target = np.asarray(self.exec.act(obs), np.float32)
        target = q + np.clip(target - q, -self.max_step, self.max_step)
        self._send(target)


def main():
    rclpy.init()
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
