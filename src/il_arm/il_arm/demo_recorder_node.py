"""Record teleoperated demonstrations on a real (or Gazebo/MoveIt) arm into the shared HDF5 format.

Arm-agnostic: it samples the latest messages at a fixed rate.
  state  <- sensor_msgs/JointState on `joint_states_topic` (positions of `joint_names`, in order)
  action <- sensor_msgs/JointState on `action_topic` - the *commanded* joint targets from your
            teleop (leader arm, SpaceMouse+IK, MoveIt Servo output converted to joints, ...).
            If `action_is_next_state` is true, no action topic is needed: action[t] = state[t+1].
  image  <- optional sensor_msgs/Image on `image_topic`, resized to image_size.
  object <- optional geometry_msgs/PoseStamped on `object_pose_topic` (AprilTag, detector, mocap).
            Its xyz is appended to the state vector, so the policy knows where the object is;
            policy_runner reads the same topic at run time.

Services (std_srvs/Trigger):
  ~/start          begin an episode
  ~/stop_success   end it and save as successful
  ~/stop_fail      end it and save as failed (kept, but skipped by default when training)
  ~/discard        end it without saving

  ros2 run il_arm demo_recorder --ros-args -p out:=data/real_pick.hdf5 \
      -p joint_names:="[joint1,joint2,joint3,joint4,joint5,joint6,gripper]" -p action_is_next_state:=true
  ros2 service call /demo_recorder/start std_srvs/srv/Trigger
"""
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger

from il_arm.dataset import DemoWriter


class DemoRecorder(Node):
    def __init__(self):
        super().__init__('demo_recorder')
        p = self.declare_parameter
        p('out', 'data/real_demos.hdf5')
        p('env_name', 'real_arm')
        p('joint_names', [''])
        p('joint_states_topic', '/joint_states')
        p('action_topic', '/teleop/joint_command')
        p('action_is_next_state', False)
        p('image_topic', '')
        p('object_pose_topic', '')
        p('image_size', [128, 128])
        p('rate_hz', 15.0)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.joints = [j for j in gp('joint_names') if j]
        if not self.joints:
            raise RuntimeError('set joint_names (ordered list, include the gripper joint)')
        self.next_state = bool(gp('action_is_next_state'))
        self.img_size = tuple(gp('image_size'))
        self.writer = DemoWriter(gp('out'), gp('env_name'))
        self._lock = threading.Lock()
        self._state = self._action = self._image = self._obj = None
        self._recording = False
        self._buf = None

        self.create_subscription(JointState, gp('joint_states_topic'), self._on_state, qos_profile_sensor_data)
        if not self.next_state:
            self.create_subscription(JointState, gp('action_topic'), self._on_action, 10)
        self.use_object = bool(gp('object_pose_topic'))
        if self.use_object:
            from geometry_msgs.msg import PoseStamped
            self.create_subscription(PoseStamped, gp('object_pose_topic'), self._on_obj, 10)
        if gp('image_topic'):
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.create_subscription(Image, gp('image_topic'), self._on_image, qos_profile_sensor_data)
        for name, fn in (('start', self._start), ('stop_success', lambda r, s: self._stop(s, True)),
                         ('stop_fail', lambda r, s: self._stop(s, False)), ('discard', self._discard)):
            self.create_service(Trigger, f'~/{name}', fn)
        self.create_timer(1.0 / float(gp('rate_hz')), self._tick)
        self.get_logger().info(f'recorder ready, {self.writer.n} demos already in {gp("out")}')

    def _extract(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in idx for j in self.joints):
            return None
        return np.array([msg.position[idx[j]] for j in self.joints], np.float32)

    def _on_state(self, msg):
        v = self._extract(msg)
        if v is not None:
            with self._lock:
                self._state = v

    def _on_obj(self, msg):
        p = msg.pose.position
        with self._lock:
            self._obj = np.array([p.x, p.y, p.z], np.float32)

    def _on_action(self, msg):
        v = self._extract(msg)
        if v is not None:
            with self._lock:
                self._action = v

    def _on_image(self, msg):
        import cv2
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        with self._lock:
            self._image = img

    def _tick(self):
        if not self._recording:
            return
        with self._lock:
            s, a, im, obj = self._state, self._action, self._image, self._obj
        if s is None or (a is None and not self.next_state) or (self.use_object and obj is None):
            self.get_logger().warn('waiting for state/action/object messages', throttle_duration_sec=2.0)
            return
        self._buf['state'].append(np.concatenate([s, obj]) if self.use_object else s)
        self._buf['joints'].append(s)
        if not self.next_state:
            self._buf['action'].append(a)
        if im is not None:
            self._buf['image'].append(im)

    def _start(self, req, resp):
        if self._recording:
            resp.success, resp.message = False, 'already recording'
            return resp
        self._buf = {'state': [], 'joints': [], 'action': [], 'image': []}
        self._recording = True
        resp.success, resp.message = True, f'recording demo_{self.writer.n}'
        self.get_logger().info(resp.message)
        return resp

    def _stop(self, resp, success):
        if not self._recording:
            resp.success, resp.message = False, 'not recording'
            return resp
        self._recording = False
        b = self._buf
        states = np.stack(b['state']) if b['state'] else None
        if states is None or len(states) < 5:
            resp.success, resp.message = False, 'episode too short, discarded'
            return resp
        if self.next_state:
            joints = np.stack(b['joints'])                   # action = next joint positions (not object)
            actions = np.concatenate([joints[1:], joints[-1:]])
        else:
            actions = np.stack(b['action'])
        obs = {'state': states}
        if len(b['image']) == len(states):
            obs['image'] = np.stack(b['image'])
        self.writer.add(obs, actions, success, joint_names=','.join(self.joints))
        resp.success = True
        resp.message = f'saved demo_{self.writer.n - 1}: {len(states)} steps, success={success}'
        self.get_logger().info(resp.message)
        return resp

    def _discard(self, req, resp):
        self._recording = False
        resp.success, resp.message = True, 'discarded'
        return resp

    def destroy_node(self):
        self.writer.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = DemoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
