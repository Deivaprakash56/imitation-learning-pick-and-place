"""Simulated arm (Gymnasium-Robotics FetchPickAndPlace, MuJoCo) + a scripted expert.

Observation used by the policy (31-d): Fetch `observation` (25) + `desired_goal` (3) +
`desired_goal - object_pos` (3). Action (4): end-effector dx, dy, dz in [-1,1] (x0.05 m) + gripper
(+1 open, -1 close).
"""
import gymnasium as gym
import numpy as np

ENV_ID = 'FetchPickAndPlace-v4'


def make_env(render_mode=None, max_episode_steps=50):
    import gymnasium_robotics  # noqa: F401
    gym.register_envs(gymnasium_robotics)
    return gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=max_episode_steps)


def flatten_obs(o: dict) -> np.ndarray:
    obs = o['observation']
    obj = obs[3:6]
    return np.concatenate([obs, o['desired_goal'], o['desired_goal'] - obj]).astype(np.float32)


class ScriptedExpert:
    """Approach -> descend -> grasp -> carry to goal. ~95% success on FetchPickAndPlace."""

    def __init__(self, gain=10.0):
        self.gain = gain
        self.reset()

    def reset(self):
        self.phase = 0
        self.grasp_steps = 0

    def act(self, o: dict) -> np.ndarray:
        obs = o['observation']
        grip, obj = obs[0:3], obs[3:6]
        goal = o['desired_goal']
        a = np.zeros(4, dtype=np.float32)

        if self.phase == 0:      # hover above the object
            target = obj + np.array([0, 0, 0.06])
            a[3] = 1.0
            if np.linalg.norm((target - grip)[:2]) < 0.008 and abs(target[2] - grip[2]) < 0.015:
                self.phase = 1
        if self.phase == 1:      # descend
            target = obj + np.array([0, 0, -0.005])
            a[3] = 1.0
            if abs(grip[2] - target[2]) < 0.01:
                self.phase = 2
        if self.phase == 2:      # close gripper
            target = obj
            a[3] = -1.0
            self.grasp_steps += 1
            if self.grasp_steps >= 4:
                self.phase = 3
        if self.phase == 3:      # carry to goal, keep closed
            target = goal
            a[3] = -1.0
            if np.linalg.norm(obj - grip) > 0.05:   # dropped it -> start over
                self.reset()
        a[:3] = np.clip(self.gain * (target - grip), -1, 1)
        return a
