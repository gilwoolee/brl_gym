import numpy as np
import os
from os import path as osp
from gym import utils
from gym.spaces import Box, Discrete, Tuple
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer
import mujoco_py


class WamEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self.frame_skip = 1

        # Get asset dir
        dir_path = os.path.dirname(os.path.realpath(__file__))
        asset_path = osp.join(dir_path, "assets/wam","pusher_plane_cube_obstacle_2d.xml")
        self.asset_path = asset_path
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, asset_path, self.frame_skip)

        high = np.ones(7, dtype=np.float32) # x, y, theta
        self.action_space = Tuple([Box(low=-high, high=high), Discrete(1)]) # Discrete is for lift attempt

    def step(self, a):

        if isinstance(a, list):
            a = a[0]
            if a[1] == 1:
                # lift attempt
                pass
            else:
                self.do_simulation(a[0], self.frame_skip)
        else:
            self.do_simulation(a, self.frame_skip)
        done = False
        reward = -0.1 # Running cost

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        ob = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])
        return ob

    def reset(self):
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        return self._get_obs()

    def get_state(self):
        return self._get_obs()

    def set_state(self, qpos, qvel):
        state = self.sim.get_state()
        for i in range(len(qpos)):
            state.qpos[i] = qpos[i]
        for i in range(len(qvel)):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()


if __name__ == "__main__":
    env = WamEnv()

    o = env.reset()
    # import IPython; IPython.embed(); import sys; sys.exit(0)

    while True:
        o, _, _, _ = env.step([env.action_space.sample()])
        print(o)
        env.render()
