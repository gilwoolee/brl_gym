import numpy as np
import os
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides
from mujoco_py import MjViewer
from gym import utils

# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)


class HopperEnv(MujocoEnv, utils.EzPickle):

    @autoargs.arg('alive_coeff', type=float,
                  help='reward coefficient for being alive')
    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            alive_coeff=1,
            ctrl_cost_coeff=0.01,
            *args, **kwargs):
        self.alive_coeff = alive_coeff
        self.ctrl_cost_coeff = ctrl_cost_coeff

        xml_string = kwargs.get("xml_string", "")
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_dir, 'assets/hopper.xml')

        MujocoEnv.__init__(self, model_path, frame_skip=5, xml_string=xml_string)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[0:1].flat,
            self.data.qpos[2:].flat,
            np.clip(self.data.qvel, -10, 10).flat,
            np.clip(self.data.qfrc_constraint, -10, 10).flat,
            self.data.body_xpos[0].flat,
        ])

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        vel = self.data.qvel[0]
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        reward = vel + self.alive_coeff - ctrl_cost
        state = self.state_vector()
        notdone = np.isfinite(state).all() and \
            (np.abs(state[3:]) < 100).all() and (state[0] > .7) and \
            (abs(state[2]) < .2)
        done = not notdone
        return next_obs, reward, done, dict(forward_reward=reward, ctrl_cost=ctrl_cost)

    def reset_model(self, randomize=True):
        nq = self.init_qpos.shape[0]
        nv = self.init_qvel.shape[0]
        if randomize:
            qpos = self.init_qpos + self.np_random.uniform(size=nq, low=-.1, high=.1)
            qvel = self.init_qvel + self.np_random.randn(nv) * .1
        else:
            qpos = self.init_qpos
            qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent*1.2

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
