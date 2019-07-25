import os
import numpy as np
from gym import utils
from mujoco_py import MjViewer
from rllab import spaces
from rllab.envs.mujoco import MujocoEnv
from rllab.envs.mujoco.model_updater import MujocoUpdater
from rllab.envs.mujoco.mujoco_env import positive_params
from rllab.misc import logger
from rllab.misc.overrides import overrides


class SwimmerEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        xml_string = kwargs.get("xml_string", "")
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_dir, 'assets/swimmer.xml')

        self.param_value1 = kwargs.get("param_value1", 1.0)
        self.param_value2 = kwargs.get("param_value2", 1.0)

        MujocoEnv.__init__(self, model_path, frame_skip=5, xml_string=xml_string)
        utils.EzPickle.__init__(self)

        self.body_names_front = ['body1', 'body2']
        self.body_names_back = ['body3', 'body4']

        self.default_params = MujocoUpdater(self.default_xml).get_params()

        self.default_params_geom_friction = self._get_params_geom_friction(self.default_params)
        self.default_params_body_pos = self._get_params_body_pos(self.default_params)
        self.default_params_geom_size = self._get_params_geom_size(self.default_params)
        self.default_params_geom_pos = self._get_params_joint_pos(self.default_params)

        self.default_params_body_pos_front \
            = self._get_params(self.default_params, 'body__pos', self.body_names_front)
        self.default_params_geom_size_front \
            = self._get_params(self.default_params, 'geom__size', self.body_names_front)
        self.default_params_geom_pos_front \
            = self._get_params(self.default_params, 'geom__pos', self.body_names_front)

        self.default_params_body_pos_back \
            = self._get_params(self.default_params, 'body__pos', self.body_names_back)
        self.default_params_geom_size_back \
            = self._get_params(self.default_params, 'geom__size', self.body_names_back)
        self.default_params_geom_pos_back \
            = self._get_params(self.default_params, 'geom__pos', self.body_names_back)

        self.set_param_space(eps_scale=[0.2, 0.2])

    def _step(self, a):
        xposbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.qpos[0]

        vel_x = (xposafter - xposbefore) / self.dt
        vel_reward = -vel_x  # make swimmer move in negative x direction
        ctrl_cost = 1e-3 * np.square(a).sum()
        reward = vel_reward - ctrl_cost
        done = False

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
        ])

    def reset_model(self, randomize=True):
        qpos_init = self.init_qpos.copy()
        if randomize:
            qpos_init[2] = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.set_state(qpos_init, self.init_qvel)
        self.sim.forward()
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 1.2

    @overrides
    def log_diagnostics(self, paths):
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)

    def get_params(self):
        return {**self._get_param1(), **self._get_param2()}

    def set_params(self, params):
        # invalidate cached properties
        self.__dict__.pop('action_space', None)
        self.__dict__.pop('observation_space', None)

        params = {**self._set_param1(params['param_value1']),
                  **self._set_param2(params['param_value2'])}
        self.model_xml = MujocoUpdater.set_params(self.model_xml, params)

        self.__init__(xml_string=self.model_xml,
                      param_value1=self.param_value1,
                      param_value2=self.param_value2)

        self.reset()

        return self

    def set_param_space(self, eps_scale=0.5):
        """
        Set param_space
        @param param_space: dict(string, rllab.space.base.Space)
        """
        params = {**self._get_param1(), **self._get_param2()}

        if not isinstance(eps_scale, list):
            eps_scale = np.ones(len(params)) * eps_scale

        self._param_space = dict()
        i = 0
        for param, value in params.items():
            eps = np.abs(value) * eps_scale[i]
            ub = value + eps
            lb = value - eps
            for name in positive_params:
                if name in param:
                    lb = np.clip(lb, 1e-3, ub)
                    break
            space = spaces.Box(lb, ub)
            self._param_space[param] = space
            i += 1

    def _get_params(self, params, param_name, body_names):
        sub_params = dict()
        for key, value in params.items():
            if param_name in key:
                for body_name in body_names:
                    if body_name in key:
                        sub_params[key] = value
        return sub_params

    def _get_params_body_pos(self, params):
        params_body_pos = dict()
        for key, value in params.items():
            if 'torso' in key:
                continue
            if 'body__pos' in key:
                params_body_pos[key] = value
        return params_body_pos

    def _get_params_geom_friction(self, params):
        params_geom_friction = dict()
        for key, value in params.items():
            if 'geom__friction' in key:
                params_geom_friction[key] = value
        return params_geom_friction

    def _get_params_geom_size(self, params):
        params_geom_size = dict()
        for key, value in params.items():
            if 'geom__size' in key:
                params_geom_size[key] = value
        return params_geom_size

    def _get_params_joint_pos(self, params):
        params_geom_pos = dict()
        for key, value in params.items():
            if 'geom__pos' in key:
                params_geom_pos[key] = value
        return params_geom_pos

    def _set_param1(self, param_value):
        self.param_value1 = param_value
        return self._set_param_geom_length_front(param_value)

    def _get_param1(self):
        return {'param_value1': self.param_value1}

    def _set_param2(self, param_value):
        self.param_value2 = param_value
        return self._set_param_geom_length_back(param_value)

    def _get_param2(self):
        return {'param_value2': self.param_value2}

    def _set_param_geom_friction(self, param_value):
        body_frictions = self.default_params_geom_friction.copy()

        for key, value in body_frictions.items():
            value *= param_value

        return body_frictions

    def _set_param_geom_length(self, param_value):
        params_geom_size = self.default_params_geom_size.copy()
        params_geom_pos = self.default_params_geom_pos.copy()
        params_body_pos = self.default_params_body_pos.copy()

        for key, value in params_geom_size.items():
            value *= param_value
        for key, value in params_geom_pos.items():
            value *= param_value
        for key, value in params_body_pos.items():
            value *= param_value

        return {**params_geom_size, **params_geom_pos, **params_body_pos}

    def _set_param_geom_length_front(self, param_value):
        params_geom_size = self.default_params_geom_size_front.copy()
        params_geom_pos = self.default_params_geom_pos_front.copy()
        params_body_pos = self.default_params_body_pos_front.copy()

        for key, value in params_geom_size.items():
            value *= param_value
        for key, value in params_geom_pos.items():
            value *= param_value
        for key, value in params_body_pos.items():
            value *= param_value

        return {**params_geom_size, **params_geom_pos, **params_body_pos}

    def _set_param_geom_length_back(self, param_value):
        params_geom_size = self.default_params_geom_size_back.copy()
        params_geom_pos = self.default_params_geom_pos_back.copy()
        params_body_pos = self.default_params_body_pos_back.copy()

        for key, value in params_geom_size.items():
            value *= param_value
        for key, value in params_geom_pos.items():
            value *= param_value
        for key, value in params_body_pos.items():
            value *= param_value

        return {**params_geom_size, **params_geom_pos, **params_body_pos}
