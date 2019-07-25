import numpy as np
from gym import utils
from rllab import spaces
from rllab.envs.mujoco.model_updater import MujocoUpdater
from rllab.envs.mujoco.mujoco_env import MujocoEnv, positive_params
from mujoco_py import MjViewer
import os


class AntEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        xml_string = kwargs.get("xml_string", "")
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_dir, 'assets/ant.xml')

        self.param_value1 = kwargs.get("param_value1", 1.0)
        self.param_value2 = kwargs.get("param_value2", 1.0)

        MujocoEnv.__init__(self, model_path, frame_skip=5, xml_string=xml_string, )
        utils.EzPickle.__init__(self)

        self.body_names_front = ['front_', 'aux_1', 'aux_2']
        self.body_names_back = ['back_', 'aux_3', 'aux_4']

        self.body_names_front_left = ['front_left_leg', 'aux_1']
        self.body_names_front_right = ['front_right_leg', 'aux_2']
        self.body_names_back_left = ['back_leg', 'aux_3']
        self.body_names_back_right = ['right_back_leg', 'aux_4']

        self.default_params = MujocoUpdater(self.default_xml).get_params()

        self.default_params_geom_friction = self._get_params_geom_friction(self.default_params)

        self.default_params_geom_density = self._get_params_geom_density(self.default_params)

        self.default_params_body_pos_front = self._get_params_body_pos_front(self.default_params)
        self.default_params_geom_fromto_front = self._get_params_geom_fromto_front(self.default_params)
        self.default_params_geom_size_front = self._get_params_geom_size_front(self.default_params)

        self.default_params_body_pos_back = self._get_params_body_pos_back(self.default_params)
        self.default_params_geom_fromto_back = self._get_params_geom_fromto_back(self.default_params)
        self.default_params_geom_size_back = self._get_params_geom_size_back(self.default_params)

        self.default_params_body_pos_front_left \
            = self._get_params(self.default_params, 'body__pos', self.body_names_front_left)
        self.default_params_geom_fromto_front_left \
            = self._get_params(self.default_params, 'geom__fromto', self.body_names_front_left)
        self.default_params_geom_size_front_left \
            = self._get_params(self.default_params, 'geom__size', self.body_names_front_left)

        self.default_params_body_pos_front_right \
            = self._get_params(self.default_params, 'body__pos', self.body_names_front_right)
        self.default_params_geom_fromto_front_right \
            = self._get_params(self.default_params, 'geom__fromto', self.body_names_front_right)
        self.default_params_geom_size_front_right \
            = self._get_params(self.default_params, 'geom__size', self.body_names_front_right)

        self.default_params_body_pos_back_left \
            = self._get_params(self.default_params, 'body__pos', self.body_names_back_left)
        self.default_params_geom_fromto_back_left \
            = self._get_params(self.default_params, 'geom__fromto', self.body_names_back_left)
        self.default_params_geom_size_back_left \
            = self._get_params(self.default_params, 'geom__size', self.body_names_back_left)

        self.default_params_body_pos_back_right \
            = self._get_params(self.default_params, 'body__pos', self.body_names_back_right)
        self.default_params_geom_fromto_back_right \
            = self._get_params(self.default_params, 'geom__fromto', self.body_names_back_right)
        self.default_params_geom_size_back_right \
            = self._get_params(self.default_params, 'geom__size', self.body_names_back_right)

        self.set_param_space(eps_scale=[0.2, 0.2])

    def _step(self, a):
        xposbefore = self.data.body_xpos[1, 0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.body_xpos[1, 0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .1 * np.square(a).sum()
        reward = forward_reward - ctrl_cost
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(forward_reward=forward_reward, ctrl_cost=ctrl_cost)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
        ])

    def reset_model(self, randomize=True):
        if randomize:
            qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
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
        self.viewer.cam.distance = self.model.stat.extent * 1.2

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

    def _get_params_geom_friction(self, params):
        params_geom_friction = dict()
        for key, value in params.items():
            if 'geom__friction' in key:
                params_geom_friction[key] = value
        return params_geom_friction

    def _get_params_geom_density(self, params):
        params_geom_density = dict()
        for key, value in params.items():
            if 'geom__density' in key and 'torso' in key:
                params_geom_density[key] = value
        return params_geom_density

    def _get_params(self, params, param_name, body_names):
        sub_params = dict()
        for key, value in params.items():
            if param_name in key:
                for body_name in body_names:
                    if body_name in key:
                        sub_params[key] = value
        return sub_params

    def _get_params_body_pos_front(self, params):
        params_body_pos = dict()
        for key, value in params.items():
            if 'body__pos' in key:
                for body_name in self.body_names_front:
                    if body_name in key:
                        params_body_pos[key] = value
        return params_body_pos

    def _get_params_body_pos_back(self, params):
        params_body_pos = dict()
        for key, value in params.items():
            if 'body__pos' in key:
                for body_name in self.body_names_back:
                    if body_name in key:
                        params_body_pos[key] = value
        return params_body_pos

    def _get_params_geom_fromto_front(self, params):
        params_geom_size = dict()
        for key, value in params.items():
            if 'geom__fromto' in key:
                for body_name in self.body_names_front:
                    if body_name in key:
                        params_geom_size[key] = value
        return params_geom_size

    def _get_params_geom_fromto_back(self, params):
        params_geom_size = dict()
        for key, value in params.items():
            if 'geom__fromto' in key:
                for body_name in self.body_names_back:
                    if body_name in key:
                        params_geom_size[key] = value
        return params_geom_size

    def _get_params_geom_size_front(self, params):
        params_geom_size = dict()
        for key, value in params.items():
            if 'geom__size' in key:
                for body_name in self.body_names_front:
                    if body_name in key:
                        params_geom_size[key] = value
        return params_geom_size

    def _get_params_geom_size_back(self, params):
        params_geom_size = dict()
        for key, value in params.items():
            if 'geom__size' in key:
                for body_name in self.body_names_back:
                    if body_name in key:
                        params_geom_size[key] = value
        return params_geom_size

    def _set_param1(self, param_value):
        self.param_value1 = param_value
        return self._set_param_front_left_leg(param_value)

    def _get_param1(self):
        return {'param_value1': self.param_value1}

    def _set_param2(self, param_value):
        self.param_value2 = param_value
        return self._set_param_front_right_leg(param_value)

    def _get_param2(self):
        return {'param_value2': self.param_value2}

    def _set_param_geom_friction(self, param_value):
        body_frictions = self.default_params_geom_friction.copy()

        for key, value in body_frictions.items():
            value *= param_value

        return body_frictions

    def _set_param_geom_density(self, param_value):
        geom_density = self.default_params_geom_density.copy()

        for key, value in geom_density.items():
            value *= param_value

        return geom_density

    def _set_param_geom_length(self, param_value):
        param_value_front = param_value
        param_value_back = 2.0 - param_value

        params_geom_fromto_front = self.default_params_geom_fromto_front.copy()
        params_geom_fromto_back = self.default_params_geom_fromto_back.copy()

        params_geom_size_front = self.default_params_geom_size_front.copy()
        params_geom_size_back = self.default_params_geom_size_back.copy()

        params_body_pos_front = self.default_params_body_pos_front.copy()
        params_body_pos_back = self.default_params_body_pos_back.copy()

        for key, value in params_geom_fromto_front.items():
            value *= param_value_front
        for key, value in params_geom_size_front.items():
            value *= param_value_front
        for key, value in params_body_pos_front.items():
            value *= param_value_front

        for key, value in params_geom_fromto_back.items():
            value *= param_value_back
        for key, value in params_geom_size_back.items():
            value *= param_value_back
        for key, value in params_body_pos_back.items():
            value *= param_value_back

        return {**params_geom_fromto_front, **params_geom_size_front, **params_body_pos_front,
                **params_geom_fromto_back,  **params_geom_size_back,  **params_body_pos_back}

    def _set_param_front_left_leg(self, param_value):
        param_value_front = param_value

        params_geom_fromto_front_left = self.default_params_geom_fromto_front_left.copy()
        params_geom_size_front_left = self.default_params_geom_size_front_left.copy()
        params_body_pos_front_left = self.default_params_body_pos_front_left.copy()

        for key, value in params_geom_fromto_front_left.items():
            value *= param_value_front
        for key, value in params_geom_size_front_left.items():
            value *= param_value_front
        for key, value in params_body_pos_front_left.items():
            value *= param_value_front

        return {**params_geom_fromto_front_left,
                **params_geom_size_front_left,
                **params_body_pos_front_left}

    def _set_param_front_right_leg(self, param_value):
        param_value_front = param_value

        params_geom_fromto_front_right = self.default_params_geom_fromto_front_right.copy()
        params_geom_size_front_right = self.default_params_geom_size_front_right.copy()
        params_body_pos_front_right = self.default_params_body_pos_front_right.copy()

        for key, value in params_geom_fromto_front_right.items():
            value *= param_value_front
        for key, value in params_geom_size_front_right.items():
            value *= param_value_front
        for key, value in params_body_pos_front_right.items():
            value *= param_value_front

        return {**params_geom_fromto_front_right,
                **params_geom_size_front_right,
                **params_body_pos_front_right}
