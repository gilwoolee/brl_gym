import os

import gym
from brl_gym.envs.mujoco.model_updater import MujocoUpdater

from gym import error
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import time as timer
from gym import spaces

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, load_model_from_xml, MjSim, MjViewer
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

positive_params = ["size", "damping", "stiffness"]

DEFAULT_SIZE = 500

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip=1, xml_string=""):
        """
        @param model_path path of the default model
        @param xml_string if given, the model will be reset using these values
        """

        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model = load_model_from_path(fullpath)
        with open(fullpath, 'r') as f:
            self.model_xml = f.read()
            self.default_xml = self.model_xml

        if xml_string != "":
            self.model = load_model_from_xml(xml_string)
            self.model_xml = xml_string

        self.frame_skip = frame_skip

        self.sim = MjSim(self.model)
        self.data = self.sim.data

        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.mujoco_render_frames = False

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()
        self.set_param_space()

    def get_params(self):
        """
        Returns a dict of (param_name, param_value)
        """
        return MujocoUpdater(self.model_xml).get_params()

    def set_params(self, params):
        """
        @param params: dict(param_name, param_value)
        param_name should be a string of bodyname__type__paramname
        where type is either geom or joint,
        e.g. thigh__joint__friction,
        and param_value is a numpy array
        """
        # invalidate cached properties
        self.__dict__.pop('action_space', None)
        self.__dict__.pop('observation_space', None)

        new_xml = MujocoUpdater.set_params(self.model_xml, params)
        self.__init__(xml_string=new_xml)
        self.reset()
        return self


    def set_param_space(self, param_space=None, eps_scale=0.5, replace=True):
        """
        Set param_space
        @param param_space: dict(string, gym.space.base.Space)
        @param eps_scale: scale of variation applied to all params
        @param replace: if true, param_space overwrites default param_space.
                        Default behavior is to merge.
        """
        if param_space is not None:
            if replace:
                self._param_space = param_space
            else:
                self._param_space = {**self._param_space, **param_space}
        else:
            params = MujocoUpdater(self.model_xml).get_params()
            self._param_space = dict()
            for param, value in params.items():
                eps = np.abs(value) * eps_scale
                ub = value + eps
                lb = value - eps
                for name in positive_params:
                    if name in param:
                       lb = np.clip(lb, 1e-3, ub)
                       break
                space = spaces.Box(lb, ub)
                self._param_space[param] = space

    def get_geom_params(self, body_name):
        geom = MujocoUpdater(self.model_xml).get_geom(body_name)
        return {
            k: v for k, v in geom.attrib.items()
            if k not in MujocoUpdater.ignore_params
        }

    def get_joint_params(self, body_name):
        joint = MujocoUpdater(self.model_xml).get_joint(body_name)
        return {
            k: v for k, v in joint.attrib.items()
            if k not in MujocoUpdater.ignore_params
        }

    def get_body_names(self):
        return MujocoUpdater(self.model_xml).get_body_names()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_current_obs(self):
        return self._get_full_obs()

    def _get_full_obs(self):
        data = self.sim.data
        cdists = np.copy(self.model.geom_margin).flat
        for c in self.sim.data.contact:
            cdists[c.geom2] = min(cdists[c.geom2], c.dist)
        obs = np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            # data.cdof.flat,
            data.cinert.flat,
            data.cvel.flat,
            # data.cacc.flat,
            data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
            data.qfrc_constraint.flat,
            cdists,
            # data.qfrc_bias.flat,
            # data.qfrc_passive.flat,
            self.dcom.flat,
        ])
        return obs

    @property
    def _state(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    @property
    def _full_state(self):
        return np.concatenate([
            self.sim.data.qpos,
            self.sim.data.qvel,
            self.sim.data.qacc,
            self.sim.data.ctrl,
        ]).ravel()

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def mj_viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the standard viewer cannot be used
        with this set-up. Instead we use this mujoco specific function.
        """
        pass

    def viewer_setup(self):
        """
        Does not work. Use mj_viewer_setup() instead
        """
        pass

    # -----------------------------

    def reset(self, randomize=True):
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    # Added for bayesian_rl
    def get_sim_state(self):
        return self.sim.get_state()

    # Added for bayesian_rl
    def set_sim_state(self, state):
        self.sim.set_state(state)

    # Added for bayesian_rl
    def set_state_vector(self, state_vector):
        qpos = state_vector[:self.model.nq]
        qvel = state_vector[self.model.nq:]
        self.set_state(qpos, qvel)

    # Added for bayesian_rl
    def get_state_vector(self):
        return self.state_vector()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        state = self.sim.get_state()
        for i in range(self.model.nq):
            state.qpos[i] = qpos[i]
        for i in range(self.model.nv):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]
        for _ in range(n_frames):
            self.sim.step()
            if self.mujoco_render_frames is True:
                self.mj_render()

    def mj_render(self):
        try:
            self.viewer.render()
        except:
            self.mj_viewer_setup()
            self.viewer._run_speed = 1.0
            #self.viewer._run_speed /= self.frame_skip
            self.viewer.render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    # def step(self, a):
    #     return self._step(a)

    # Added for bayesian_rl
    def take_action(self, a):
        self.step(a)
        return self.get_sim_state()

    def state_vector(self):
        state = self.sim.get_state()
        return np.concatenate([
            state.qpos.flat, state.qvel.flat])

    # -----------------------------

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        self.mujoco_render_frames = True
        for ep in range(num_episodes):
            o = self.reset()
            d = False
            t = 0
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
        self.mujoco_render_frames = False

    def visualize_policy_offscreen(self, policy, horizon=1000,
                                   num_episodes=1,
                                   mode='exploration',
                                   save_loc='/tmp/',
                                   filename='newvid',
                                   camera_name=None):
        import skvideo.io
        for ep in range(num_episodes):
            print("Episode %d: rendering offline " % ep, end='', flush=True)
            o = self.reset()
            d = False
            t = 0
            arrs = []
            t0 = timer.time()
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['mean']
                o, r, d, _ = self.step(a)
                t = t+1
                curr_frame = self.sim.render(width=640, height=480, mode='offscreen',
                                             camera_name=camera_name, device_id=0)
                arrs.append(curr_frame[::-1,:,:])
                print(t, end=', ', flush=True)
            file_name = save_loc + filename + '_' + str(ep) + ".mp4"
            skvideo.io.vwrite( file_name, np.asarray(arrs))
            print("saved", file_name)
            t1 = timer.time()
            print("time taken = %f"% (t1-t0))

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()
