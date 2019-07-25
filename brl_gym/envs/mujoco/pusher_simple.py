import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

# from rllab.misc.overrides import overrides
# from rllab.envs.env_spec import EnvSpec
# from rllab import spaces
from gym import spaces

class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, process_noise_std=0.1, blocker_active=True):
        utils.EzPickle.__init__(self)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_dir, 'assets/pusher.xml')

        high = np.array([ 0.25,  0.25,  2.5,  0.0])
        low  = np.array([-0.25, -0.25, -2.5, -1.0])

        self.action_weights = 1.0 /(high - low)
        self.action_weights = self.action_weights / np.linalg.norm(self.action_weights)
        self.cam_name = 'top_cam'
        self.process_noise_std = process_noise_std
        self.blocker_active = blocker_active
        self.obs_noise_std = 0.0
        self.action_space = spaces.Box(low=low, high=high)

        mujoco_env.MujocoEnv.__init__(self, model_path, 2)
        self.frame_skip=2
        self.action_space = spaces.Box(low=low, high=high)
        self.observation_space = spaces.Box(low=np.ones(11) * -2, high=np.ones(11) * 2)
        # self.spec = EnvSpec(self.observation_space, self.action_space)

    def step(self, a):
        pusher_before_action = self.data.qpos[:3]
        a += np.random.normal(size=a.shape[0], scale=self.process_noise_std)
        u_clipped = np.clip(a, self.action_space.low, self.action_space.high)
        self.data.qvel[:4] = u_clipped[:4]
        u_clipped[:4] = 0
        self.do_simulation(u_clipped, self.frame_skip)
        self.data.qvel[4:] = 0
        self.set_state(self.data.qpos, self.data.qvel)

        vec_1 = self.get_body_com("object") - self.get_body_com("pusher")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        reward_near = - np.linalg.norm(vec_1[:2])
        reward_dist = - np.linalg.norm(vec_2[:2])
        # Penalize pushing more
        reward_ctrl = - np.square(a*self.action_weights)[:3].sum() - np.abs(a[-1]) * 2

        # blocker is encouraged to be on the left of the object
        view = self.get_body_com("blocker")[0] - self.get_body_com("object")[0]
        if view > 0:
            reward_view = - np.abs(view)
        else:
            reward_view = 0

        reward = reward_dist + 0.01 * reward_ctrl +  0.1 * reward_near #+ 0.1 * reward_view
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl, noise_std=self.obs_noise_std,
                pusher_before_action=pusher_before_action)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def terminate(self):
        pass

    def reset_model(self):
        qpos = self.init_qpos
        self.cylinder_pos = np.asarray([0, 0])
        while True:
            self.goal_pos = np.concatenate([
                    self.np_random.uniform(low=-0.2, high=0.2, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            #self.cylinder_pos = np.concatenate([
            #        self.np_random.uniform(low=-0.02, high=0.02, size=1),
            #        self.np_random.uniform(low=-0.02, high=0.02, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qpos[0] = self.cylinder_pos[0] - 0.05
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        # Noisy
        obj_pos = self.get_body_com("object").copy()[:2]
        blocker = self.get_body_com("blocker")[:2]
        pusher = self.get_body_com("pusher")[:2]
        goal = self.get_body_com("goal")[:2]

        dist = self.get_body_com("blocker")[0] - obj_pos[0]
        if dist > 0 and self.blocker_active:
            # blocker blocks the view
            self.obs_noise_std = np.abs(dist) * 0.5
        else:
            self.obs_noise_std = 1e-6

        obj_pos += np.random.normal(size=2,
                scale=self.obs_noise_std)

        return np.concatenate([
            obj_pos,
            goal,
            blocker[:1] - obj_pos[0],
            goal - obj_pos,
            goal - pusher,
            pusher - obj_pos
        ])

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def get_sim_state(self):
        return None
    def get_state_vector(self):
        return None


if __name__ == "__main__":
    env = PusherEnv()
    import time
    i = 0
    while True:
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.01)

