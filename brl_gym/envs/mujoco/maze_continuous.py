import numpy as np
from gym import utils
from gym.spaces import Box
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer
import os
asset_dir = "/home/gilwoo/Workspace/brl_gym/brl_gym/envs/mujoco/"

# Goal can be anywhere
class MazeContinuous(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.agent_bid = 0
        utils.EzPickle.__init__(self)
        self.target_sid = 0

        self.fullpath = os.path.join(asset_dir, "assets", 'mazeContinuous.xml')
        mujoco_env.MujocoEnv.__init__(self, self.fullpath, 30) # originally 5

        self.agent_bid = self.sim.model.body_name2id('agent')

        self.action_space = Box(np.ones(3) * -1, np.ones(3))

    def step(self, a):
        # if len(a) == 3:
        #     a = np.clip(a, np.array([-1.4, -1.4, -1]), np.array([1.4, 1.4, 1]))
        # else:
        #     a = np.clip(a, np.array([-1.4, -1.4]), np.array([1.4, 1.4]))
        self.do_simulation(a[:2], self.frame_skip)

        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()
        dist = np.linalg.norm(agent_pos-target_pos)

        reward = 0 #-np.linalg.norm(a) * 0.1
        done = False
        if dist < 0.1:
            reward = 500.0 # bonus for being very close
            done = True

        obs = self._get_obs()
        if len(a) == 3:
            if a[2] > 0:
                dist, noise_scale = self._sense()
                reward += -0.1 # used to be -0.1
                obs[-2:] = dist
                info = {'goal_dist':dist, 'noise_scale':noise_scale}
            else:
                info = {}
        else:
            info = {}

        return obs, reward, done, info

    def _get_obs(self):
        goal_pose = self.model.site_pos[self.target_sid][:2]
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()

        return np.concatenate([agent_pos[:2], self.data.qvel.ravel(), [0, 0]])

    def _sense(self):
        goal_pose = self.model.site_pos[self.target_sid][:2]
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        goal_dist = goal_pose - agent_pos[:2]

        # Noisy distance
        noise_scale = np.linalg.norm(goal_dist) * 2.0 # (1.8*np.sqrt(2))
        goal_dist += np.random.normal() * noise_scale

        return goal_dist, noise_scale


    def reset_model(self):

        # randomize the agent and goal
        agent_x = 0.0 #self.np_random.uniform(low=-0.2, high=0.2)
        agent_y = self.np_random.uniform(low=-0.4, high=0.4)

        qp = np.array([agent_x, agent_y])
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.target_sid = self.sim.model.site_name2id('target')

        goal_x = self.np_random.uniform(low=-1.3, high=1.3)
        goal_x = round(goal_x / 0.3) * 0.3 - 0.05
        goal_y = self.np_random.uniform(low=-1.3, high=1.3)

        self.model.site_pos[self.target_sid][0] = goal_x
        self.model.site_pos[self.target_sid][1] = goal_y

        # Visualize the goal
        self.model.site_rgba[self.target_sid] = np.array([1.0, 0.0, 0.0, 0.1])

        self.sim.forward()
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.sim.forward()

    def get_state(self):

        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def set_state(self, qpos, qvel):
        state = self.sim.get_state()
        for i in range(len(qpos)):
            state.qpos[i] = qpos[i]
        for i in range(len(qvel)):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()



if __name__ == "__main__":
    env = MazeContinuous()
    o = env.reset()

    d = False
    while not d:
        # o, r, d, info = env.step(env.action_space.sample())
        o, r, d, info = env.step(np.array([0,-0.1]))
        #print(info)
        env.render()
    print("final reward", r)
