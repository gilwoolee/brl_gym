import numpy as np
from gym import utils
from gym.spaces import Box
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer
import os
asset_dir = "/home/gilwoo/Workspace/brl_gym/brl_gym/envs/mujoco/"

GOAL_POSE = np.array([[-0.25, 0.3], [1.2, 0.0]])

class PointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.agent_bid = 0
        utils.EzPickle.__init__(self)
        self.target = 0
        self.target_sid = 0

        self.fullpath = os.path.join(asset_dir, "assets", 'point_mass.xml')
        mujoco_env.MujocoEnv.__init__(self, self.fullpath, 5)

        self.agent_bid = self.sim.model.body_name2id('agent')
        self.target_sid = self.sim.model.site_name2id('target')

        self.action_space = Box(np.concatenate([self.action_space.low, [-1]]), np.concatenate([self.action_space.high, [1]]))

    def step(self, a):
        if len(a) == 3:
            a = np.clip(a, np.array([-1.4, -1.4, -1]), np.array([1.4, 1.4, 1]))
        else:
            a = np.clip(a, np.array([-1.4, -1.4]), np.array([1.4, 1.4]))
        self.do_simulation(a[:2], self.frame_skip)

        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()
        dist = np.linalg.norm(agent_pos-target_pos)

        # Distance to the other target
        other_target = 1 if self.target == 0 else 0
        dist_to_other = np.linalg.norm(GOAL_POSE[other_target] - agent_pos[:2])
        print(other_target, self.target, dist_to_other, dist, self.target_sid, target_pos)
        #reward = -0.01*dist
        # reward = -2
        reward = 0
        done = False
        if dist < 0.2:
            reward = 500.0 # bonus for being very close
            done = True
        if dist_to_other < 0.2:
            reward = -500 # Penalty for getting close to the other target
            done = True

        if len(a) == 3:
            if a[2] > 0:
                reward = -1 #0.1
                info = {'goal_dist':dist}
            else:
                info = {}
        else:
            info = {}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()
        goal_dist = GOAL_POSE - agent_pos[:2]
        dist = np.linalg.norm(goal_dist[self.target]) + np.random.normal() * 0.5
        return np.concatenate([agent_pos[:2], self.data.qvel.ravel(), goal_dist.ravel(),
            [dist]])

    def reset_model(self):
        # randomize the agent and goal
        # agent_x = self.np_random.uniform(low=-0.2, high=0.2) + 0.3
        # agent_y = self.np_random.uniform(low=-0.2, high=0.2) + 0.3

        agent_x = 0.3
        agent_y = 0.3


        target = self.target #np.random.choice(4)
        qp = np.array([agent_x, agent_y])
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.site_pos[self.target_sid][0] = GOAL_POSE[target,0]
        self.model.site_pos[self.target_sid][1] = GOAL_POSE[target,1]

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
    env = PointMassEnv()
    env.target = 0
    o = env.reset()
    # import IPython; IPython.embed()

    from brl_gym.envs.mujoco.motion_planner.maze import MotionPlanner

    start = o[:2]
    target = o[-2:]
    mp = MotionPlanner()
    waypoints = mp.motion_plan(start, target)

    # print(waypoints)
    # import IPython; IPython.embed()
    lookahead = 2

    d = False
    while not d:

        #idx = min(mp.get_closest_point(waypoints, o[:2]) + lookahead, waypoints.shape[0]-1)

        #direction = waypoints[idx] - o[:2]
        #direction /= (np.linalg.norm(direction) + 1e-3)

        # xy = np.array([direction[1], direction[0]])
        # print(start, target, direction)
        #o, _, _, _ = env.step(direction + np.random.normal(size=2)*0.02)
        o, r, d, _ = env.step(env.action_space.sample())
        env.render()
    print("final reward", r)