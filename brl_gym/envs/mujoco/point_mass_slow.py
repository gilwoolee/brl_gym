import numpy as np
from gym import utils
from gym.spaces import Box
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer
import os
asset_dir = "/home/gilwoo/Workspace/brl_gym/brl_gym/envs/mujoco/"

GOAL_POSE = np.array([[-0.25, 0.3], [1.2, 1.2], [1.2, 0.6], [0.25, 0.8]])

class PointMassSlowEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.agent_bid = 0
        utils.EzPickle.__init__(self)
        self.target = 0
        self.target_sid = 0

        self.fullpath = os.path.join(asset_dir, "assets", 'point_mass_slow.xml')
        mujoco_env.MujocoEnv.__init__(self, self.fullpath, 1)

        self.agent_bid = self.sim.model.body_name2id('agent')
        self.target_sid = self.sim.model.site_name2id('target')

        self.action_space = Box(np.ones(3) * -1, np.ones(3))

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
        dist_to_others = np.linalg.norm(GOAL_POSE - agent_pos[:2], axis=1)
        dist_to_others = np.array([x for i, x in enumerate(dist_to_others) if i != self.target])

        reward = 0 #-np.linalg.norm(a) * 0.1
        done = False
        if dist < 0.1:
            reward = 500.0 # bonus for being very close
            print("reward", reward)
            done = False#True
            self.model.site_rgba[-1, 2] = 1.0
            self.model.site_rgba[-1, -1] = 1.0
        if np.any(dist_to_others < 0.3):
            reward = -500 # Penalty for getting close to the other target
            print("penalty", reward)
            done = False # True
            self.model.site_rgba[-1, 2] = 1.0
            self.model.site_rgba[-1, -1] = 1.0

        obs = self._get_obs()
        if len(a) == 3:
            if a[2] > 0:
                dist, noise_scale = self._sense()
                reward += -1 # used to be -0.1
                obs[-1] = dist
                info = {'goal_dist':dist, 'noise_scale':noise_scale}
            else:
                info = {}
        else:
            info = {}

        return obs, reward, done, info

    def _get_obs(self):
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()
        goal_dist = GOAL_POSE - agent_pos[:2]

        return np.concatenate([agent_pos[:2], self.data.qvel.ravel(), goal_dist.ravel(),
            [0]])

    def _sense(self):
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        goal_dist = GOAL_POSE - agent_pos[:2]

        # Noisy distance
        noise_scale = np.linalg.norm(goal_dist[self.target]) / (1.8*np.sqrt(2)) * 3.0
        dist = np.linalg.norm(goal_dist[self.target]) + np.random.normal() * noise_scale
        return dist, noise_scale


    def reset_model(self):
        # randomize the agent and goal
        agent_x = self.np_random.uniform(low=-0.5, high=0.5)
        agent_y = -1.3

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

    def viewer_setup(self):
        print("viewer setup")
        self.viewer.cam.trackbodyid = -1         # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 1.0         # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.5         # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 90              # camera rotation around the camera's vertical axis


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
    env.target = 3
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
    # while not d:
    while True:

        #idx = min(mp.get_closest_point(waypoints, o[:2]) + lookahead, waypoints.shape[0]-1)

        #direction = waypoints[idx] - o[:2]
        #direction /= (np.linalg.norm(direction) + 1e-3)

        # xy = np.array([direction[1], direction[0]])
        # print(start, target, direction)
        #o, _, _, _ = env.step(direction + np.random.normal(size=2)*0.02)
        o, r, d, _ = env.step(env.action_space.sample())
        env.render()
    print("final reward", r)
