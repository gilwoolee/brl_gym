import random
import numpy as np
from gym import utils
from gym.spaces import Box
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer
import mujoco_py
import os
asset_dir = "/home/gilwoo/Workspace/brl_gym/brl_gym/envs/mujoco/"

class DoorsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    This environment has N doors to the goal. The doors open/closed with
    random probability. At least one of the doors is open at each time.
    The agent can move (x, y) or sense (all doors at the same time).
    The sensing returns noisy observation with error probability proportional
    to the distance to the doors.
    """
    def __init__(self):
        self.agent_bid = 0
        self.target_sid = 0
        utils.EzPickle.__init__(self)
        self.open_doors = np.array([0, 1, 0, 0]).astype(np.bool)


        self.fullpath = os.path.join(asset_dir, "assets", 'doors.xml')
        mujoco_env.MujocoEnv.__init__(self, self.fullpath, 5)

        self.agent_bid = self.sim.model.body_name2id('agent')
        self.target_sid = self.sim.model.site_name2id('target')
        self.doors_sids = [self.sim.model.site_name2id('door{}'.format(x)) for x in range(4)]
        self.door_pos = np.array([self.data.site_xpos[x].ravel() for x in self.doors_sids])
        self.target_pos = self.data.site_xpos[self.target_sid].ravel()[:2]
        self.action_space = Box(np.concatenate([self.action_space.low, [-1]]), np.concatenate([self.action_space.high, [1]]))

    def step(self, a):
        if len(a) == 3:
            a = np.clip(a, np.array([-1.0, -1.0, -1]), np.array([1.0, 1.0, 1]))
        else:
            # This is used only during initial setup.
            a = np.clip(a, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        # self.data.qvel[:2] = 0

        self.do_simulation(a[:2], self.frame_skip)

        agent_pos = self.data.body_xpos[self.agent_bid].ravel()[:2]
        target_pos = self.data.site_xpos[self.target_sid].ravel()[:2]
        dist = np.linalg.norm(agent_pos-target_pos)

        reward = 0 #-dist * 0.01 # reward if closer
        if len(a) == 3 and a[2] > 0:
            doors, accuracy =  self._sense()
            info = {'doors': doors, 'accuracy': accuracy}
            reward -= 0.1
        else:
            info = {}

        obs = self._get_obs()
        if np.any(obs[-8:-4] == 1.0):
            info['collision'] = np.argmax(obs[-8:-4] == 1)
            reward = -10 # Collision
        if np.any(obs[-4:] == 1.0):
            info['pass_through'] = np.argmax(obs[-4:] == 1)

        done = False

        if dist < 0.2:
            reward = 100.0 # bonus for being very close
            done = True

        # reward -= np.linalg.norm(a[:2])*0.1

        return obs, reward, done, info

    def _get_obs(self):
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()

        goal_dist = target_pos[:2] - agent_pos[:2]

        sim = self.sim
        collision_with_door = np.zeros(4).astype(np.int)
        pass_through_door = np.zeros(4).astype(np.int)

        for i in range(sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = sim.data.contact[i]
            name1 = sim.model.geom_id2name(contact.geom1)
            name2 = sim.model.geom_id2name(contact.geom2)
            if name1 == 'agent' and 'door' in name2:
                door_id = int(name2.split("_")[0][-1])
                collision_with_door[door_id] = 1
            elif name2 == 'agent' and 'door' in name1:
                door_id = int(name1.split("_")[0][-1])
                collision_with_door[door_id] = 1

        if agent_pos[1] >= 0.2246 and agent_pos[1] <= 0.27:
            # check for door pass-through
            if agent_pos[0] >= -1.15 and agent_pos[0] <= -0.75:
                pass_through_door[0] = 1
            elif agent_pos[0] >= -0.55 and agent_pos[0] <= -0.15:
                pass_through_door[1] = 1
            elif agent_pos[0] >= 0.25 and agent_pos[0] <= 0.55:
                pass_through_door[2] = 1
            elif agent_pos[0] >= 0.85 and agent_pos[0] <= 1.15:
                pass_through_door[3] = 1


        return np.concatenate([
            agent_pos[:2],
            self.data.qvel.ravel(),
            goal_dist.ravel(),
            collision_with_door,
            pass_through_door
            ])

    def _sense(self):
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()

        door_dist = np.linalg.norm(self.door_pos[:, :2] - agent_pos[:2], axis=1)
        accuracy = 0.5 + np.exp(-door_dist) / 2.0

        noisy_sensing = []
        for x in range(4):
            r = random.random()
            if r < accuracy[x]:
                noisy_sensing += [self.open_doors[x]]
            else:
                noisy_sensing += [not self.open_doors[x]]

        with open("door_sensing.txt","a+") as f:
            f.write("{}\t{}\n".format(agent_pos[0], agent_pos[1]))

        return np.array(noisy_sensing), accuracy


    def reset_model(self):
        agent_x = np.random.uniform(-1, 1)
        agent_y = -1.0

        qp = np.array([agent_x, agent_y])
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.sim.forward()

        for i, door in enumerate(self.open_doors):
            if door:
                self.model.geom_conaffinity[10 + i] = 0
                self.model.geom_rgba[10+i, -1] = 0.0 # Make it transparent

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
    env = DoorsEnv()
    env.reset()
    print (env.model.geom_conaffinity[10:14])

    while True:
        env.step(env.action_space.sample())
        env.render()
