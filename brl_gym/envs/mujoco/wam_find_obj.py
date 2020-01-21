import numpy as np
import os
from os import path as osp
from brl_gym.envs.mujoco import utils
from gym.spaces import Box, Discrete, Tuple
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer
import mujoco_py
from brl_gym.envs.mujoco import robot_env
from transforms3d import quaternions
from mujoco_py import cymj


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)
        pos_delta = action[:, :3].ravel()

        utils.reset_mocap2body_xpos(sim)

        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta

        # upright
        default = np.array([1,0,1,0.0])
        default /= np.linalg.norm(default)
        sim.data.mocap_quat[:] = default


class WamFindObjEnv(robot_env.RobotEnv):
    def __init__(self, noise_scale=1.0):
        self.frame_skip = 50
        self.noise_scale = noise_scale
        self._sense = 0.0

        # Get asset dir
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = osp.join(dir_path, "assets/wam/wam_table.xml")

        self.action_space = Box(-1., 1., shape=(4,), dtype='float32')

        robot_env.RobotEnv.__init__(self, model_path=model_path, n_substeps=self.frame_skip,
            initial_qpos={})

    def _get_obs(self):
        sense = self._sense > 0

        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        obj_pos = self.sim.data.get_body_xpos("object0")
        dist = obj_pos - grip_pos

        num_contacts_with_obj = 0
        num_contacts_with_shelf = 0

        # print(self.sim.data.con)
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            name1 = self.sim.model.geom_id2name(c.geom1)
            name2 = self.sim.model.geom_id2name(c.geom2)
            if name1 is None or name2 is None:
                continue
            # print(name1, name2)
            if ('object0' in name2 or 'object0' in name1) and ('finger' in name1 or 'finger' in name2 or 'hand' in name1 or 'hand' in name2):
                num_contacts_with_obj += 1
            if 'shelf' in name2 or 'shelf' in name1 and ('finger' in name1 or 'finger' in name2 or 'hand' in name1 or 'hand' in name2):
                num_contacts_with_shelf += 1
                # print(name1, name2)
            if 'shelf' in name2 or 'shelf' in name1 and ('wam' in name1 or 'wam' in name2):
                num_contacts_with_shelf += 1
                # print(name1, name2)
        self.num_contacts_with_shelf = num_contacts_with_shelf
        self.num_contacts_with_obj = num_contacts_with_obj

        # dist to top shelf
        top_shelf = self.sim.data.get_site_xpos('top-shelf-center')
        dist_to_top = (top_shelf - grip_pos)

        # bottom shelf
        bottom_shelf = self.sim.data.get_site_xpos('bottom-shelf-center')
        dist_to_bottom = (bottom_shelf - grip_pos)

        if sense:
            self.noise_scale = np.linalg.norm(dist_to_top[0])
        else:
            self.noise_scale = np.linalg.norm(dist_to_top[0]) * 5.0

        # make dist noisy
        obj_pos_noisy = obj_pos + np.random.normal(size=3, scale=self.noise_scale)

        obs = np.concatenate([grip_pos,
            obj_pos_noisy,
            top_shelf,
            bottom_shelf,
            [num_contacts_with_shelf], [num_contacts_with_obj],
            self.sim.data.qpos,
            [self.noise_scale]])
        self.obj_pos = obj_pos
        self.grip_pos = grip_pos
        return obs

    def _terminate(self):
        return False

    def _viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        # self.viewer.cam.distance = 4.0
        self.viewer.cam.distance = self.sim.model.stat.extent * 1.0
        # self.viewer.cam.lookat[2] +=  .8
        self.viewer.cam.elevation = -20.0
        self.viewer.cam.azimuth = -180

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

    def _env_setup(self, initial_qpos):
        self.angle = -np.pi/4.0 + np.clip(np.random.normal(), -0.3, 0.3)

        body_id = self.sim.model.body_name2id('object0')

        # self.sim.data.qpos[7:10] = np.array([0.2, -0.2, 0.4])
        self.sim.data.qpos[:7] = np.array([5.65, -1.76, -0.26,  1.96, -1.15 , 0.87, -1.43])
        utils.reset_mocap_welds(self.sim)

        # Move end effector into position.
        target_xy = (np.random.uniform(size=2) - 0.5)*0.5
        gripper_target = self.sim.data.get_site_xpos('robot0:grip').copy()
        gripper_rotation = np.array([1., 0., 1., 0.])

        for _ in range(100):
            self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
            self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
            self.sim.step()

        utils.reset_mocap2body_xpos(self.sim)

        self.site_poses = [self.sim.data.get_site_xpos('bookcase:pos{}'.format(x)) for x in range(2)]
        self.initial_qpos = self.sim.data.qpos.copy()

    def _is_success(self):
        obj_pos = self.sim.data.get_body_xpos("object0")
        hand_pos = self.sim.data.get_body_xpos('robot0:grip')

        # Reward if distance to the target is small (the item is within the hand)
        dist = np.linalg.norm(hand_pos - obj_pos)

        if dist < 0.08:
            return True

        return False

    def _set_action(self, action):
        """
        # Expert action: Get closer to target
        obj_pos = self.sim.data.get_body_xpos("object0")
        hand_pos = self.sim.data.get_body_xpos('robot0:grip')

        dist = obj_pos - hand_pos
        pos_ctrl = obj_pos - hand_pos
        pos_ctrl = pos_ctrl / np.linalg.norm(pos_ctrl)
        pos_ctrl *= 0.01

        # dist to top shelf
        top_shelf = self.sim.data.get_site_xpos('top-shelf-center')

        # bottom shelf
        bottom_shelf = self.sim.data.get_site_xpos('bottom-shelf-center')

        # Avoid colliding with shelf or can
        if obj_pos[2] > top_shelf[2] and hand_pos[2] < top_shelf[2] + 0.1:
            pos_ctrl[2] += 0.01
        elif obj_pos[2] < top_shelf[2] and hand_pos[2] < bottom_shelf[2]:
            pos_ctrl[2] += 0.01
        if hand_pos[1] > obj_pos[1] + 0.01 and dist[0] < 0.15:
            pos_ctrl[0] = 0.0

        rot = np.array([1,0,0,0], dtype=np.float32)
        action = np.concatenate([pos_ctrl, rot])
        """
        self._sense = action[-1]
        action = action[:-1]
        rot = np.array([1,0,0,0], dtype=np.float32)
        action = np.concatenate([action * 0.01, rot])

        mocap_set_action(self.sim, action)

    def compute_reward(self):
        obj_pos = self.sim.data.get_body_xpos("object0")
        hand_pos = self.sim.data.get_body_xpos('robot0:grip')

        # Reward if distance to the target is small (the item is within the hand)
        dist = np.linalg.norm(hand_pos - obj_pos)

        if dist < 0.08:
            return 1.0

        reward = -dist

        # See if object is in hand
        finger_left = self.sim.data.get_body_xpos('finger2_1')
        finger_right = self.sim.data.get_body_xpos('finger0_1')

        in_hand = (finger_left[1] <= obj_pos[1]
                    and finger_right[1] >= obj_pos[1]
                    and finger_left[0] <= obj_pos[0]
                    and finger_left[1] <= obj_pos[0] + 0.05)
        in_hand = in_hand and (finger_left[2] <= obj_pos[2] + 0.04
                               and finger_left[2] >= obj_pos[2])

        if in_hand:
            reward += 1.0
        # Penalize on collision with shelf
        reward -= self.num_contacts_with_shelf + self.num_contacts_with_obj

        if self._sense > 0.0:
            reward -= 1.0
        return reward

    def _reset_sim(self):
        # Randomly choose a site (shelf)
        choice = np.random.choice(2)
        site_pos = self.site_poses[choice].copy()

        # Move the object to that shelf, randomizing it
        body_id = self.sim.model.body_name2id('object0')
        xy = (np.random.uniform(size=2)-0.5)*np.array([0.16, 0.4]) + np.array([0.08, 0.0])
        xy = np.clip(xy, np.array([0.0, -0.2]), np.array([0.16, 0.2]))

        site_pos[:2] += xy
        site_pos[-1] -= 0.095


    #     """Resets a simulation and indicates whether or not it was successful.
    #     If a reset was unsuccessful (e.g. if a randomized state caused an error in the
    #     simulation), this method should indicate such a failure by returning False.
    #     In such a case, this method will be called again to attempt a the reset again.
    #     """
        self.sim.set_state(self.initial_state)
        self.sim.model.body_pos[body_id] = site_pos
        self.sim.forward()
        self._sense = 0.0
        return True

if __name__ == "__main__":
    env = WamFindObjEnv()

    o = env.reset()
    t = 0
    while True:
        t += 1
        action = env.action_space.sample()
        # action[3] = 0
        o, r, d, info = env.step(action)
        # print("done", d)
        # print("info", info)
        print("rew", r)
        #if r == 1:
        #    print(t)
        #    break

        # if d:
        #     import IPython; IPython.embed(); import sys; sys.exit(0)


        env.render()
