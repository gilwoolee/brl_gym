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

class WamEnv(robot_env.RobotEnv):
    def __init__(self):

        self.frame_skip = 1

        # Get asset dir
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = osp.join(dir_path, "assets/wam/wam_pusher.xml")

        self.action_space = Tuple([Box(-1., 1., shape=(3,), dtype='float32'), Discrete(2)])

        robot_env.RobotEnv.__init__(self, model_path=model_path, n_substeps=self.frame_skip,
            initial_qpos={})

        self.angle = 0.0

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos, gripper_state, grip_velp, gripper_vel
        ])

        cymj._mj_inverse(self.sim.model, self.sim.data)

        obj_pos = self.sim.data.get_body_xpos('object0')
        # print(self.sim.data.qpos)
        print('obj pos', obj_pos)

        # print('actuator force', self.sim.data.qfrc_inverse)
        print('sensor force', np.around(self.sim.data.sensordata[:3],2))
        print('sensor torque', np.around(self.sim.data.sensordata[3:],2))

        num_contacts = 0
        for i, c in enumerate(self.sim.data.contact):
            name1 = self.sim.model.geom_id2name(c.geom1)
            name2 = self.sim.model.geom_id2name(c.geom2)
            if name1 is None or name2 is None:
                continue
            if 'object0-bottom' in name2 or 'object0-bottom' in name1:
                # if c.dist < -1e-5:
                num_contacts += 1
                print (i, name1, name2, c.dist, c.pos)

        print('-------', len(self.sim.data.contact), '----')

        object_in_contact = num_contacts >= 2
        force = self.sim.data.sensordata[:3]
        # if (not object_in_contact or obj_pos[2]>0.115) and force[2] <0:
        # if force[2] < 0:
            # import IPython; IPython.embed(); import sys; sys.exit(0)
        return obs


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

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
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.data.qpos[:] = np.array([5.65, -1.76, -0.26,  1.96, -1.15 , 0.87, -1.43])

        utils.reset_mocap_welds(self.sim)
        self.sim.step()

        # # Move end effector into position.
        print('grip',self.sim.data.get_site_xpos('robot0:grip'))
        gripper_target = np.array([0, 0.0, -0.1]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        for _ in range(1000):
            self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
            self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

            self.sim.step()

        utils.reset_mocap2body_xpos(self.sim)
        print('grip', self.sim.data.get_site_xpos('robot0:grip'))
        print(self.sim.data.qpos)
        for _ in range(1000):
            self.sim.step()
        print(self.sim.data.qpos)
        # import IPython; IPython.embed(); import sys; sys.exit(0)



    def _is_success(self):
        return False # Not implemented

    def _set_action(self, action):

        obj_pos = self.sim.data.get_body_xpos('object0')
        # print(self.sim.data.qpos)
        # print('obj pos', obj_pos)


        action = action[0]
        action, lift = action[0], action[1]

        action = np.clip(action, self.action_space[0].low, self.action_space[0].high)

        lift = 0
        if lift == 1:
            pos_ctrl = np.array([0,0,0.05])
            rot_ctrl = quaternions.axangle2quat([1, 0, .0], 0)
            action = np.concatenate([pos_ctrl, rot_ctrl])

        else:
            action = action.copy()  # ensure that we don't change the action outside of this scope

            pos_ctrl, rot_ctrl = action[:2], action[2]


            pos_ctrl *= 0.001  # limit maximum change in position
            rot_ctrl *= 0.05   # limit maximum change in rotation

            pos_ctrl = np.hstack([pos_ctrl, [-0.01]])
            pos_ctrl[0] = 0.005
            pos_ctrl[1] = 0.005
            self.angle += rot_ctrl
            print(self.angle)
            rot_ctrl = quaternions.axangle2quat([1, 0, .0], self.angle)
            action = np.concatenate([pos_ctrl, rot_ctrl])

        utils.mocap_set_action(self.sim, action)

    def compute_reward(self):
        # import IPython; IPython.embed(); import sys; sys.exit(0)
        return 0

if __name__ == "__main__":
    env = WamEnv()

    o = env.reset()

    while True:
        action = env.action_space.sample()
        o, _, _, _ = env.step([action])

        env.render()
