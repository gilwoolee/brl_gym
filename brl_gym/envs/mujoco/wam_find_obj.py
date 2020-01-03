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

class WamFindObjEnv(robot_env.RobotEnv):
    def __init__(self):
        self.frame_skip = 1

        # Get asset dir
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = osp.join(dir_path, "assets/wam/wam_table.xml")

        self.action_space = Box(-1., 1., shape=(4,), dtype='float32')

        robot_env.RobotEnv.__init__(self, model_path=model_path, n_substeps=self.frame_skip,
            initial_qpos={})

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # body_id = self.sim.model.body_name2id('object0')
        # print("body", self.sim.model.body_pos[body_id])
        # print("q   ", np.around(self.sim.data.qpos[7:],2))

        # return robot_qpos
        # obj_pos = self.sim.data.get_body_xpos('object0')
        num_contacts = 0
        for i, c in enumerate(self.sim.data.contact):
            name1 = self.sim.model.geom_id2name(c.geom1)
            name2 = self.sim.model.geom_id2name(c.geom2)
            # print(i, name1, name2)
            if name1 is None or name2 is None:
                # print(i, name1, name2)
                continue
            if 'object0' in name2 or 'object0' in name1:
                # print(i, name1, name2)
                num_contacts += 1
        # print("num c", num_contacts)
        return robot_qpos
        # force = self.sim.data.sensordata[0] # Normal force from table
        # object_in_contact = num_contacts >= 1 #and force > 0

        # print(num_contacts, force)
        obs = np.concatenate([
            grip_pos, [int(object_in_contact)], [num_contacts / 3.0], [force / 300.0],
            self.sim.data.qpos
        ])

        self.obj_pos = obj_pos
        self.object_in_contact = object_in_contact

        # print(np.around(self.joint_effort,1))
        # print("grip", grip_pos)
        return obs

    # def reset(self):
    #     print("----- reset ------")
    #     # Attempt to reset the simulator. Since we randomize initial conditions, it
    #     # is possible to get into a state with numerical issues (e.g. due to penetration or
    #     # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
    #     # In this case, we just keep randomizing until we eventually achieve a valid initial
    #     # configuration.
    #     # super(RobotEnv, self).reset()

    #     did_reset_sim = False
    #     while not did_reset_sim:
    #         try:
    #             self._env_setup({})
    #             did_reset_sim = True
    #         except :
    #             continue
    #     #     did_reset_sim = self._reset_sim()
    #     # self.goal = self._sample_goal().copy()

    #     # self.sim.data.qpos[:] = self.initial_qpos

    #     obs = self._get_obs()

    #     return obs

    def _terminate(self):
        # if not self.object_in_contact and not self.limit_satisfied():
        #     print(self.mass, "BAD limit", np.around(self.joint_effort,1))
        #     return True
        # if self.obj_pos[2] > LIFT_HEIGHT:
        #     print(self.mass, 'GOOD height')
        #     return True
        # else:
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
        print("-------------------- ENV SETUP -------------------------")
        # self.sim.forward()
        self.angle = -np.pi/4.0 #+ np.clip(np.random.normal(), -0.3, 0.3)

        body_id = self.sim.model.body_name2id('object0')

        # self.sim.model.body_pos[body_id] = np.array([0.2, -0.2, 0.4])
        self.sim.data.qpos[7:10] = np.array([0.2, -0.2, 0.4])
        self.sim.data.qpos[:7] = np.array([5.65, -1.76, -0.26,  1.96, -1.15 , 0.87, -1.43])
        utils.reset_mocap_welds(self.sim)

        # Move end effector into position.
        target_xy = (np.random.uniform(size=2) - 0.5)*0.5
        # gripper_target = np.array([target_xy[0], target_xy[1] - 0.4, -0.02]) + self.sim.data.get_site_xpos('robot0:grip')
        # gripper_target = np.array([target_xy[0], target_xy[1], -0.02]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = self.sim.data.get_site_xpos('robot0:grip').copy()
        # gripper_rotation = self.sim.data.get_body_xquat('robot0:grip').copy()
        gripper_rotation = np.array([1., 0., 1., 0.])

        for _ in range(100):
            self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
            self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
            self.sim.step()

        utils.reset_mocap2body_xpos(self.sim)
        obj_quat = self.sim.data.get_body_xquat('robot0:grip')

        self.site_poses = [self.sim.data.get_site_xpos('bookcase:pos{}'.format(x)) for x in range(2)]
        print([np.around(x,2) for x in self.site_poses])
        self.initial_qpos = self.sim.data.qpos.copy()

    def _is_success(self):
        return False
        # return self.limit_satisfied() and self.obj_pos[2] > LIFT_HEIGHT

    def _set_action(self, action):
        # return
        # Get closer to target
        obj_pos = self.sim.data.get_body_xpos("object0")
        hand_pos = self.sim.data.get_body_xpos('robot0:grip')

        pos_ctrl = obj_pos - hand_pos
        pos_ctrl = pos_ctrl / np.linalg.norm(pos_ctrl)
        pos_ctrl *= 0.01
        rot = np.array([1,0,0,0], dtype=np.float32)
        action = np.concatenate([pos_ctrl, rot])
        utils.mocap_set_action(self.sim, action)
        # Current hand pose


        # self.sim.data.qpos[:7] = np.array([5.65, -1.76, -0.26,  1.96, -1.15 , 0.87, -1.43])

        # site_poses = [self.sim.data.get_site_xpos('bookcase:pos{}'.format(x)) for x in range(4)]

        # utils.mocap_set_action(self.sim, np.array([0.3,-0.2,0,1,0,0,0]))

        return
        # obj_pos = self.sim.data.get_body_xpos('robot0:grip')
        # obj_quat = self.sim.data.get_body_xquat('robot0:grip')

        # action = np.clip(action, self.action_space.low, self.action_space.high)
        # action, lift = action[:3], action[3]
        # action = action.copy()

        # rot_ctrl = quaternions.axangle2quat([1, 0, .0], self.angle)

        # if not self.object_in_contact:
        #     # Once lifted you can only lift or hold
        #     if lift <=0:
        #         pos_ctrl = np.zeros(3)
        #         action = np.concatenate([pos_ctrl, rot_ctrl])
        #     else:
        #         pos_ctrl = np.array([0,0,0.01])
        #         action = np.concatenate([pos_ctrl, rot_ctrl])
        # else:
        #     pos_ctrl, rot_ctrl = action[:2], action[2]
        #     pos_ctrl *= 0.05  # limit maximum change in position
        #     rot_ctrl *= 0.05   # limit maximum change in rotation

        #     pos_ctrl = np.hstack([pos_ctrl, [-0.001]])
        #     self.angle += rot_ctrl
        #     rot_ctrl = quaternions.axangle2quat([1, 0, .0], self.angle)
        #     action = np.concatenate([pos_ctrl, rot_ctrl])

        # # print("action", action)
        # utils.mocap_set_action(self.sim, action)

    def compute_reward(self):
        return 0
        # ctrl = np.linalg.norm(self.ctrl_force / JOINT_EFFORT_LIMITS)**2

        # if self.object_in_contact:
        #     return 0.01

        # if not self.limit_satisfied():
        #     return -ctrl * 50

        # height = self.obj_pos[2] - 0.11

        # if self.obj_pos[2] >= LIFT_HEIGHT:
        #     # print("good", 1.0 - 0.1*ctrl)
        #     return 0.1 + 10 - 0.1*ctrl

        # print("up", height - 0.1*ctrl)
        return  0.1 + height - 0.1*ctrl


    def _reset_sim(self):
        print("--------- reset sim ------ ")
        # Randomly choose a site
        choice = np.random.choice(2)
        # choice = 0
        print("choice", choice)
        site_pos = self.site_poses[choice].copy()
        # Change the pose of the object
        body_id = self.sim.model.body_name2id('object0')
        # Move x/y randomly
        xy = np.random.normal(size=2)*np.array([0.1, 0.2])
        xy = np.clip(xy, np.array([-0.06, -0.2]), np.array([0.15, 0.2]))

        # xy = np.array([0.15, -0.2])
        # print(xy)

        # print("body pos", self.sim.model.body_pos[body_id])
        site_pos[:2] += xy
        site_pos[-1] -= 0.095

        # self.sim.model.body_pos[body_id] = site_pos
        print("site pos", np.around(site_pos,2))
        # print("body pos", self.sim.model.body_pos[body_id])
        self.sim.data.qpos[7:10] = site_pos


    #     """Resets a simulation and indicates whether or not it was successful.
    #     If a reset was unsuccessful (e.g. if a randomized state caused an error in the
    #     simulation), this method should indicate such a failure by returning False.
    #     In such a case, this method will be called again to attempt a the reset again.
    #     """
        self.sim.set_state(self.initial_state)
        self.sim.data.qpos[7:10] = site_pos
        self.sim.forward()
        print("object0", np.around(self.sim.data.get_body_xpos('object0'),1))

        return True

if __name__ == "__main__":
    env = WamFindObjEnv()

    o = env.reset()

    while True:
        action = env.action_space.sample()
        # action[3] = 0
        o, r, d, info = env.step(action)
        # print("done", d)
        # print("info", info)
        # print("rew", r)

        # if d:
        #     import IPython; IPython.embed(); import sys; sys.exit(0)


        env.render()
