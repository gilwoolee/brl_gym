import numpy as np
import os.path as osp
from gym import utils

from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler
from brl_gym.envs.mujoco.point_mass import PointMassEnv
from brl_gym.envs.mujoco.point_mass_slow import PointMassSlowEnv
from brl_gym.envs.mujoco.maze10 import Maze10
from brl_gym.envs.mujoco.maze10easy import Maze10Easy
from brl_gym.envs.mujoco.maze10easy_slow import Maze10EasySlow

from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.envs.mujoco.point_mass import GOAL_POSE as GOAL_POSE4
from brl_gym.envs.mujoco.maze10 import GOAL_POSE as GOAL_POSE10

from brl_gym.estimators.bayes_maze_estimator import BayesMazeEstimator, LearnableMazeBF

from gym.spaces import Box, Dict
from brl_gym.wrapper_envs.util import discount
from brl_gym.envs.mujoco.motion_planner.maze import MotionPlanner
from brl_gym.envs.mujoco.motion_planner.VectorFieldGenerator import VectorField

from brl_gym.experts.expert import Expert

OBS_DIM = dict()
GOAL_POSE = dict()
ENVS = dict()
ENVS[4] = PointMassEnv
ENVS[10] = Maze10
ENVS[(10, 'easy')] = Maze10Easy
ENVS[(4, 'slow')] = PointMassSlowEnv
ENVS[((10, 'easy'),'slow')] = Maze10EasySlow

env4 = PointMassEnv()
OBS_DIM[4] = env4.observation_space.shape[0]
GOAL_POSE[4] = GOAL_POSE4.copy()
GOAL_POSE[(4, 'slow')] = GOAL_POSE4.copy()

env10 = Maze10()
OBS_DIM[10] = env10.observation_space.shape[0]
GOAL_POSE[10] = GOAL_POSE10.copy()
GOAL_POSE[(10, 'easy')] = GOAL_POSE10.copy()
GOAL_POSE[((10, 'easy'),'slow')] = GOAL_POSE10.copy()


def get_closest_point(waypoints, position):
    if waypoints is None or waypoints is False:
        raise RuntimeError
    waypoints = waypoints.reshape(-1, 2)
    dist = np.linalg.norm(waypoints - position, axis=1)
    idx = np.argmin(dist)
    return idx

def simple_expert_actor(mp, pose, target):
    start = pose[:2]
    waypoints = mp.motion_plan(start, target)

    if not isinstance(waypoints, np.ndarray) and (waypoints == False or waypoints is None):
        return None

    lookahead = 10
    idx = min(get_closest_point(waypoints, pose[:2]) + lookahead, waypoints.shape[0]-1)

    direction = waypoints[idx] - pose[:2]

    direction /= (np.linalg.norm(direction) + 1e-3)


    while True:
        # Add noise
        directions = direction.copy() + np.random.normal(size=(100,2)) * 0.5
        # check for collision
        step_forward = start + directions * 0.05
        idx = np.all(step_forward < 1.5, axis=1)
        step_forward = step_forward[idx]
        directions = directions[idx]
        idx = np.all(step_forward > -1.5, axis=1)
        step_forward = step_forward[idx]
        directions = directions[idx]

        collision_free = mp.state_validity_checker(step_forward, use_sampling_map=True)
        directions = directions[collision_free]

        if len(directions) == 0:
            continue

        # choose the best direction
        dist = np.linalg.norm(direction - directions, axis=1)

        direction = directions[np.argmin(dist)]

        return direction


def simple_combined_expert(mp, s, bel, use_vf, maze_type=4):
    assert use_vf

    actions = []
    # print("bel", np.around(bel,1), "s", s)
    # for s, b in zip(start, bel):
    if not use_vf:
        if (np.any(bel > 0.9)):
            idx = np.argwhere(bel.ravel()>0.9)[0,0]
            action = simple_expert_actor(mp, s, GOAL_POSE[idx])
            if not isinstance(action, np.ndarray) and action == None:
                return np.zeros(2)
                # continue
            else:
                return action

        action = []
        for i, gp in enumerate(GOAL_POSE[maze_type]):
            if not use_vf:
                action += [simple_expert_actor(mp, s, gp)]
            else:
                action += [simple_expert_actor(mp[i], s, gp)]
            if not isinstance(action[-1], np.ndarray) and action[-1] == None:
                return np.zeros(2)

        action = np.array(action)
        return np.sum(action * bel.reshape(-1,1), axis=0)

    else:
        actions = []
        for idx, m in enumerate(mp):
            actions += [m.motion_plan(s, GOAL_POSE[maze_type][idx])]
        actions = np.array(actions)

        actions = actions.transpose(1, 0, 2)
        actions_cp = actions.copy()

        high_belief = bel > 0.9
        bel = bel[:,:,None]

        actions = np.sum(actions * bel, axis=1)
        actions[np.any(high_belief, axis=1), :] = actions_cp[high_belief, :]
        return actions


def split_inputs(inputs, infos, maze_type=4):

    if isinstance(inputs, np.ndarray):
        if inputs.shape[1] == OBS_DIM[maze_type] + GOAL_POSE[maze_type].shape[0]:
            obs, bel = inputs[:, :-GOAL_POSE[maze_type].shape[0]], inputs[:, -GOAL_POSE[maze_type].shape[0]:]
        else:
            obs = inputs
            bel = None
    else:
        if inputs[0].shape[0] > 1:
            obs = inputs[0].squeeze()
            bel = inputs[1].squeeze()
            if len(bel.shape) == 1:
                bel = None # last elt is entropy
        else:
            obs = inputs[0]
            bel = inputs[1]
            if bel.shape[0] == 1:
                bel = None

    if not isinstance(bel, np.ndarray) and bel is None:
        if len(infos) == 0:
            bel = np.ones((obs.shape[0], GOAL_POSE[maze_type].shape[0])) / GOAL_POSE[maze_type].shape[0]
        else:
            bel = np.array([info['bel'] for info in infos])

    return obs, bel


class MazeExpert(Expert):
    def __init__(self, nenv=10, use_vf=True, mle=False, maze_type=4):
        if not use_vf:
            self.mps = [MotionPlanner(maze_type=maze_type) for i in range(nenv)]
        else:
            self.mps = [VectorField(target=i, maze_type=maze_type) for i in range(len(GOAL_POSE[maze_type]))]
        self.use_vf = use_vf
        self.bel_dim = len(GOAL_POSE[maze_type])
        self.mle = mle
        self.maze_type = maze_type

    def action(self, inputs, infos=[]):
        obs, bel = split_inputs(inputs, infos, maze_type=self.maze_type)

        if self.mle:
            mle_indices = np.argmax(bel, axis=1)
            bel_cp = np.zeros(bel.shape)
            bel_cp[tuple(np.array([np.arange(len(mle_indices)), mle_indices]))] = 1.0
            bel = bel_cp

        actions = []
        if not self.use_vf:
            for i, mp in enumerate(self.mps):
                actions += [simple_combined_expert(
                    mp, obs[i].squeeze()[:GOAL_POSE[self.maze_type].shape[1]], bel[i], use_vf=False, maze_type=self.maze_type)]
        else:
            actions = simple_combined_expert(
                self.mps, obs[:, :GOAL_POSE[self.maze_type].shape[1]], bel, use_vf=True, maze_type=self.maze_type)

        action = np.array(actions)
        action = action.reshape(len(actions), -1)

        action = np.concatenate([action, np.zeros((action.shape[0], 1))], axis=1) * 1.0
        return action + np.random.normal(size=3) * 0.1

    def qvals(self, inputs):
        obs = inputs[0].squeeze()
        bel = inputs[1].squeeze()

        return qmdp_expert(obs, bel).squeeze()

