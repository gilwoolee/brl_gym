import numpy as np
import os.path as osp
from gym import utils

from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.envs.mujoco.maze_continuous import MazeContinuous
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.estimators.ekf_maze_goal_estimator import EKFMazeGoalEstimator

from gym.spaces import Box, Dict
from brl_gym.wrapper_envs.util import discount
from brl_gym.envs.mujoco.motion_planner.maze import MotionPlanner
from brl_gym.envs.mujoco.motion_planner.VectorFieldGenerator import VectorField

import cProfile

env = MazeContinuous()

class BayesMazeContinuousEnv(ExplicitBayesEnv, utils.EzPickle):
    def __init__(self, reward_entropy=True, reset_params=True, entropy_weight=1.0, maze_slow=False):

        self.estimator = EKFMazeGoalEstimator()
        env = MazeContinuous()
        super(BayesMazeContinuousEnv, self).__init__(env, self.estimator)

        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zbel": self.estimator.belief_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params
        self.reward_entropy = reward_entropy
        if reward_entropy:
            self.entropy_weight = entropy_weight
        else:
            self.entropy_weight = 0.0
        utils.EzPickle.__init__(self)

    def _update_belief(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        self.estimator.estimate(
                action, obs, **kwargs)
        belief = self.estimator.get_belief()
        return belief, kwargs

    def step(self, action):
        prev_state = self.env.get_state().copy()
        obs, reward, done, info = self.env.step(action)
        info['prev_state'] = prev_state
        info['curr_state'] = self.env.get_state()

        bel, info = self._update_belief(
                                        action,
                                        obs,
                                        **info)

        entropy = bel[-1]
        ent_reward = -(entropy - self.prev_entropy)
        self.prev_entropy = entropy
        reward += ent_reward * self.entropy_weight
        info['entropy'] = entropy
        info['bel'] = bel

        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        self.last_obs = (obs, bel)
        entropy = bel[-1]
        self.prev_entropy = entropy

        return {'obs':obs, 'zbel':bel}


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
        return np.random.normal(size=2) * 0.1

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


def split_inputs(inputs, infos, maze_type=4):
    obs, bel = inputs[:, :-3], inputs[:, -3:]
    return obs, bel


class Expert:
    def __init__(self):
        self.mp = MotionPlanner(maze_type='cont', make_new=False)
        self.maze_type = 'cont'

    def action(self, inputs, infos=[]):
        obs, bel = split_inputs(inputs, infos, maze_type=self.maze_type)
        actions = []
        for o, b in zip(obs, bel):
            actions += [simple_expert_actor(self.mp, o[:2], b)]
        action = np.array(actions).reshape(len(actions), -1)
        action = np.concatenate([action, np.zeros((action.shape[0], 1))], axis=1) * 1.0
        return action + np.random.normal(size=3) * 0.1

    def reset(self):
        self.mp = MotionPlanner(maze_type='cont', make_new=False)


if __name__ == "__main__":

    env = BayesMazeContinuousEnv(reset_params=False, maze_slow=False)
    exp = Expert()
    all_rewards = []

    # Test expert
    o = env.reset()
    done = False
    val_approx = []
    rewards = []
    t = 0
    while True:
        action = exp.action(np.concatenate([o['obs'], o['zbel']]).reshape(1,-1)).ravel()

        o, r, done, _ = env.step(action)
        rewards += [r]
        env.render()

        t += 1
        if done:
            print("done")
            break
        else:
            t += 1

    print(t, r)