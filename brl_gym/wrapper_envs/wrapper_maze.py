import numpy as np
import os.path as osp
from gym import utils

from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler
from brl_gym.envs.mujoco.point_mass import PointMassEnv, GOAL_POSE
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv

from brl_gym.estimators.bayes_maze_estimator import BayesMazeEstimator

from gym.spaces import Box, Dict
# from multiprocessing import Pool
from brl_gym.wrapper_envs.util import discount
from brl_gym.envs.mujoco.motion_planner.maze import MotionPlanner

import cProfile


class ExplicitBayesMazeEnv(ExplicitBayesEnv, utils.EzPickle):
    def __init__(self, reward_entropy=True, reset_params=True):

        envs = []
        for i in range(GOAL_POSE.shape[0]):
            env = PointMassEnv()
            env.target = i
            envs += [env]

        self.estimator = BayesMazeEstimator()
        self.env_sampler = DiscreteEnvSampler(envs)
        super(ExplicitBayesMazeEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zbel": self.estimator.belief_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params
        self.reward_entropy = reward_entropy
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

        entropy = np.sum(-np.log(bel+1e-5)/np.log(len(bel)) * bel)
        ent_reward = -(entropy - self.prev_entropy)
        self.prev_entropy = entropy
        if self.reward_entropy:
            reward += ent_reward * 10
        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        if self.reset_params:
            self.env = self.env_sampler.sample()
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        self.last_obs = (obs, bel)
        entropy = np.sum(-np.log(bel)/np.log(bel.shape[0]) * bel)
        self.prev_entropy = entropy
        return {'obs':obs, 'zbel':bel}




class ExplicitBayesMazeEnvNoEntropyReward(ExplicitBayesMazeEnv):
    def __init__(self):
        super(ExplicitBayesMazeEnvNoEntropyReward, self).__init__(False, True)


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

def get_value_approx(pose, target):
    start = pose[:, :2]

    # waypoints = mp.motion_plan(start, target)
    # if not isinstance(waypoints, np.ndarray) and (waypoints == False or waypoints is None):
    #     return np.linalg.norm(pose - target)
    # idx = min(get_closest_point(waypoints, pose[:2]), waypoints.shape[0]-1)

    return -np.linalg.norm(target - start, axis=1)
    # rewards = -np.sum(dist[idx:] * 0.01) + 1


    # return rewards

def qmdp_expert(obs, bel):
    obs = obs.squeeze()
    start = obs[:, :2]

    # values = []

    # for s, b in zip(start, bel):

        # vals = [np.array(get_value_approx(mp, s, gp)) for gp in GOAL_POSE]
        # vals = np.array(vals).transpose()
        # values += np.sum(vals * bel)

    vals = [np.array(get_value_approx(start, gp)) for gp in GOAL_POSE]
    vals = np.array(vals).transpose()
    return np.sum(vals * bel, axis=1)

def simple_combined_expert(mp, obs, bel):
    obs = obs.squeeze()
    s = obs[:GOAL_POSE.shape[0]]
    b = bel
    actions = []

    # for s, b in zip(start, bel):
    if (np.any(bel > 0.9)):
        idx = np.argwhere(bel.ravel()>0.9)[0,0]
        action = simple_expert_actor(mp, s, GOAL_POSE[idx])
        if not isinstance(action, np.ndarray) and action == None:
            # print("noise")
            return np.zeros(2)
            # continue
        else:
            return action
    action = []
    for gp in GOAL_POSE:
        action += [simple_expert_actor(mp, s, gp)]
        if not isinstance(action[-1], np.ndarray) and action[-1] == None:
            return np.zeros(2)

    action = np.array(action)

    # if np.any(action == None):
    #     return np.zeros(2) #random.normal(size=2)*0.05

    return np.sum(action * b.reshape(-1,1), axis=0)

class Expert:
    def __init__(self, nenv=10):
        self.mps = [MotionPlanner() for i in range(nenv)]
        self.bel_dim = len(GOAL_POSE)

    def action(self, inputs):
        if isinstance(inputs, np.ndarray):
            obs, bel = inputs[:, :-GOAL_POSE.shape[0]], inputs[:, -GOAL_POSE.shape[0]:]
        else:
            if inputs[0].shape[0] > 1:
                obs = inputs[0].squeeze()
                bel = inputs[1].squeeze()
            else:
                obs = inputs[0]
                bel = inputs[1]

        actions = []
        for i, mp in enumerate(self.mps):
            actions += [simple_combined_expert(mp, obs[i], bel[i])]

        action = np.array(actions)
        if obs.shape[0] >1:
            action = action.squeeze()

        action = np.concatenate([action, np.zeros((action.shape[0], 1))], axis=1) * 0.2
        return action

    def qvals(self, inputs):

        obs = inputs[0].squeeze()
        bel = inputs[1].squeeze()

        return qmdp_expert(obs, bel).squeeze()

def trial_combined_expert(i):

    gamma = 0.999
    env = ExplicitBayesMazeEnv(reset_params=True)
    exp = Expert(nenv=1)
    all_rewards = []

    o = env.reset()
    print(env.env.target)

    done = False
    # profile = cProfile.Profile()
    # profile.enable()
    val_approx = []
    rewards = []
    for t in range(400):
        # val_approx += [get_value_approx(o['obs'].reshape(1, -1), friction)]
        inp = np.concatenate([o['obs'].ravel(), o['zbel'].ravel()]).reshape(1, -1)
        action = exp.action(inp)
        # action = simple_expert_actor(exp.mp, o['obs'], GOAL_POSE[env.env.target])
        print(action)
        # action[2] = np.random.normal()*0.1
        o, r, done, _ = env.step(action)
        rewards += [r]

        # env.render()

        # input('continue')
        print(np.around(o['zbel'],2))

    # profile.disable()
    # profile.print_stats()
    rewards = np.array(rewards)
    discounted_sum = discount(rewards, gamma)[0]
    undiscounted_sum = np.sum(rewards)
    print(' discounted sum', discounted_sum)
    # import IPython; IPython.embed();
    return discounted_sum, undiscounted_sum

class ExplicitBayesMazeEnvWithExpert(ExplicitBayesEnv, utils.EzPickle):
    """
    Expert proposal is part of the observation
    """
    def __init__(self, reward_entropy=False, reset_params=True):

        envs = []
        for i in range(GOAL_POSE.shape[0]):
            env = PointMassEnv()
            env.target = i
            envs += [env]

        self.estimator = BayesMazeEstimator()
        self.env_sampler = DiscreteEnvSampler(envs)
        super(ExplicitBayesMazeEnvWithExpert, self).__init__(env, self.estimator)
        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zbel": self.estimator.belief_space, "expert": env.action_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params
        self.reward_entropy = reward_entropy
        self.expert = Expert(nenv=1)
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

        entropy = np.sum(-np.log(bel+1e-5)/np.log(len(bel)) * bel)
        ent_reward = -(entropy - self.prev_entropy)
        self.prev_entropy = entropy
        if self.reward_entropy:
            reward += ent_reward * 10

        expert_action = self.expert.action(np.concatenate([obs, bel]).reshape(1, -1)).ravel()

        return {'obs':obs, 'zbel':bel, "expert":expert_action}, reward, done, info

    def reset(self):
        if self.reset_params:
            self.env = self.env_sampler.sample()
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        self.last_obs = (obs, bel)
        entropy = np.sum(-np.log(bel)/np.log(bel.shape[0]) * bel)
        self.prev_entropy = entropy
        expert_action = self.expert.action(np.concatenate([obs, bel]).reshape(1, -1)).ravel()

        return {'obs':obs, 'zbel':bel, 'expert': expert_action}

if __name__ == "__main__":

    # gamma = 0.999
    # env = ExplicitBayesMazeEnv(reset_params=True)
    # # env.env.target = 4
    # exp = Expert(nenv=1)
    # all_rewards = []

    # o = env.reset()
    # print(env.env.target)

    # done = False

    # val_approx = []
    # rewards = []
    # for t in range(250):
    #     # val_approx += [get_value_approx(o['obs'].reshape(1, -1), friction)]
    #     action = exp.action(np.concatenate([o['obs'], o['zbel']]).reshape(1,-1))
    #     # action = simple_expert_actor(exp.mp, o['obs'], GOAL_POSE[env.env.target])
    #     # print("action", action)
    #     action = action.squeeze()
    #     if t < 50:
    #         action[2] = action[2] + np.random.normal()*0.1
    #     o, r, done, _ = env.step(action)
    #     rewards += [r]
    #     #print(np.around(o['zbel'],2))

    #     env.render()

    #     # input('continue')
    #     # print(np.around(o['zbel'],2))

    # rewards = np.array(rewards)
    # # discounted_sum = discount(rewards, gamma)[0]
    # undiscounted_sum = np.sum(rewards)
    # print(' discounted sum', undiscounted_sum)
    # import IPython; IPython.embed();


    env = ExplicitBayesMazeEnvWithExpert()
    o = env.reset()

    rewards = []
    for t in range(350):
        action = o['expert']
        if t < 50:
            action[2] = action[2] + np.random.normal()*0.1
        o, r, d, _ = env.step(action)
        env.render()
        print(o['zbel'])
        rewards += [r]
        if d:
            break

    undiscounted_sum = np.sum(rewards)

    print('undiscounted sum', undiscounted_sum)
    import IPython; IPython.embed();
