import numpy as np
import os.path as osp
from gym import utils

from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler
from brl_gym.envs.mujoco.point_mass import PointMassEnv
from brl_gym.envs.mujoco.maze10 import Maze10

from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.envs.mujoco.point_mass import GOAL_POSE as GOAL_POSE4
from brl_gym.envs.mujoco.maze10 import GOAL_POSE as GOAL_POSE10

from brl_gym.estimators.bayes_maze_estimator import BayesMazeEstimator

from gym.spaces import Box, Dict
# from multiprocessing import Pool
from brl_gym.wrapper_envs.util import discount
from brl_gym.envs.mujoco.motion_planner.maze import MotionPlanner
from brl_gym.envs.mujoco.motion_planner.VectorFieldGenerator import VectorField

import cProfile

OBS_DIM = dict()
GOAL_POSE = dict()
ENVS = dict()
ENVS[4] = PointMassEnv
ENVS[10] = Maze10

env4 = PointMassEnv()
OBS_DIM[4] = env4.observation_space.shape[0]
GOAL_POSE[4] = GOAL_POSE4.copy()

env10 = Maze10()
OBS_DIM[10] = env10.observation_space.shape[0]
GOAL_POSE[10] = GOAL_POSE10.copy()

class ExplicitBayesMazeEnv(ExplicitBayesEnv, utils.EzPickle):
    def __init__(self, maze_type=4, reward_entropy=True, reset_params=True, entropy_weight=1.0):

        self.GOAL_POSE = GOAL_POSE[maze_type]
        envs = []
        for i in range(GOAL_POSE[maze_type].shape[0]):
            env = ENVS[maze_type]()
            env.target = i
            envs += [env]

        self.estimator = BayesMazeEstimator(maze_type=maze_type)
        self.env_sampler = DiscreteEnvSampler(envs)
        super(ExplicitBayesMazeEnv, self).__init__(env, self.estimator)
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

        entropy = np.sum(-np.log(bel+1e-5)/np.log(len(bel)) * bel)
        ent_reward = -(entropy - self.prev_entropy)
        self.prev_entropy = entropy
        reward += ent_reward * self.entropy_weight
        info['entropy'] = entropy
        info['bel'] = bel
        self.color_belief(bel)
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

    def color_belief(self, bel):
        for i, b in enumerate(bel.ravel()):
            self.env.model.site_rgba[i, 0] = b
            self.env.model.site_rgba[i, -1] = b



class ExplicitBayesMazeEnvNoEntropyReward(ExplicitBayesMazeEnv):
    def __init__(self):
        super(ExplicitBayesMazeEnvNoEntropyReward, self).__init__(False, True)


class UPMLEMazeEnv(ExplicitBayesEnv, utils.EzPickle):
    def __init__(self, maze_type=4, reward_entropy=True, reset_params=True):

        self.GOAL_POSE = GOAL_POSE[maze_type]
        envs = []
        for i in range(GOAL_POSE[maze_type].shape[0]):
            env = ENVS[maze_type]()
            env.target = i
            envs += [env]

        self.estimator = BayesMazeEstimator(maze_type=maze_type)
        self.env_sampler = DiscreteEnvSampler(envs)
        super(UPMLEMazeEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zparam": self.estimator.param_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params
        self.reward_entropy = reward_entropy

        # Copy the reward entropy
        bayes_env = ExplicitBayesMazeEnv(
            maze_type=maze_type,
            reward_entropy=reward_entropy)
        self.entropy_weight = bayes_env.entropy_weight
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
        reward += ent_reward * self.entropy_weight
        info['entropy'] = entropy
        param = self.estimator.get_mle()
        self.color_mle(param)
        return {'obs':obs, 'zparam':param}, reward, done, info

    def color_mle(self, mle):
        idx = np.argmax(mle)
        for i, o in enumerate(mle):
            self.env.model.site_rgba[i, 0] = 0.0
        # print(idx)
        self.env.model.site_rgba[idx, 0] = 1.0

    def reset(self):
        if self.reset_params:
            self.env = self.env_sampler.sample()
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        self.last_obs = (obs, bel)
        entropy = np.sum(-np.log(bel)/np.log(bel.shape[0]) * bel)
        self.prev_entropy = entropy
        param = self.estimator.get_mle()
        return {'obs':obs, 'zparam':param}


class UPMLEMazeEnvNoEntropyReward(UPMLEMazeEnv):
    def __init__(self):
        super(UPMLEMazeEnvNoEntropyReward, self).__init__(False, True)


class BayesMazeEntropyEnv(ExplicitBayesMazeEnv, utils.EzPickle):
    """
    Environment that provides entropy instead of belief as observation
    """
    def __init__(self, maze_type=4, reward_entropy=True, reset_params=True, observe_entropy=True):
        super(BayesMazeEntropyEnv, self).__init__(
            maze_type=maze_type, reward_entropy=reward_entropy, reset_params=reset_params)
        utils.EzPickle.__init__(self)

        entropy_space = Box(np.array([0.0]), np.array([1.0]))

        self.observe_entropy = observe_entropy
        if observe_entropy:
            self.observation_space = Dict(
                {"obs": ENVS[maze_type].observation_space, "zentropy": entropy_space})
        else:
            self.observation_space = ENVS[maze_type].observation_space

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['bel'] = obs['zbel'].copy()
        del obs['zbel']
        if self.observe_entropy:
            obs['zentropy'] = np.array([info['entropy']])
            return obs, reward, done, info
        else:
            return obs['obs'], reward, done, info


    def reset(self):
        obs = super().reset()
        del obs['zbel']
        if self.observe_entropy:
            obs['zentropy'] = np.array([self.prev_entropy])
            return obs
        else:
            return obs['obs']

# class BayesMazeHiddenEntropyEnv(BayesMazeEntropyEnv):
#     """
#     Hides entropy. Info has everything experts need
#     """
#     def __init__(self, reward_entropy=True):
#         super(BayesMazeHiddenEntropyEnv, self).__init__(reward_param=True,
#             reward_entropy=reward_entropy, observe_entropy=False)
#         self.observation_space = env.observation_space

#     def step(self, action):
#         obs, reward, done, info = super().step(action)
#         return obs['obs'], reward, done, info

#     def reset(self):
#         obs = super().reset()
#         return obs['obs']

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


class Expert:
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
        return action

    def qvals(self, inputs):
        obs = inputs[0].squeeze()
        bel = inputs[1].squeeze()

        return qmdp_expert(obs, bel).squeeze()


if __name__ == "__main__":

    maze_type = 10
    env = ExplicitBayesMazeEnv(reset_params=False, maze_type=maze_type)
    env.env.target = 5
    exp = Expert(nenv=1, maze_type=maze_type)
    all_rewards = []

    # Test expert
    o = env.reset()
    done = False
    val_approx = []
    rewards = []
    t = 0
    while True:
        bel = np.zeros(maze_type)
        bel[env.env.target] = 1

        # action = exp.action(np.concatenate([o['obs'], o['zbel']]).reshape(1,-1))
        action = exp.action(np.concatenate([o['obs'], bel]).reshape(1,-1))
        action = action.squeeze() + np.random.normal()
        print(action)

        o, r, done, _ = env.step(action)
        rewards += [r]
        env.render()

        t += 1
        if done:
            print(t)
            break
        else:
            t += 1

    print(t)
    # rewards = np.array(rewards)
    # # discounted_sum = discount(rewards, gamma)[0]
    # undiscounted_sum = np.sum(rewards)
    # print(' discounted sum', undiscounted_sum)
    # import IPython; IPython.embed();

    # for _ in range(5):
    #     env = ExplicitBayesMazeEnvWithExpert()
    #     o = env.reset()
    #     print(env.env.target)


    # rewards = []
    # for t in range(500):
    #     action = o['expert']
    #     if t < 50:
    #         action[2] = action[2] + np.random.normal()*0.1
    #     o, r, d, _ = env.step(action)
    #     env.render()
    #     print(o['zbel'])
    #     rewards += [r]
    #     if d:
    #         break

    # undiscounted_sum = np.sum(rewards)

    # print('undiscounted sum', undiscounted_sum)
    # import IPython; IPython.embed();

    # Test UPMLE
    # env = UPMLEMazeEnv(maze_type=maze_type)
    # o = env.reset()
    # for _ in range(5000):
    #     o, r, d, info = env.step(env.action_space.sample())
    #     print(o['zparam'], np.around(env.estimator.get_belief(), 2))
    #     env.render()
    # print(env.env.target)

    # Test VF expert
    # env = ExplicitBayesMazeEnv(reset_params=True)
    # # env.env.target = 3
    # o = env.reset()
    # print("env", env.env.target)

    # exp = Expert(nenv=1, use_vf=True)
    # done = False
    # while True:
    #     action = exp.action(
    #         (o['obs'].reshape(1, -1),
    #             o['zbel'].reshape(1, -1))).ravel()
    #     print("Zbel", np.around(o['zbel'], 1), "Action", action)
    #     action[-1] = 1
    #     o, r, d, info = env.step(action.ravel())
    #     env.render()
    #     if d:
    #         break


    # Test entropy-only env
    # maze_type = 10
    # env = BayesMazeEntropyEnv(maze_type=maze_type, reset_params=False)
    # env.env.target = 3
    # expert = Expert(mle=True, maze_type=maze_type)
    # o = env.reset()
    # print(o)
    # info = []
    # for _ in range(500):
    #     o = np.concatenate([o['obs'], o['zentropy']], axis=0).reshape(1, -1)
    #     action = expert.action(o, info).ravel()
    #     action[-1] = 1
    #     o, r, d, info = env.step(action)
    #     info = [info]

    #     env.render()
    import IPython; IPython.embed()