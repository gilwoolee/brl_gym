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
# from multiprocessing import Pool
from brl_gym.wrapper_envs.util import discount
from brl_gym.envs.mujoco.motion_planner.maze import MotionPlanner
from brl_gym.envs.mujoco.motion_planner.VectorFieldGenerator import VectorField



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

class ExplicitBayesMazeEnv(ExplicitBayesEnv, utils.EzPickle):
    def __init__(self, maze_type=4, reward_entropy=True, reset_params=True, entropy_weight=1.0,
        difficulty='hard', maze_slow=False, learnable_bf=False):

        if difficulty == 'easy':
            maze_type = (maze_type, 'easy')

        if maze_slow:
            maze_type = (maze_type, 'slow')

        self.GOAL_POSE = GOAL_POSE[maze_type]
        envs = []
        for i in range(GOAL_POSE[maze_type].shape[0]):
            env = ENVS[maze_type]()
            env.target = i
            envs += [env]

        if not learnable_bf:
            self.estimator = BayesMazeEstimator(maze_type=maze_type)
        else:
            self.estimator = LearnableMazeBF(maze_type=maze_type)

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


    def set_bayes_filter(self, file):
        self.estimator.set_bayes_filter(file)

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
        info['done'] = done
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
        info['label'] = self.env.target
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
        self.color_belief(bel)
        return {'obs':obs, 'zbel':bel}

    def color_belief(self, bel):
        for i, b in enumerate(bel.ravel()):
            self.env.model.site_rgba[i, 0] = 1.0
            self.env.model.site_rgba[i, -1] = b



class ExplicitBayesMazeEnvNoEntropyReward(ExplicitBayesMazeEnv):
    def __init__(self):
        super(ExplicitBayesMazeEnvNoEntropyReward, self).__init__(False, True)


class UPMLEMazeEnv(ExplicitBayesEnv, utils.EzPickle):
    def __init__(self, maze_type=4,
        reward_entropy=True, reset_params=True,
        entropy_weight=None,
        difficulty='hard'):

        if difficulty == 'easy':
            maze_type = (maze_type, 'easy')

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
        if entropy_weight is None:
            bayes_env = ExplicitBayesMazeEnv(
                maze_type=maze_type,
                reward_entropy=reward_entropy)
            self.entropy_weight = bayes_env.entropy_weight
        else:
            self.entropy_weight = entropy_weight
        print("UPMLE Entropy weight", self.entropy_weight)
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
    def __init__(self, maze_type=4, reward_entropy=True, reset_params=True, observe_entropy=True, entropy_weight=1.0,
        difficulty='hard'):

        super(BayesMazeEntropyEnv, self).__init__(
            maze_type=maze_type, reward_entropy=reward_entropy,
            reset_params=reset_params, entropy_weight=entropy_weight,
            difficulty=difficulty)
        utils.EzPickle.__init__(self)

        entropy_space = Box(np.array([0.0]), np.array([1.0]))

        if difficulty == 'easy':
            maze_type = (maze_type, 'easy')

        env = ENVS[maze_type]()
        self.observe_entropy = observe_entropy
        if observe_entropy:
            self.observation_space = Dict(
                {"obs": env.observation_space, "zentropy": entropy_space})
        else:
            self.observation_space = env.observation_space

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


if __name__ == "__main__":

    from brl_gym.experts.maze.expert import MazeExpert as Expert

    maze_type = 10
    env = ExplicitBayesMazeEnv(reset_params=False,
        maze_type=maze_type, difficulty="easy",
        maze_slow=True)
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
        bel[env.env.target] = 5

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

    print(t, r)
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
