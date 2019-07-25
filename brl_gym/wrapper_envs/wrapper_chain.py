import gym
from gym import utils
import numpy as np
from gym.spaces import Box, Dict

from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.wrapper_envs.mle_env import MLEEnv
from brl_gym.envs import Chain
from brl_gym.envs.chain import Action
from brl_gym.estimators.bayes_chain_estimator import BayesChainEstimator
from brl_gym.wrapper_envs.util import to_one_hot


class BayesChain(BayesEnv):
    def __init__(self, semitied=False):
        env = Chain(semitied=semitied)
        estimator = BayesChainEstimator(np.array([0.1,0.2,0.3,0.4,0.5]), semitied=semitied)
        super(BayesChain, self).__init__(env, estimator)


class MLEChain(MLEEnv):
    def __init__(self, semitied=False):
        env = Chain(semitied=semitied)
        estimator = BayesChainEstimator(np.array([0.1,0.2,0.3,0.4,0.5]), semitied=semitied)
        super(MLEChain, self).__init__(env, estimator)



class ExplicitBayesChain(ExplicitBayesEnv):
    def __init__(self, slip_prob=None, semitied=False):
        if slip_prob is not None:
            env = Chain(semitied=semitied, random_slip_prob=False, slip_prob_a=slip_prob)
        else:
            env = Chain(semitied=semitied)
        estimator = BayesChainEstimator(np.array([0.1,0.2,0.3,0.4,0.5]), semitied=semitied)
        super(ExplicitBayesChain, self).__init__(env, estimator)

        self.n = env.observation_space.n
        obs_space = Box(np.zeros(self.n), np.ones(self.n), dtype=np.float32)
        self.observation_space = Dict({"obs": obs_space, "zbel": estimator.belief_space})
        self.internal_observation_space = obs_space

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
        obs, reward, done, info = self.env.step(action)
        bel, info = self._update_belief(
                                        action,
                                        obs,
                                        **info)

        obs = to_one_hot(obs, self.n)
        self.last_obs = (obs, bel)
        true_prob = self.env.slip_prob[Action.A]
        # print(true_prob)
        info['expert'] = np.where(np.array([0.1,0.2,0.3,0.4,0.5]) == true_prob)[0][0]

        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        obs = to_one_hot(obs, self.n)
        self.last_obs = (obs, bel)
        return {'obs':obs, 'zbel':bel}


def get_value(slip_prob, action):
    if slip_prob < 0.4:
        if action == Action.A:
            return 33
        else:
            return 25
    else:
        if action == Action.A:
            return 20
        else:
            return 20

def collect_batches(n_iterations):
    experiences = []

    for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
        observations = []
        values = []
        new_observations = []
        rewards = []
        dones = []
        actions = []

        for _ in range(n_iterations):
            env = ExplicitBayesChain(slip_prob=i)
            o = env.reset()
            done = False
            while not done:
                # action = Action.OPEN_RIGHT if i == TigerLocation.LEFT else Action.OPEN_LEFT
                # if np.random.rand() < 0.05:
                action = env.action_space.sample()
                values += [get_value(i, action)]
                observations += [np.concatenate([o['obs'], o['zbel']])]
                o, r, done, _ = env.step(action)
                actions += [action]
                new_observations += [np.concatenate([o['obs'], o['zbel']])]
                rewards += [r]
                dones += [done]

        experiences += [(np.array(observations), np.array(values), np.array(actions),
                 np.array(rewards).reshape(1, -1), np.array(new_observations), dones)]

    return experiences
if __name__ == "__main__":

    print("=============== Bayes ===============")
    env = ExplicitBayesChain()
    obs = env.reset()
    print (obs)
    for _ in range(100):
        obs, _, _, info = env.step(0)
        print(obs['obs'], np.around(obs['zbel'], 2), info['expert'])


    # print("=============== Bayes ===============")
    # env = BayesChain()
    # obs = env.reset()
    # print (obs)
    # for _ in range(100):
    #     obs, _, _, info = env.step(0)
    #     print(np.around(obs, 2))


