import numpy as np
import gym
import tensorflow as tf

from brl_baselines.qmdp import QMDPPolicy, QFunction
from baselines.common.math_util import discount
from brl_gym.envs.tiger import Tiger, Action, TigerLocation

class TigerQFunction(object):
    def __call__(self, tiger_location, action):
        if tiger_location == TigerLocation.LEFT:
            if action == Action.OPEN_LEFT:
                return -10
            elif action == Action.OPEN_RIGHT:
                return 1
            else:
                return (-1 + 0.95 * 10) / 10.0
        else:
            if action == Action.OPEN_LEFT:
                return 1
            elif action == Action.OPEN_RIGHT:
                return -10
            else:
                return (-1 + 0.95 * 10) / 10.0


class TigerQMDPQFunction(QFunction):

    def __init__(self):
        tiger_dim = 1
        qfunc = TigerQFunction()
        self.qvals = []
        for action in np.arange(3):
            self.qvals += [np.array([qfunc(
                    tiger_location, action) for tiger_location in np.arange(2)], dtype=np.float32)]

        self.qvals = np.array(self.qvals).transpose()

    def _eval(self, observation, belief):
        return tf.matmul(belief, self.qvals)

    def step(self, observation, belief=None, **extra_feed):
        winner = tf.argmax(self._eval(observation, belief), 1)
        return winner

    def value(self, observation, belief=None):
        return self._eval(observation, belief)

    def q_value(self, observation, action, belief=None):
        if belief is None:
            belief = observation[-2:]
        return np.sum(belief * self.qvals[action])

    def __call__(self, observation, belief):
        return self._eval(observation, belief)

class TigerQMDPQFunctionNP(QFunction):

    def __init__(self):
        tiger_dim = 1
        qfunc = TigerQFunction()
        self.qvals = []
        for action in np.arange(3):
            self.qvals += [np.array([qfunc(
                    tiger_location, action) for tiger_location in np.arange(2)], dtype=np.float32)]

        self.qvals = np.array(self.qvals).transpose()

    def _eval(self, observation, belief):
        return np.dot(belief, self.qvals)

    def step(self, observation, belief=None, **extra_feed):
        winner = np.argmax(self._eval(observation, belief), 1)
        return winner

    def value(self, observation, belief=None):
        return self._eval(observation, belief)

    def q_value(self, observation, action, belief=None):
        if belief is None:
            belief = observation[-2:]
        return np.sum(belief * self.qvals[action])

    def __call__(self, observation, belief):
        return self._eval(observation, belief)


def main():
    env = gym.make('bayes-tiger-v0')

    agent = TigerQMDPQFunction()

    discounted_rewards = []
    num_iter = 1000
    for _ in range(num_iter):
        o = env.reset()
        done = False
        rewards = []
        while not done:
            action = agent.step(o)
            o, r, done, _ = env.step(action)
            print("Action", action, "Reward", r)
            rewards += [r]
        discounted_rewards += [discount(np.array(rewards), 0.95)[0]]
    print("Average rewards", np.mean(discounted_rewards), np.std(discounted_rewards) / np.sqrt(num_iter))



if __name__ == "__main__":
    main()
