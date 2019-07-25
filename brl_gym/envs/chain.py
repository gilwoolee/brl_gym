import numpy as np

import gym
from gym.spaces import Discrete
from gym import utils
from gym.utils import seeding

class State:
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class Action:
    A = 0
    B = 1


class Chain(gym.Env, utils.EzPickle):
    def __init__(self, random_slip_prob=True,
        slip_prob_a=0.2, slip_prob_b=0.2,
        slip_prob_low=0.0, slip_prob_high=1.0, semitied=False):
        self.nS = 5
        self.nA = 2
        self.semitied = semitied

        self.slip_prob = dict()

        # Default
        self.slip_prob[Action.A] = slip_prob_a
        self.slip_prob[Action.B] = slip_prob_b

        self.random_slip_prob = random_slip_prob
        self.slip_prob_low = slip_prob_low
        self.slip_prob_high = slip_prob_high

        self.action_space = Discrete(self.nA)
        self.observation_space = Discrete(self.nS)

        self.horizon = 100
        self.t = 0

        self.seed()
        self.reset()

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.t = 0
        if self.random_slip_prob:
        #     if self.semitied:
        #         for key in self.slip_prob:
        #             self.slip_prob[key] = self.np_random.uniform(
        #             self.slip_prob_low, self.slip_prob_high)
        #     else:
        #         prob = self.np_random.uniform(
        #             self.slip_prob_low, self.slip_prob_high)
        #         for key in self.slip_prob:
        #             self.slip_prob[key] = prob
            slip_prob = np.random.choice(np.array([0.1,0.2,0.3,0.4,0.5]))
            for key in self.slip_prob:
                self.slip_prob[key] = slip_prob

        # for key in self.slip_prob:
        #     self.slip_prob[key] = 0.2

        self.state = State.ZERO
        return self.state

    def step(self, action):
        self.t += 1
        if self.t == self.horizon:
            done = True
        else:
            done = False

        if action != Action.A and action != Action.B:
            raise ValueError("Unknown action {}".format(action))

        r = self.np_random.uniform(0, 1)
        slip = False
        if action == Action.A:
            if r <= self.slip_prob[Action.A]:
                action = Action.B
                slip = True
        elif action == Action.B:
            if r < self.slip_prob[Action.B]:
                action = Action.A
                slip = True
        else:
            raise ValueError

        if action == Action.A:
            reward = 0
            state = self.state + 1
            if state == self.nS:
                state = State.FOUR
                reward = 1
        elif action == Action.B:
            reward = 0.2
            state = State.ZERO

        self.state = state
        return state, reward, done, {"slip":slip}

if __name__ == "__main__":
    env = Chain()
    state = env.reset()
    import IPython; IPython.embed()
