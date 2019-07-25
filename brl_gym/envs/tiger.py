import numpy as np

import gym
from gym.spaces import Discrete
from gym import utils
from gym.utils import seeding

class State:
    CLOSED = 0 # Initial state

class Sound:
    SOUND_LEFT = 0
    SOUND_RIGHT = 1
    SOUND_NONE = 2

class Action:
    LISTEN = 0
    OPEN_LEFT = 1
    OPEN_RIGHT = 2

class TigerLocation:
    LEFT = 0
    RIGHT = 1

ACTION_NAME = {
    None: "NONE",
    Action.LISTEN: "LISTEN    ",
    Action.OPEN_LEFT: "OPEN_LEFT",
    Action.OPEN_RIGHT: "OPEN_RIGHT"
    }

OBS_NAME = {
    Sound.SOUND_NONE: "NONE",
    Sound.SOUND_LEFT: "LEFT",
    Sound.SOUND_RIGHT: "RIGHT"
    }

class Tiger(gym.Env, utils.EzPickle):
    def __init__(self, tiger=None, discount=0.95):
        self.nS = 3
        self.nA = 3
        self.obs_error = 0.15
        self.start_tiger = tiger
        self.discount = discount
        self.continuous = False
        self.tiger_left_init_prob = 0.5

        self.action_space = Discrete(self.nA)
        self.observation_space = Discrete(self.nS)

        self.seed()
        self.reset()

    def render(self, mode='human'):
        print("")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, randomize=False):
        if self.start_tiger is None:
            r = self.np_random.uniform(0, 1)
            if r <= self.tiger_left_init_prob:
                self.tiger = TigerLocation.LEFT
            else:
                self.tiger = TigerLocation.RIGHT
        else:
            self.tiger = self.start_tiger
        # self.tiger = TigerLocation.RIGHT
        self.state = State.CLOSED
        return Sound.SOUND_NONE

    def step(self, action):
        reward = -1
        done = False
        obs = Sound.SOUND_NONE
        obs_error = False
        if action == Action.LISTEN:
            r = self.np_random.uniform(0, 1)
            if r <= self.obs_error:
                obs_error = True
                if self.tiger == TigerLocation.LEFT:
                    obs = Sound.SOUND_RIGHT
                else:
                    obs = Sound.SOUND_LEFT
            else:
                if self.tiger == TigerLocation.LEFT:
                    obs = Sound.SOUND_LEFT
                else:
                    obs = Sound.SOUND_RIGHT
            done = False
        elif action == Action.OPEN_LEFT:
            if self.tiger == TigerLocation.LEFT:
                reward = -100
            else:
                reward = 10
            done = True
        else:
            if self.tiger == TigerLocation.RIGHT:
                reward = -100
            else:
                reward = 10
            done = True

        reward /= 10
        return obs, reward, done, {'sound': obs}

