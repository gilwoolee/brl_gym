import numpy as np
from brl_gym.estimators.bayes_doors_estimator import BayesDoorsEstimator #, LearnableDoorsBF
from brl_gym.envs.mujoco.doors import DoorsEnv
from brl_gym.envs.mujoco.doors_slow import DoorsSlowEnv
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler

from gym.spaces import Box, Dict
from gym import utils
# Divide regions into 4 regions, L0, L1, L2, L3 from left to right
REGIONS = [0, 1, 2, 3]
CLOSEST_DOORS = {0:dict(), 1:dict(), 2:dict(), 3:dict()}

def map_to_region(xs):
    regions = np.zeros(xs.shape[0])
    regions[xs <= -0.7] = 0
    regions[np.logical_and(xs <=0, xs > -0.7)] = 1
    regions[np.logical_and(xs > 0, xs <= 0.7)] = 2
    regions[xs >= 0.7] = 3
    return regions.astype(np.int)

CASES = ['{{:0{}b}}'.format(4).format(x) \
                      for x in range(1, 16)]
for binary in CASES:
    # L0
    if binary[0] == '1':
        CLOSEST_DOORS[0][binary] = 0
    elif binary[:2] == '01':
        CLOSEST_DOORS[0][binary] = 1
    elif binary[:3] == '001':
        CLOSEST_DOORS[0][binary] = 2
    elif binary == '0001':
        CLOSEST_DOORS[0][binary] = 3

    # L3
    flip = binary[::-1]
    if binary[0] == '1':
        CLOSEST_DOORS[3][binary] = 0
    elif binary[:2] == '01':
        CLOSEST_DOORS[3][binary] = 1
    elif binary[:3] == '001':
        CLOSEST_DOORS[3][binary] = 2
    elif binary == '0001':
        CLOSEST_DOORS[3][binary] = 3

    # L1
    if binary[1] == '1':
        CLOSEST_DOORS[1][binary] = 1
    elif binary[:2] == '10':
        CLOSEST_DOORS[1][binary] = 0
    elif binary[:3] == '001':
        CLOSEST_DOORS[1][binary] = 2
    else:
        CLOSEST_DOORS[1][binary] = 3

    # L2
    if binary[2] == '1':
        CLOSEST_DOORS[2][binary] = 2
    elif binary[1:3] == '10':
        CLOSEST_DOORS[2][binary] = 1
    elif binary[1:] == '001':
        CLOSEST_DOORS[2][binary] = 3
    else:
        CLOSEST_DOORS[2][binary] = 0


def get_closest_door(open_doors, states):
    region = map_to_region(states[:, 0])
    closest_doors = np.zeros(states.shape[0], dtype=np.int)

    for i, binary in enumerate(open_doors):
        closest_doors[i] = CLOSEST_DOORS[region[i]][binary]
    return closest_doors


class SimpleExpert:
    def __init__(self):
        env = DoorsEnv()
        self.target_pos = np.array([0.0, 1.2])
        self.door_pos = env.door_pos[:, :2]
        self.door_pos[:, 1] = 0.25

    def action(self, open_doors, states):
        binary = []
        if len(open_doors.shape) == 2:
            for x in open_doors:
                binary += [''.join(str(int(y)) for y in x)]
        else:
            binary = [CASES[x] for x in open_doors]
        open_doors = binary

        actions = np.zeros((states.shape[0], 2))

        # door_pos = self.door_pos[open_doors]
        target_pos = self.target_pos

        # If the agent is above the bar, go straight to the goal
        # if state[1] >= door_pos[0, 1]:
        direction_to_target = target_pos - states
        above_bar = states[:, 1] >= 0.25 # bar colliding height self.door_pos[0, 1]
        actions[above_bar] = direction_to_target[above_bar]
        door = get_closest_door(open_doors, states)

        direction_to_door = self.door_pos[door] - states

        actions[np.logical_not(above_bar)] = direction_to_door[np.logical_not(above_bar)]

        actions = (actions / np.linalg.norm(actions, axis=1).reshape(-1, 1)) * 0.1

        return actions

env = DoorsEnv()
OBS_DIM = env.observation_space.shape[0]
estimator = BayesDoorsEstimator()

def split_inputs(inputs, infos):
    if isinstance(inputs, np.ndarray):
        if inputs.shape[1] == OBS_DIM + 16:
            obs, bel = inputs[:, :-16], inputs[:, -16:]
            num_inputs = inputs.shape[0]
        elif inputs.shape[1] == OBS_DIM + 4:
            obs, bel = inputs[:, :-4], inputs[:, -4:]
            bel = np.array([estimator.flatten_to_belief(x) for x in bel])
            num_inputs = inputs.shape[0]
        else:
            obs = inputs
            num_inputs = inputs.shape[0]
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
                bel = None # last elt is entropy
        num_inputs = inputs[0].shape[0]

    if not isinstance(bel, np.ndarray) and bel is None:
        if len(infos) == 0:
            bel = np.ones((obs.shape[0], 16))
        else:
            bel = np.array([info['bel'] for info in infos])

    return obs, bel, num_inputs


class DoorsExpert:
    def __init__(self, mle=False):
        env = DoorsEnv()
        self.target_pos = env.data.site_xpos[env.target_sid].ravel()[:2]
        self.door_pos = env.door_pos[:, :2]
        self.simple_expert = SimpleExpert()
        self.mle = mle

    def action(self, inputs, infos=[]):
        door_pos = self.door_pos
        target_pos = self.target_pos
        obs, bel, num_inputs = split_inputs(inputs, infos)

        if self.mle:
            mle_indices = np.argmax(bel, axis=1)
            bel_cp = np.zeros(bel.shape)
            bel_cp[tuple(np.array([np.arange(len(mle_indices)), mle_indices]))] = 1.0
            bel = bel_cp

        actions = np.zeros((num_inputs, 2))

        for i, case in enumerate(np.arange(15)):
            c = np.array([case] * obs.shape[0])
            proposal = self.simple_expert.action(c, obs[:, :2])
            actions += proposal *  bel[:, [i+1]]

        actions = np.concatenate([actions, np.zeros((actions.shape[0], 1))], axis=1)
        actions += np.random.normal(size=3)*0.1
        return actions

    def __call__(self, inputs, infos=[]):
        return self.action(inputs)

if __name__ == "__main__":
    from brl_gym.wrapper_envs.wrapper_doors import ExplicitBayesDoorsEnv
    env = ExplicitBayesDoorsEnv()
    expert = DoorsExpert()

    rewards = np.zeros(100)
    for i in range(100):
        obs = env.reset()
        t = 0
        done = False
        while not done:
            action = expert(obs.reshape(1,-1))
            obs, reward, done, info = env.step(action.ravel())
            rewards[i] += reward
            t += 1
            if t >= 300:
                break
        print(rewards[i])

    print(np.mean(rewards))
