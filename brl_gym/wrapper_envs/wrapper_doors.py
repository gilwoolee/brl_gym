import numpy as np
from brl_gym.estimators.bayes_doors_estimator import BayesDoorsEstimator
from brl_gym.envs.mujoco.doors import DoorsEnv
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler

from gym.spaces import Box, Dict
from gym import utils

class ExplicitBayesDoorsEnv(ExplicitBayesEnv, utils.EzPickle):
    def __init__(self, reset_params=True):

        self.num_doors = 4
        self.num_cases = 2**self.num_doors
        self.cases =  ['{{:0{}b}}'.format(self.num_doors).format(x) \
                      for x in range(self.num_cases)]
        self.cases_np = [np.array([int(x) for x in case]) for case in self.cases]

        envs = []
        for case in self.cases_np:
            env = DoorsEnv()
            env.open_doors = case.astype(np.bool)
            envs += [env]

        self.estimator = BayesDoorsEstimator()

        self.env_sampler = DiscreteEnvSampler(envs)
        super(ExplicitBayesDoorsEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zbel": self.estimator.belief_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params
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

        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        if self.reset_params:
            while True:
                self.env = self.env_sampler.sample()
                if not np.all(self.env.open_doors == False):
                    break

        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)

        return {'obs':obs, 'zbel':bel}


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
        self.target_pos = env.data.site_xpos[env.target_sid].ravel()[:2]
        self.door_pos = env.door_pos[:, :2]
        self.door_pos[:, 1] = 0.21

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
        direction_to_bar = target_pos - states
        above_bar = states[:, 1] >= 0.225 # bar colliding height self.door_pos[0, 1]
        actions[above_bar] = direction_to_bar[above_bar]

        door = get_closest_door(open_doors, states)

        direction_to_door = self.door_pos[door] - states
        actions[np.logical_not(above_bar)] = direction_to_door[np.logical_not(above_bar)]

        actions = (actions / np.linalg.norm(actions, axis=1))  * 0.1
        # actions[actions[:, 1] < 0.05] = 0.05
        return actions


class Expert:
    def __init__(self):
        env = DoorsEnv()
        self.target_pos = env.data.site_xpos[env.target_sid].ravel()[:2]
        self.door_pos = env.door_pos[:, :2]
        self.simple_expert = SimpleExpert()

    def action(self, inputs):
        door_pos = self.door_pos
        target_pos = self.target_pos
        if isinstance(inputs, np.ndarray):
            obs, bel = inputs[:, :-16], inputs[:, -16:]
            num_inputs = inputs.shape[0]
        else:
            if inputs[0].shape[0] > 1:
                obs = inputs[0].squeeze()
                bel = inputs[1].squeeze()
            else:
                obs = inputs[0]
                bel = inputs[1]
            num_inputs = inputs[0].shape[0]

        actions = np.zeros((num_inputs, 2))

        for i, case in enumerate(np.arange(15)):
            c = np.array([case] * obs.shape[0])
            proposal = self.simple_expert.action(c, obs[:, :2])
            # print(CASES[case], "proposal", proposal, bel[:, i+1])
            actions += proposal *  bel[:, i+1]

        return actions

if __name__ == "__main__":
    # Test simple experts
    # env = ExplicitBayesDoorsEnv()
    # obs = env.reset()
    # doors = env.env.open_doors
    # simple_expert = SimpleExpert()

    # done = False
    # while not done:
    #     action = simple_expert.action(doors.reshape(1, -1), obs['obs'][:2].reshape(1, -1))
    #     obs, _, done, _ = env.step(action[0])
    #     env.render()

    # # Test expert
    env = ExplicitBayesDoorsEnv()
    obs = env.reset()
    doors = env.env.open_doors
    expert = Expert()

    done = False
    rewards = []
    while not done:
        action = expert.action((obs['obs'].reshape(1, -1), obs['zbel'].reshape(1, -1)))
        print('obs', np.around(obs['obs'][:2], 2), 'act', action, 'zbel', obs['zbel'])
        obs, r, done, _ = env.step(action)

        env.render()

        rewards += [r]
        if done:
            break

    print(np.sum(rewards))