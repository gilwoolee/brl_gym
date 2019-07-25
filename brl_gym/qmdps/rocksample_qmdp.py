import gym
import yaml
import glob
import os
import numpy as np
from multiprocessing import Pool

from baselines.common.cmd_util import make_vec_env, make_env

from brl_baselines.qmdp import QMDPPolicy, QFunction
from brl_gym.envs.rocksample import RockSample, load_env

from baselines.common.math_util import discount

import cProfile


class RockSampleVFunction(object):
    def __init__(self, solution_file):
        self.init_rock_state = [int(x) for x in solution_file.split("_")[-1].replace(".yaml", "")]
        with open(solution_file) as f:
            self.solution = yaml.load(f, Loader=yaml.Loader)

    def __call__(self, state):
        return self.solution[tuple(state)]['discounted_return']

def load_solutions(rocksample, cases, states, num_envs, nominal_env, solution_dir="/home/gilwoo/rocksample_solutions/rocksample7x8/values/"):
    qfuncs = np.zeros((len(cases), len(states) + 1))
    # envs = dict()
    rockstates = []
    for i, case in enumerate(cases):
        files = glob.glob(
            os.path.join(solution_dir, "{}_{}.yaml".format(rocksample, case)))
        assert len(files) == 1
        print(case, files[0])
        case = tuple([int(x) for x in case])

        qfunc = RockSampleVFunction(files[0])

        for s in states:
            qfuncs[i, s[0] * 7 + s[1]] = qfunc(s)

        rockstates += [[int(x) for x in case]] * num_envs

    # env = make_vec_env('explicit-bayes-rocksample-v0', 'wrapper_envs', num_envs * len(cases), seed=None,
    #     rockstates=rockstates)

    return qfuncs

class RockSampleQMDPQFunction(QFunction):
    def __init__(self, num_rocks, num_envs):
        if num_rocks == 4:
            rocksample = "rocksample7x4"
            solution_dir = "/home/gilwoo/rocksample_solutions/rocksample7x4/values/"
        else:
            rocksample = "rocksample7x8"
            solution_dir = "/home/gilwoo/rocksample_solutions/rocksample7x8/values/"

        self.nominal_env = load_env(("{}.json".format(rocksample)))

        self.transition = np.zeros((49, 4, 2))
        self.states = []
        for i in range(7):
            for j in range(7):
                coords = [i, j]
                self.states += [tuple(coords)]
                x = i * 7 + j
                for a in range(4):
                    o, r, d, _, state = self.nominal_env.reset_state_and_step(coords, a)
                    if d:
                        y = -1
                    else:
                        next_coords = state[:2]
                        y = next_coords[0] * 7 + next_coords[1]
                    self.transition[x][a][0] = y
                    self.transition[x][a][1] = r

        self.grid_size = 7
        self.obs_dim = -num_rocks
        self.num_rocks = num_rocks
        self.num_cases = 2**self.num_rocks
        self.cases = ['{{:0{}b}}'.format(self.num_rocks).format(x) \
                      for x in range(self.num_cases)]
        self.cases_np = [np.array([int(x) for x in case]) for case in self.cases]

        # self.sampling_transition = np.ones((49, self.num_cases)) * -100
        self.rps = np.array(self.nominal_env.rock_positions)
        self.rps_states = []
        for rp in self.rps:
            self.rps_states += [rp[0] * 7 + rp[1]]
        self.rps_states = np.array(self.rps_states)
        self.cps = np.array(self.cases_np)
        # for i in range(7):
        #     for j in range(7):
        #         coords = np.array([i, j])
        #         x = i * 7 + j
        #         if np.any(np.all(coords == rps, axis=1)):
        #             rp = np.argwhere(np.all(coords == rps, axis=1))[0, 0]
        #             self.sampling_transition[x, cps[:, rp] == 1] = 1
        #             self.sampling_transition[x, cps[:, rp] == 0] = -1

        self.vfuncs = load_solutions(rocksample, self.cases, self.states, num_envs, self.nominal_env, solution_dir)
        self.cases = [tuple([int(x) for x in case]) for case in self.cases]

        self.action_space = self.nominal_env.action_space
        self.num_envs = num_envs
        self.vfunc_list = []
        for i in range(self.num_cases):
            self.vfunc_list += self.num_envs * [i]

    def _eval(self, observation, belief, **extra_feed):
        # Belief approximation
        belief[belief <= 0.05] = 0.0
        belief[belief >= 0.95] = 1.0

        expert_i = None if 'expert' not in extra_feed else extra_feed['expert']

        coords = self.nominal_env._decode_observations(np.argmax(observation, 1))
        coords = np.tile(coords, (self.num_cases, 1))
        states = coords[:, 0] * self.grid_size + coords[:, 1]

        values = np.zeros((self.num_envs, self.action_space.n))
        expanded_belief = self.flatten_to_belief(belief)

        # Sensings have the same value
        x = tuple(np.asarray([self.vfunc_list, states]))
        next_values_for_sensing = self.vfuncs[x].reshape((self.num_cases, self.num_envs)).transpose()
        if not expert_i:
            sensing_val = np.sum(expanded_belief * next_values_for_sensing.transpose(), axis=0)
        else:
            sensing_val = next_values_for_sensing[:, expert_i]
        values[:, 5:] = 0.95 * np.tile(sensing_val, (self.num_rocks, 1)).transpose()

        # look up transitions
        for a in range(4):
            t = self.transition[tuple(np.asarray([states, [a] * states.shape[0]]))]
            next_states = t[:, 0].astype(np.int)
            r = t[:, 1].reshape((self.num_cases, self.num_envs))
            next_states[next_states == -1] = 49 # zeros
            x = tuple(np.asarray([self.vfunc_list, next_states]))
            next_values = 0.95 * self.vfuncs[x].reshape((self.num_cases, self.num_envs))
            if not expert_i:
                val = np.sum(expanded_belief * r, axis=0) + np.sum(expanded_belief * next_values, axis=0)
            else:
                val = r[expert_i] + next_values[expert_i]
            values[:, a] = val
            # if val == 9.5:
            #     import IPython; IPython.embed(); import sys; sys.exit(0)
        # sampling transtions
        coords = self.nominal_env._decode_observations(np.argmax(observation, 1))
        states = coords[:, 0] * self.grid_size + coords[:, 1]
        for i, s in enumerate(states):
            if np.any(self.rps_states == s):
                idx = np.argwhere(self.rps_states == s)[0,0]
                next_belief = belief[i].copy()
                next_belief[idx] = 0
                expanded_next_belief = self.flatten_to_belief(next_belief.reshape(1, -1))
                if not expert_i:
                    expected_r = belief[i, idx] * 10 + (1 - belief[i, idx]) * -10
                    values[i, 4] = expected_r + 0.95 * np.sum(expanded_next_belief.ravel() * next_values_for_sensing[i].ravel())
                else:
                    if self.cases_np[expert_i][idx] == 1:
                        values[i, 4] = 0.95 * next_values_for_sensing[:, expert_i]
                    else:
                        values[i, 4] = -10 + 0.95 * next_values_for_sensing[:, expert_i]
            else:
                values[i, 4] = 0

        known_indices = np.logical_or(belief >= 0.95 , belief <= 0.05)
        sampled_indices = np.hstack([np.zeros((belief.shape[0], 5)), known_indices]).astype(np.bool)
        values[sampled_indices] = 0

        return values

    def step(self, observation, **extra_feed):
        obs, belief = observation[:-self.num_rocks], observation[-self.num_rocks:]
        values = self._eval(obs.reshape(1, -1), belief.reshape(1, -1)).ravel()
        winner = np.argwhere(values == np.max(values)).flatten()
        # import IPython; IPython.embed(); import sys; sys.exit(0)
        action = np.random.choice(winner)
        return action

    def value(self, observation, **extra_feed):
        return self._eval(observation, **extra_feed)

    def __call__(self, observation, belief, **extra_feed):
        return self._eval(observation, belief, **extra_feed)


    def flatten_to_belief(self, belief_per_rock, approximate=False):

        if approximate:
            belief_per_rock[belief_per_rock <= 0.05] = 0.0
            belief_per_rock[belief_per_rock >= 0.95] = 1.0

        belief = []

        # Bernoulli distribution for each rock
        good_probs = [belief_per_rock ** case * (1 - belief_per_rock) ** (1 - case) for case in self.cases_np]

        # Each rock is independent, so multiply the probabilities
        belief = np.prod(np.array(good_probs), axis=2)

        # Should already sum to 1 but doing this just for numerical stability
        belief /= np.sum(belief, axis=0)

        return belief

def simulate(iter):
    env = gym.make(env_name)
    rocksample_env = env.env.env
    actions = []
    states = [rocksample_env.state.copy()]
    actual_grids = [rocksample_env.grid.copy()]

    print(rocksample_env.state)
    o = env.reset()
    info = dict()
    info['coords'] = env.env.env.start_state[:2].copy()
    observations = [o]

    rewards = []

    done = False

    while not done:
        action = agent.step(o, **info)
        actions += [action]
        # print(action)
        o, r, done, info = env.step(action)
        observations += [o]
        rewards += [r]
        states += [rocksample_env.state.copy()]
        actual_grids += [rocksample_env.grid.copy()]

    # print (discount(np.array(rewards), 0.95)[0])
    return discount(np.array(rewards), 0.95)[0]
    # return np.sum(rewards)



if __name__ == "__main__":

    num_rocks = 8
    if num_rocks == 4:
        env_name = 'bayes-rocksample-v0'
    else:
        env_name = 'bayes-rocksample-grid7rock8-v0'

    agent = RockSampleQMDPQFunction(num_rocks=num_rocks, num_envs=1)

    pool = Pool(10)

    n_iter = 100
    returns = pool.map(simulate, np.arange(n_iter))

    # import IPython; IPython.embed()
    # simulate(0)

    print("Average return: {} +/- ste {}".format(
        np.mean(returns), np.std(returns) / np.sqrt(n_iter)))

