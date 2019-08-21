from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.mle_env import MLEEnv

from brl_gym.envs.rocksample import load_env, Action
from brl_gym.estimators.bayes_rocksample_estimator import BayesRockSampleEstimator
from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.wrapper_envs.util import to_one_hot

import gym
from gym import utils
from gym.spaces import Box, Dict
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

class BayesRockSample(BayesEnv):
    def __init__(self, env_config="../resource/rocksample7x4.json"):
        env = load_env(env_config)
        estimator = BayesRockSampleEstimator(
            env.num_rocks, env.observation_space, env.action_space,
            env.grid_size, env.default_start_coords, env.rock_positions,
            env.good_rock_probability)
        super(BayesRockSample, self).__init__(env, estimator)

    def render(self, mode='human'):
        fig, ax, plt = self.env.render()

        belief_estimates = self.last_obs[-self.env.num_rocks:]

        for i, rp in enumerate(self.env.rock_positions):
            color = 'g' if belief_estimates[i] > 0.5 else 'r'
            if belief_estimates[i] > 0.5:
                alpha = (belief_estimates[i] - 0.5) * 2
            else:
                alpha = (0.5 - belief_estimates[i]) * 2

            ax.add_patch(Rectangle((rp[0], rp[1]), 1.0, 1.0, edgecolor=color,facecolor=color, alpha=alpha))
            ax.text(rp[0], rp[1], round(belief_estimates[i], 1),
                bbox=dict(facecolor='w', alpha=1.0), fontsize=12)

        plt.savefig("path_{}.png".format(self.env.timestep))


    def reset(self):
        obs = super(BayesRockSample, self).reset()
        self.history = dict(
            states=[self.env.state.copy()],
            actions=[],
            rewards=[],
            grids=[self.env.grid.copy()],
            observations=[obs.copy()])
        return obs


    def step(self, action):
        obs, reward, done, info = super(BayesRockSample, self).step(action)

        self.history['states'] += [self.env.state.copy()]
        self.history['actions'] += [action]
        self.history['rewards'] += [reward]
        self.history['grids'] += [self.env.grid.copy()]
        self.history['observations'] += [obs.copy()]

        return obs, reward, done, info

    def render_history(self, saveto='path.png'):
        """
        Renders the whole history
        """
        action_names = ["right" ,"left", "down", "up", "sample"]

        fig, ax = plt.subplots()

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(0, self.env.grid_size));
        ax.set_yticks(np.arange(0, self.env.grid_size));

        grid = np.zeros(self.env.grid.shape)

        rock_positions = np.array(self.env.rock_positions)
        prev_state = self.history['states'][0][:2]

        for j, (action, reward, state, obs, actual_grid) in enumerate(zip(self.history['actions'],
            self.history['rewards'], self.history['states'][1:], self.history['observations'][1:], self.history['grids'][1:])):

            if action == Action.SAMPLE:
                ax.scatter(
                    state[0] + 0.3,
                    state[1] + 0.5, c='magenta', s=150)
            if action > Action.SAMPLE:
                rock_idx = action - 5
                ax.scatter(
                    rock_positions[rock_idx, 0] + 0.3,
                    rock_positions[rock_idx, 1] + 0.5, c='cyan', s=50)

            belief_estimates = obs[-self.env.num_rocks:]

            for i, rp in enumerate(self.env.rock_positions):
                color = 'g' if belief_estimates[i] > 0.5 else 'r'
                if belief_estimates[i] > 0.5:
                    alpha = (belief_estimates[i] - 0.5) * 2
                else:
                    alpha = (0.5 - belief_estimates[i]) * 2

                ax.add_patch(Rectangle((rp[0], rp[1]), 1.0, 1.0, facecolor='w', alpha=1.0))
                ax.add_patch(Rectangle((rp[0], rp[1]), 1.0, 1.0, edgecolor=color,facecolor=color, alpha=alpha))
                ax.text(rp[0], rp[1], round(belief_estimates[i], 1),
                    bbox=dict(facecolor='w', alpha=1.0), fontsize=12)

            if j == len(self.history['actions']) - 1:
                if reward == -100:
                    ax.text(prev_state[0], prev_state[1], "DEAD", fontsize=15)
                else:
                    ax.text(prev_state[0], prev_state[1], "DONE", fontsize=15)
            else:
                ax.plot([prev_state[0] + 0.5, state[0] + 0.5],
                        [prev_state[1] + 0.5, state[1] + 0.5], 'r')

            prev_state = state

            plt.xlim(0, self.env.grid_size)
            plt.ylim(0, self.env.grid_size)

            if action <= 4:
                plt.title(action_names[action] + " Reward {}".format(reward))
            else:
                plt.title("Sense ({},{}) Reward {}".format(
                    self.env.rock_positions[rock_idx][0],
                    self.env.rock_positions[rock_idx][1],
                    reward))

            filename = saveto[:-4] + "_{:03}".format(j) + ".png"
            print (filename)
            plt.savefig(filename)
        plt.close()


class MLERockSample(MLEEnv):
    def __init__(self, env_config="../resource/rocksample7x4.json"):
        env = load_env(env_config)
        estimator = BayesRockSampleEstimator(
            env.num_rocks, env.observation_space, env.action_space,
            env.grid_size, env.default_start_coords, env.rock_positions,
            env.good_rock_probability)
        super(MLERockSample, self).__init__(env, estimator)


""" Simple wrapper for env registration """
class BayesRockSampleGrid7Rock8(BayesRockSample):
    def __init__(self):
        super(BayesRockSampleGrid7Rock8, self).__init__("../resource/rocksample7x8.json")

class ExplicitBayesRockSample(ExplicitBayesEnv):
    def __init__(self, env_config="../resource/rocksample7x4.json"):
        env = load_env(env_config)
        estimator = BayesRockSampleEstimator(
            env.num_rocks, env.observation_space, env.action_space,
            env.grid_size, env.default_start_coords, env.rock_positions,
            env.good_rock_probability)
        super(ExplicitBayesRockSample, self).__init__(env, estimator)

        obs_space = Box(np.zeros(env.observation_space.n), np.ones(env.observation_space.n), dtype=np.float32)
        self.num_rocks = env.num_rocks
        self.observation_space = Dict({"obs": obs_space,
            "zbel": estimator.belief_space})

    def _update_belief(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        self.estimator.estimate(
                action, obs, **kwargs)
        belief = self.estimator.get_belief()
        return belief, kwargs

    def reset(self):
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        obs = to_one_hot(obs, self.env.nS)
        self.last_obs = (obs, bel)
        return {'obs':obs, 'zbel':bel}

    def step(self, action):
        obs, reward, done, info = super(ExplicitBayesRockSample, self).step(action)
        obs['obs'] = to_one_hot(obs['obs'], self.env.nS)

        info['expert'] = int(''.join(str(int(x)) for x in self.env.state[2:]), 2)
        return obs, reward, done, info


class ExplicitBayesRockSampleRock8(ExplicitBayesRockSample):
    def __init__(self):
        super(ExplicitBayesRockSampleRock8, self).__init__("../resource/rocksample7x8.json")

class Expert:
    def __init__(self, num_rocks, num_envs, obs_dim):
        self.qmdp_q = RockSampleQMDPQFunction(
                num_rocks=num_rocks, num_envs=num_envs)
        self.obs_dim = obs_dim

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

        qvals = qmdp_q(obs, bel)
        actions = np.argmax(qvals, axis=1)
        return actions

def collect_batches(niter):
    experiences = []

    from baselines.common.cmd_util import make_vec_env, make_env
    from brl_gym.qmdps.rocksample_qmdp import RockSampleQMDPQFunction
    num_envs = 1
    qmdp_q = RockSampleQMDPQFunction(num_rocks=4, num_envs=1)

    for i, case in enumerate(qmdp_q.cases[:1]):
        rockstate = [int(x) for x in case]
        # env = make_vec_env('explicit-bayes-rocksample-v0', 'wrapper_envs', num_envs, seed=None,
        #     rockstates=rockstates)

        env = ExplicitBayesRockSample()
        env.env.set_start_rock_state(np.array(rockstate))

        observations = []
        values = []
        new_observations = []
        rewards = []
        dones = []
        actions = []

        for _ in range(1):
            env.env.set_start_rock_state(np.array(rockstate))
            o = env.reset()
            sum_rewards = 0
            for _ in range(100):
                expert_i = np.argwhere(np.all(qmdp_q.cases_np == env.env.state[2:], axis=1))[0,0]

                qvals = qmdp_q(o['obs'].reshape(1, -1), o['zbel'].reshape(1, -1), expert=expert_i)
                values += [qvals[0]]
                observations += [np.concatenate([o['obs'], o['zbel']])]
                if np.random.random() < 0.3:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(qvals[0])
                o, r, done, _ = env.step(action)
                sum_rewards += r

                actions += [action]
                new_observations += [np.concatenate([o['obs'], o['zbel']])]
                rewards += [r]
                dones += [done]
                if done:
                    o = env.reset()
                    sum_rewards = 0

        experiences += [(np.array(observations), np.array(values), np.array(actions),
                 np.array(rewards).reshape(1, -1), np.array(new_observations), dones)]

        print("Collected batches")
    return experiences

if __name__ == "__main__":

    experiences = collect_batches(10)
    import IPython; IPython.embed()

    # from brl_gym.envs.rocksample import Action


    # print("=============== Bayes ===============")
    # env = ExplicitBayesRockSample()
    # obs = env.reset()
    # print (obs)
    # for i in range(4):
    #     print(env.step(5 + i))

    # import IPython; IPython.embed()

#     print("=============== Bayes ===============")
#     env = BayesRockSampleGrid7Rock8()
#     obs = env.reset()
#     print (obs)
#     for i in range(5):
#         print(env.step(5 + i))



#     print("=============== MLE   ===============")
#     env = MLERockSample()
#     obs = env.reset()
#     print (obs)
#     for i in range(5):
#         print(env.step(5 + i))
# #
