import numpy as np
import os.path as osp

from brl_gym.wrapper_envs import BayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler, UniformSampler
from brl_gym.estimators.bayes_pusher_estimator import BayesPusherEstimator
from brl_gym.envs.mujoco.pusher import Pusher
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from gym.spaces import Box, Dict
from multiprocessing import Pool
from baselines.common.math_util import discount
import julia
from brl_gym.wrapper_envs.wrapper_pusher import ExplicitPusherEnv

def get_expert(target=0, index=1):
    rlopt = "/home/gilwoo/School_Workspace/rlopt"
    j = julia.Julia()

    j.include(osp.join(rlopt, "_init.jl"))
    j.include(osp.join(rlopt, "src/pg/Baseline.jl"))
    j.include(osp.join(rlopt, "src/ExpSim.jl"))
    # if target == 0:
    #     polo = "/home/gilwoo/POLO/pusher_down_"+ str(index)
    # else:
    #     polo = "/home/gilwoo/POLO/pusher_up_" + str(index)
    # baseline = j.Baseline.loadbaseline(osp.join(polo, "baseline.jld2"))
    directory = "/tmp/pusher_opt_opt_3/"
    datafile = j.ExpSim.load(osp.join(directory, "data.jld2"))

    # Replay the datafile
    state = np.squeeze(datafile["state"]).transpose()
    ctrl = np.squeeze(datafile["ctrl"]).transpose()
    obs = np.squeeze(datafile["obs"]).transpose()

    return state, ctrl

def collect_mujoco_batches():

    H = 600 #300 #64               # MPC horizon time
    TASK_T = 600         # length of a "trial" before agent is reset using it's initfunction
    NTRIALS = 1         # number of attempts agent has
    T = TASK_T*NTRIALS   # Total time alive for agent
    gamma = 1.0-1.0/TASK_T # 0.998

    experiences = []
    for i in range(1):
        observations = []
        next_observations = []
        rewards = []
        dones = []
        ctrls = []

        for k in range(NTRIALS):
            state, ctrl = get_expert(i, k + 1)
            print(state.shape)
            bayes_env = ExplicitPusherEnv(reset_params=False)
            bayes_env.env = bayes_env.envs[i]
            o = bayes_env.reset()
            obs = []
            obs += [np.concatenate([o['obs'], o['zbel']])]
            bayes_env.env.set_state(state[0,:5], state[0,5:])
            for t in range(state.shape[0]):
                o, r, _, _ = bayes_env.step(ctrl[t])
                obs += [np.concatenate([o['obs'], o['zbel']])]
                # print(o['obs'] - state[t+1])
                rewards += [r]
                ctrls += [ctrl[t]]
                dones += [False]
                # baseline_predictions += [j.Baseline.predict(baseline,
                #     o['obs'].reshape(-1,1).tolist())]
                bayes_env.env.render()
                # if t == TASK_T:
                #     o = bayes_env.reset()
                #     obs[-1] = np.concatenate([o['obs'], o['zbel']])
                dones[-1] = True

            observations += obs[:-1]
            next_observations += obs[1:]


    #     for k in range(NTRIALS):
    #         state, ctrl = get_expert((i + 1) % 2, k + 1)
    #         print(state.shape)
    #         bayes_env = ExplicitPusherEnv(reset_params=False)
    #         bayes_env.env = bayes_env.envs[i]
    #         o = bayes_env.reset()
    #         obs = []
    #         obs += [np.concatenate([o['obs'], o['zbel']])]
    #         for t in range(state.shape[0]):
    #             bayes_env.env.set_state(state[t,:5], state[t,5:])
    #             o, r, _, _ = bayes_env.step(ctrl[t])
    #             obs += [np.concatenate([o['obs'], o['zbel']])]
    #             # print(o['obs'] - state[t+1])
    #             rewards += [r]
    #             ctrls += [ctrl[t]]
    #             dones += [False]
    #             # baseline_predictions += [j.Baseline.predict(baseline,
    #             #     o['obs'].reshape(-1,1).tolist())]
    #             # # bayes_env.env.render()
    #             # if t == TASK_T:
    #             #     o = bayes_env.reset()
    #             #     obs[-1] = np.concatenate([o['obs'], o['zbel']])
    #             dones[-1] = True

    #         observations += obs[:-1]
    #         next_observations += obs[1:]


    #     rewards = np.reshape(np.array(rewards), (TASK_T, NTRIALS * 2))
    #     values = []
    #     for j in range(NTRIALS * 2):
    #         values += [discount(rewards[:, j], gamma)]

    #     values = np.array(values).ravel()
    #     dones = [False] * values.shape[0]

    #     experiences += [(np.array(observations),
    #     np.array(ctrls), values, rewards.ravel(), np.array(next_observations), dones)]

    # return experiences

if __name__ == "__main__":
    collect_mujoco_batches()
