import numpy as np
import os.path as osp

from brl_gym.wrapper_envs import BayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler, UniformSampler
from brl_gym.estimators.bayes_pusher_estimator import BayesPusherEstimator
from brl_gym.envs.mujoco.pusher import Pusher, TARGET_LOCATIONS
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from gym.spaces import Box, Dict
from multiprocessing import Pool
from baselines.common.math_util import discount
# import julia



class BayesPusherEnv(BayesEnv):
    # Wrapper envs for mujoco envs
    def __init__(self, reset_params=True):
        envs = []
        for i in range(len(TARGET_LOCATIONS)):
            env = Pusher()
            env.target = i
            envs += [env]
        self.estimator = BayesPusherEstimator()
        self.env_sampler = DiscreteEnvSampler(envs)
        super(BayesPusherEnv, self).__init__(env, self.estimator)
        self.nominal_env = Pusher()
        self.reset_params = reset_params

    def reset(self):
        if self.reset_params:
            self.env = self.env_sampler.sample()
        return super().reset()

    def step(self, action):
        prev_belief = self.estimator.get_belief()
        prev_state = self.env.get_state()
        obs, reward, done, info = self.env.step(action)
        info['prev_state'] = prev_state
        info['curr_state'] = self.env.get_state()

        # Estimate
        self.estimator.estimate(action, obs, **info)
        belief = self.estimator.get_belief()
        info['belief'] = belief

        obs = np.concatenate([obs, belief], axis=0)
        return obs, reward, done, info


class ExplicitPusherEnv(ExplicitBayesEnv):
    def __init__(self, reset_params=True):
        envs = []
        for i in range(len(TARGET_LOCATIONS)):
            env = Pusher()
            env.target = i
            envs += [env]
        self.envs = envs
        self.estimator = BayesPusherEstimator()
        self.env_sampler = DiscreteEnvSampler(envs)
        super(ExplicitPusherEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zbel": self.estimator.belief_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params

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
        info['expert'] = self.env.target

        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        if self.reset_params:
            self.env = self.env_sampler.sample()
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        self.last_obs = (obs, bel)
        return {'obs':obs, 'zbel':bel}


def get_expert(target=0, index=1):
    rlopt = "/home/gilwoo/School_Workspace/rlopt"
    j = julia.Julia()

    j.include(osp.join(rlopt, "_init.jl"))
    j.include(osp.join(rlopt, "src/pg/Baseline.jl"))
    j.include(osp.join(rlopt, "src/ExpSim.jl"))
    if target == 0:
        polo = "/home/gilwoo/POLO/pusher_down_"+ str(index)
    else:
        polo = "/home/gilwoo/POLO/pusher_up_" + str(index)
    # baseline = j.Baseline.loadbaseline(osp.join(polo, "baseline.jld2"))
    datafile = j.ExpSim.load(osp.join(polo, "data.jld2"))

    # Replay the datafile
    state = np.squeeze(datafile["state"]).transpose()
    ctrl = np.squeeze(datafile["ctrl"]).transpose()
    obs = np.squeeze(datafile["obs"]).transpose()

    return state, ctrl


def simple_expert_actor(obs, target):
    obs = obs.ravel()
    target_loc = TARGET_LOCATIONS[target]
    dist_to_obj = obs[10:12]
    dist_to_target = obs[12:].reshape(-1, 2)[target]
    if np.linalg.norm(dist_to_obj) < 0.1:
        xy = dist_to_target/np.linalg.norm(dist_to_target)
        theta = np.arctan2(dist_to_target[1], dist_to_target[0])
    else:
        # Get to the back of the object toward the goal:
        #direction = dist_to_target * -1
        #direction /= np.linalg.norm(direction)
        #direction *= 0.1

        # Get closer to the object
        xy = dist_to_obj/np.linalg.norm(dist_to_obj)
        theta = np.arctan2(dist_to_obj[1], dist_to_obj[0])
    if theta > 1:
        theta = 1
    elif theta < -1:
        theta = -1
    return np.array([xy[0], xy[1], theta])

def simple_expert_actor(obs, target):
    target_loc = TARGET_LOCATIONS[target]
    dist_to_obj = obs[:,10:12]
    dist_to_target = obs[:, 12:12+10].reshape(-1, 5, 2)[:, target, :]
    xy1 = dist_to_target/np.tile(np.linalg.norm(dist_to_target, axis=1), (2, 1)).transpose()
    theta1 = np.arctan2(dist_to_target[:,1], dist_to_target[:,0])
    xy2 = dist_to_obj/np.tile(np.linalg.norm(dist_to_obj,axis=1), (2,1)).transpose()
    theta2 = np.arctan2(dist_to_obj[:,1], dist_to_obj[:,0])
    norm_dist_to_obj = np.linalg.norm(dist_to_obj, axis=1)
    xy = xy1.copy()
    xy[norm_dist_to_obj >= 0.1] = xy2[norm_dist_to_obj >= 0.1]
    theta = theta1
    theta[norm_dist_to_obj >= 0.1] = theta2[norm_dist_to_obj >= 0.1]
    # theta[theta>0.5] = 0.5
    # theta[theta<0.5] = -0.5
    return np.concatenate([xy, theta.reshape(-1,1)], axis=1)

def get_value(args):
    obs, target = args
    bayes_env = ExplicitPusherEnv(reset_params=False)

    env = bayes_env.envs[target]
    env.reset()
    env.set_state(obs[:5], obs[5:10])
    o = obs
    rewards = []
    done = False
    for t in range(30):
        action = simple_expert_actor(o.reshape(1,-1), target)
        o, r, done, _ = env.step(action.ravel())
        rewards += [r]
        if done:
            break

    gamma = 0.998
    value = discount(np.array(rewards), gamma)[0]
    return value

def get_simple_expert_value(target=0):
    return lambda obs: get_value(obs, target)

# pool = Pool(5)

def get_value_approx(obs, target):
    target_loc = TARGET_LOCATIONS[target]
    dist_to_obj = obs[:,10:12]
    dist_to_target = obs[:, 12:12+10].reshape(-1, 5, 2)[:, target, :]
    # Assume each take a step
    return -15 * np.linalg.norm(dist_to_obj + dist_to_target, axis=1)

def qmdp_expert(obs, bel):
    vals = []
    for i in range(bel.shape[1]):
        vals += [np.array(get_value_approx(obs, i))]
        # vals += [np.array(pool.map(get_value, [(o, i) for o in obs]))]
    vals = np.array(vals).transpose()
    return np.sum(vals * bel, axis=1)

def collect_mujoco_batches():

    H = 600 #300 #64               # MPC horizon time
    TASK_T = 600         # length of a "trial" before agent is reset using it's initfunction
    NTRIALS = 3         # number of attempts agent has
    T = TASK_T*NTRIALS   # Total time alive for agent
    gamma = 1.0-1.0/TASK_T # 0.998

    experiences = []
    for i in range(2):
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
            for t in range(state.shape[0]):
                bayes_env.env.set_state(state[t,:5], state[t,5:])
                o, r, _, _ = bayes_env.step(ctrl[t])
                obs += [np.concatenate([o['obs'], o['zbel']])]
                # print(o['obs'] - state[t+1])
                rewards += [r]
                ctrls += [ctrl[t]]
                dones += [False]
                # baseline_predictions += [j.Baseline.predict(baseline,
                #     o['obs'].reshape(-1,1).tolist())]
                # # bayes_env.env.render()
                # if t == TASK_T:
                #     o = bayes_env.reset()
                #     obs[-1] = np.concatenate([o['obs'], o['zbel']])
                dones[-1] = True

            observations += obs[:-1]
            next_observations += obs[1:]


        for k in range(NTRIALS):
            state, ctrl = get_expert((i + 1) % 2, k + 1)
            print(state.shape)
            bayes_env = ExplicitPusherEnv(reset_params=False)
            bayes_env.env = bayes_env.envs[i]
            o = bayes_env.reset()
            obs = []
            obs += [np.concatenate([o['obs'], o['zbel']])]
            for t in range(state.shape[0]):
                bayes_env.env.set_state(state[t,:5], state[t,5:])
                o, r, _, _ = bayes_env.step(ctrl[t])
                obs += [np.concatenate([o['obs'], o['zbel']])]
                # print(o['obs'] - state[t+1])
                rewards += [r]
                ctrls += [ctrl[t]]
                dones += [False]
                # baseline_predictions += [j.Baseline.predict(baseline,
                #     o['obs'].reshape(-1,1).tolist())]
                # # bayes_env.env.render()
                # if t == TASK_T:
                #     o = bayes_env.reset()
                #     obs[-1] = np.concatenate([o['obs'], o['zbel']])
                dones[-1] = True

            observations += obs[:-1]
            next_observations += obs[1:]


        rewards = np.reshape(np.array(rewards), (TASK_T, NTRIALS * 2))
        values = []
        for j in range(NTRIALS * 2):
            values += [discount(rewards[:, j], gamma)]

        values = np.array(values).ravel()
        dones = [False] * values.shape[0]

        experiences += [(np.array(observations),
        np.array(ctrls), values, rewards.ravel(), np.array(next_observations), dones)]

    return experiences

def test_simple_qmdp():
    bayes_env = ExplicitPusherEnv(reset_params=False)
    expert_envs = bayes_env.envs
    #experts = [get_simple_expert_actor(i) for i in range(2)]
    #expert_values = [get_simple_expert_value(i) for i in range(2)]
    rewards = []
    test_env = ExplicitPusherEnv(reset_params=True)
    o = test_env.reset()
    done = False
    for t in range(1000):
        print(t)
        actions = [simple_expert_actor(o['obs'].reshape(1,-1), i) for i in range(2)]
        noisy_actions = []
        state = o['obs'][:10]
        for action in actions:
            a = np.tile(action.ravel((-1, 1)), (10,1))
            noisy_actions += [a + np.random.uniform(size=(10,3))*0.05]
        qmdp_vals = []
        noisy_actions = np.concatenate(noisy_actions, axis=0)
        for a in noisy_actions:
            vals = []
            for i, env in enumerate(expert_envs):
                env.set_state(state[:5], state[5:])
                sim_o, r, d, _ = env.step(a)
                vals += [get_value((sim_o, i))]
            #     print(get_value((sim_o, i)))
            #     print(get_value_approx(sim_o.reshape(1,-1), i))
            #     print('------------------')
            qmdp_vals += [o['zbel'] * np.array(vals)]

        #qmdp_vals += [qmdp_expert(o['obs'].reshape(1,-1), o['zbel'].reshape(1,-1))]
        #import IPython; IPython.embed(); import sys; sys.exit(0)
        qmdp_vals = np.sum(np.array(qmdp_vals), axis=1)
        action = noisy_actions[np.argmax(qmdp_vals)]
        # print(o['zbel'], state[:5], np.around(action,2))
        o, r, done, _ = test_env.step(action)
        # test_env.render()
        rewards += [r]
        if done:
            break

    print(np.sum(rewards))
    return np.sum(rewards)

def qmdp_expert_actor(obs, bel):
    bayes_env = ExplicitPusherEnv(reset_params=False)
    expert_envs = bayes_env.envs
    rewards = []
    actions = [simple_expert_actor(obs, i) for i in range(2)]
    noisy_actions = []
    state = obs[:10]
    for action in actions:
        a = np.tile(action.ravel((-1, 1)), (10,1))
        noisy_actions += [a + np.random.uniform(size=(10,3))*0.05]
    qmdp_vals = []
    noisy_actions = np.concatenate(noisy_actions, axis=0)
    for a in noisy_actions:
        vals = []
        for i, env in enumerate(expert_envs):
            env.set_state(state[:5], state[5:])
            sim_o, r, d, _ = env.step(a)

            vals += [expert_values[i](sim_o)]
        qmdp_vals += [o['zbel'] * np.array(vals)]
    pass

def simple_combined_expert(obs, bel):
    actions = [simple_expert_actor(obs, i) for i in range(len(TARGET_LOCATIONS))]
    return np.sum(np.array(actions) * bel[None].transpose(2,1,0), axis=0)

def test_simple_combined():
    rewards = []
    test_env = ExplicitPusherEnv(reset_params=True)
    o = test_env.reset()
    done = False
    for _ in range(200):
        action = simple_combined_expert(o['obs'].reshape(1,-1), o['zbel'].reshape(1,-1))
        o, r, done, _ = test_env.step(action.ravel()+ np.random.uniform(size=(3))*0.05)
        test_env.render()
        print(np.around(o['zbel'],2))
        rewards += [r]
        if done:
            break

    print("pathlen", len(rewards))
    value = discount(np.array(rewards), 0.998)[0]

def test_expert(target):
    bayes_env = ExplicitPusherEnv(reset_params=False)
    bayes_env.env = bayes_env.envs[target]
    o = bayes_env.reset()
    rewards = []
    for t in range(150):
        actions = simple_expert_actor(o['obs'].reshape(1,-1), target) #+ np.random.normal(size=3)*0.05
        print(actions)
        o, r, d, _ = bayes_env.step(actions[0])
        bayes_env.render()
        rewards += [r]
        if d:
            break



def collect_batches(NTRIALS=3):
    gamma = 0.998

    experts = [get_simple_expert_actor(i) for i in range(2)]
    expert_values = [get_simple_expert_value(i) for i in range(2)]
    experiences = []

    observations = []
    next_observations = []
    rewards = []
    dones = []
    ctrls = []
    values = []

    for i in range(2):
        for k in range(NTRIALS):
            bayes_env = ExplicitPusherEnv(reset_params=False)
            bayes_env.env = bayes_env.envs[i]
            o = bayes_env.reset()
            obs = []
            obs += [np.concatenate([o['obs'], o['zbel']])]
            for t in range(100):
                values += [[expert_values[j](o['obs']) for j in range(2)]]
                actions = [experts[j](o['obs']) + np.random.normal(size=3)*0.05 for j in range(2)]
                o, r, d, _ = bayes_env.step(actions[i])
                obs += [np.concatenate([o['obs'], o['zbel']])]
                rewards += [r]
                ctrls += [actions]
                dones += [d]
                if d:
                    break

            observations += obs[:-1]
            next_observations += obs[1:]

    values = np.array(values).transpose()
    ctrls = np.array(ctrls).transpose(1,0,2)
    rewards = np.array(rewards)
    experiences = {'obs':np.array(observations),
        'action':np.array(ctrls),
        'values':values,
        'rewards':rewards.ravel(), 'next_obs':np.array(next_observations), 'done':dones}

    return experiences
if __name__ == "__main__":
#    exps = collect_batches()
    rewards = []
    for i in range(100):
        rewards += [test_simple_qmdp()]
    print("Mean")
    print(np.mean(np.array(rewards)))
    import IPython; IPython.embed()

