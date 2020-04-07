from tqdm import tqdm

import numpy as np
import pickle

import gym
import random
from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv, LQRControlCartPole
# from brl_gym.envs.classic_control.mountain_car import Continuous_MountainCarEnv
from brl_gym.wrapper_envs.classic_control.wrapper_continuous_cartpole import BayesContinuousCartPoleEnv

bayes_env = BayesContinuousCartPoleEnv()
env = bayes_env.env

SEQUENCE_SIZE = 200
n_chunks = 4
RUNS_PER_CONFIG = 50
MAX_CHUNKS = (SEQUENCE_SIZE * RUNS_PER_CONFIG) // n_chunks

def collect_data_random():
    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_params = []

    ENV_SAMPLES = 1000

    for _ in tqdm(range(ENV_SAMPLES)):
        obs = env.reset()
        t = 0
        expert = LQRControlCartPole(env)
        observations = []
        actions = []
        rewards = []
        params = []
        data = []
        for t in range(SEQUENCE_SIZE):
            a, v = expert.lqr_control(obs)
            obs, r, done, _ = env.step(a)
            a, r = np.array(a), np.array(r)
            input_data = np.hstack((obs, a, r))

            data.append(input_data)
            # observations.append(obs)
            actions.append(a)
            rewards.append(r)
            params.append(np.array((env.masscart, env.length)))

        data = np.array(data[:SEQUENCE_SIZE - 1])
        actions = np.array(actions)
        # print ("Data shape: ", data.shape)
        batch_observations += [data]
        batch_actions += [actions[1:]]
        batch_params += [params]
        # observations, 
        # actions, rewards = \
        #     np.array(observations)[1:,:], np.array(actions), np.array(rewards)
        # input_actions = np.expand_dims(actions[:SEQUENCE_SIZE - 1], axis=1)
        # input_rewards = np.expand_dims(rewards[:SEQUENCE_SIZE - 1], axis=1)
        # data = np.hstack((observations, input_actions, input_rewards))
        # batch_observations += [data]
        # batch_actions += [actions[1:]]

    batch_observations, batch_actions = \
        np.array(batch_observations), np.array(batch_actions)
    batch_params = np.array(batch_params)
    print ("Batch params: ", batch_params.shape)

    with open("bf_data_rand_3.pkl", "wb") as f:
        pickle.dump({"data": batch_observations, "output": batch_actions, 'params': batch_params}, f)

    # with open("bf_data_rand.pkl", "rb") as f:
    #     edata = pickle.load(f)
    #     data = np.array([d[:SEQUENCE_SIZE - 1,:] for d in edata["data"]])
    #     output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["output"]])
    #     print ("Data: ", data.shape)
    #     print ("Output: ", output.shape)

    # data = (data - np.mean(data)) / np.std(data)
    # train_data = data[:int(len(data)*0.8)]
    # train_output = output[:int(len(output)*0.8)]
    # test_data = data[int(len(data)*0.8):]
    # test_output = output[int(len(output)*0.8):]

def collect_data_linspace():
    env = ContinuousCartPoleEnv(random_param=False)
    env_for_expert = ContinuousCartPoleEnv(random_param=False)
    param_masscart = np.linspace(0.5, 1, 5)
    param_polelength = np.linspace(0.5, 1, 5)
    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_params = []

    observations_chunks = []
    actions_chunks = []
    rewards_chunks = []
    params_chunks = []

    ## Generate good examples
    for masscart in param_masscart:
        for polelength in param_polelength:
            params = {'length':polelength, 'masscart':masscart}
            env.set_params(params)
            env_for_expert.set_params(params)
            print ("Polelength: ", env.length)
            expert = LQRControlCartPole(env_for_expert)
            # while total_chunks < MAX_CHUNKS:
            for i in range(RUNS_PER_CONFIG):
                obs = env.reset()
                observations = []
                actions = []
                rewards = []
                params = []
                done = False
                t = 0
                for _ in range(SEQUENCE_SIZE):
                    a, v = expert.lqr_control(obs)
                    a = np.clip(a, -10, 10)
                    obs, r, done, _ = env.step(a)

                    observations.append(obs)
                    actions.append(a)
                    rewards.append(r)
                    params.append(np.array((env.masscart, env.length)))
                    # env.render()
                    t += 1
                # total = len(observations) // n_chunks
                # total_chunks += total
                # if total > 0:
                #     observations_chunks.extend([observations[i:i + n_chunks] for i in range(0, total * n_chunks, n_chunks)])
                #     actions_chunks.extend([actions[i:i + n_chunks] for i in range(0, total * n_chunks, n_chunks)])
                #     rewards_chunks.extend([rewards[i:i + n_chunks] for i in range(0, total * n_chunks, n_chunks)])
                #     params_chunks.extend([params[i:i + n_chunks] for i in range(0, total * n_chunks, n_chunks)])
                observations, actions, rewards = \
                    np.array(observations)[:SEQUENCE_SIZE - 1], np.array(actions), np.array(rewards)
                input_actions = np.expand_dims(actions[:SEQUENCE_SIZE - 1], axis=1)
                input_rewards = np.expand_dims(rewards[:SEQUENCE_SIZE - 1], axis=1)
                data = np.hstack((observations, input_actions, input_rewards))
                batch_observations += [data]
                batch_actions += [actions[1:]]
                batch_params += [params]

    # Generate bad examples
    print ("Generate bad examples")
    for masscart in param_masscart:
        for polelength in param_polelength:
            params = {'length':polelength, 'masscart':masscart}
            env.set_params(params)
            n_polelength = polelength + np.random.uniform(low=-2.0, high=2.0)
            n_masscart = masscart + np.random.uniform(low=-2.0, high=2.0)
            params_exp = {'length': n_polelength, 'masscart': n_masscart}
            env_for_expert.set_params(params_exp)
            print ("Polelength: ", env.length)
            expert = LQRControlCartPole(env_for_expert)
            # total_chunks = 0
            for i in range(RUNS_PER_CONFIG):
                obs = env.reset()
                observations = []
                actions = []
                rewards = []
                params = []
                done = False
                t = 0
                for _ in range(SEQUENCE_SIZE):
                    a, v = expert.lqr_control(obs)
                    a = np.clip(a, -10, 10)
                    obs, r, done, _ = env.step(a)
                    observations.append(obs)
                    actions.append(a)
                    rewards.append(r)
                    params.append(np.array((env.masscart, env.length)))
                    # env.render()
                    t += 1
                #     if done:
                #         env.reset()
                # total = len(observations) // n_chunks
                # total_chunks += total
                # if total > 0:
                #     observations_chunks.extend([observations[i:i + n_chunks] for i in range(0, total * n_chunks, n_chunks)])
                #     actions_chunks.extend([actions[i:i + n_chunks] for i in range(0, total * n_chunks, n_chunks)])
                #     rewards_chunks.extend([rewards[i:i + n_chunks] for i in range(0, total * n_chunks, n_chunks)])
                #     params_chunks.extend([params[i:i + n_chunks] for i in range(0, total * n_chunks, n_chunks)])
                observations, actions, rewards = \
                    np.array(observations)[:SEQUENCE_SIZE - 1], np.array(actions), np.array(rewards)
                input_actions = np.expand_dims(actions[:SEQUENCE_SIZE - 1], axis=1)
                input_rewards = np.expand_dims(rewards[:SEQUENCE_SIZE - 1], axis=1)
                data = np.hstack((observations, input_actions, input_rewards))
                batch_observations += [data]
                batch_actions += [actions[1:]]
                batch_params += [params]

    # batch_observations = np.array(observations_chunks)
    # batch_actions = np.expand_dims(np.array(actions_chunks), axis=2)
    # batch_rewards = np.expand_dims(np.array(rewards_chunks), axis=2)
    # batch_params = np.array(params_chunks)
    # batch_data = np.concatenate((batch_observations, batch_actions), axis=2)
    # print ("Obs chunk shape: ", observations_chunks.shape)
    # print ("Ac chunk shape: ", actions_chunks.shape)
    # print ("Rew chunk shape: ", rewards_chunks.shape)

    batch_observations, batch_actions = \
        np.array(batch_observations), np.array(batch_actions)#[:,1:]
    batch_params = np.array(batch_params)[:,1:,:] #np.expand_dims(np.array(batch_params), axis=2)#[:,1:,:]

    print ("Obs: ", batch_observations.shape, " Actions: ", batch_actions.shape, " Params: ", batch_params.shape)
    # # batch_observations = batch_observations[:,1:, :] - batch_observations[:,:batch_observations.shape[1] - 1, :]
    # print (batch_observations[0][0])
    # print ("batch_params: ", batch_params.shape)
    with open("bf_data_lin_30mar.pkl", "wb") as f:
        pickle.dump({"data": batch_observations, "output": batch_actions, "params": batch_params}, f)

    # with open("bf_data.pkl", "rb") as f:
    #     edata = pickle.load(f)
    #     data = np.array([d[:SEQUENCE_SIZE - 1,:] for d in edata["data"]])
    #     output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["output"]])
    #     print ("Data: ", data.shape)
    #     print ("Output: ", output.shape)

    # data = (data - np.mean(data)) / np.std(data)
    # train_data = data[:int(len(data)*0.8)]
    # train_output = output[:int(len(output)*0.8)]
    # test_data = data[int(len(data)*0.8):]
    # test_output = output[int(len(output)*0.8):]

def collect_data_car():
    env = gym.make('MountainCarContinuous-v0')
    param_power = np.linspace(0.5, 1.0, 10)
    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_params = []

    for power in param_power:
        env.power = power
        for _ in range(RUNS_PER_CONFIG):
            observations = []
            actions = []
            rewards = []
            params = []
            obs = env.reset()
            # print ("Pole angle: ", obs[2])
            for t in range(SEQUENCE_SIZE):
                a1 = np.random.uniform(low=-1.0, high=-0.5)
                a2 = np.random.uniform(low=0.5, high=1)
                a = random.sample([a1,a2], 1)
                obs, r, done, _ = env.step(a)
                a = a[0]

                observations.append(obs)
                actions.append(a)
                rewards.append(r)
                params.append(np.array((env.power)))
                # env.render()
            
            observations, actions, rewards = \
                np.array(observations)[1:,:], np.array(actions), np.array(rewards)
            input_actions = np.expand_dims(actions[:SEQUENCE_SIZE - 1], axis=1)
            input_rewards = np.expand_dims(rewards[:SEQUENCE_SIZE - 1], axis=1)
            data = np.hstack((observations, input_actions, input_rewards))
            batch_observations += [data]
            batch_actions += [actions[1:]]
            batch_params += [params]

    batch_observations, batch_actions = \
        np.array(batch_observations), np.array(batch_actions)#[:,1:]
    batch_params = np.expand_dims(np.array(batch_params), axis=2)#[:,1:,:]

    # # batch_observations = batch_observations[:,1:, :] - batch_observations[:,:batch_observations.shape[1] - 1, :]
    # print (batch_observations[0][0])
    print ("batch_params: ", batch_params.shape)
    with open("bf_data_car.pkl", "wb") as f:
        pickle.dump({"data": batch_observations, "output": batch_actions, "params": batch_params}, f)

def collect_data_pendulum():
    env = gym.make('Pendulum-v0')
    param_dt = np.linspace(0.01, 0.3, 10)
    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_params = []

    for dt in param_dt:
        env.dt = dt
        for _ in range(RUNS_PER_CONFIG):
            observations = []
            actions = []
            rewards = []
            params = []
            obs = env.reset()
            # print ("Pole angle: ", obs[2])
            for t in range(SEQUENCE_SIZE):
                a1 = np.random.uniform(low=-1.0, high=-0.5)
                a2 = np.random.uniform(low=0.5, high=1)
                a = random.sample([a1,a2], 1)
                obs, r, done, _ = env.step(a)
                a = a[0]

                observations.append(obs)
                actions.append(a)
                rewards.append(r)
                params.append(np.array((env.dt)))
                # env.render()
            
            observations, actions, rewards = \
                np.array(observations)[1:,:], np.array(actions), np.array(rewards)
            input_actions = np.expand_dims(actions[:SEQUENCE_SIZE - 1], axis=1)
            input_rewards = np.expand_dims(rewards[:SEQUENCE_SIZE - 1], axis=1)
            data = np.hstack((observations, input_actions, input_rewards))
            batch_observations += [data]
            batch_actions += [actions[1:]]
            batch_params += [params]

    batch_observations, batch_actions = \
        np.array(batch_observations), np.array(batch_actions)#[:,1:]
    batch_params = np.expand_dims(np.array(batch_params), axis=2)#[:,1:,:]

    # # batch_observations = batch_observations[:,1:, :] - batch_observations[:,:batch_observations.shape[1] - 1, :]
    # print (batch_observations[0][0])
    print ("batch_params: ", batch_params.shape)
    with open("bf_data_pendulum.pkl", "wb") as f:
        pickle.dump({"data": batch_observations, "output": batch_actions, "params": batch_params}, f)

# collect_data_pendulum()
# collect_data_car()
collect_data_linspace()
# collect_data_random()