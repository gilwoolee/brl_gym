from tqdm import tqdm

import numpy as np
import pickle

from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv, LQRControlCartPole
from brl_gym.wrapper_envs.classic_control.wrapper_continuous_cartpole import BayesContinuousCartPoleEnv

bayes_env = BayesContinuousCartPoleEnv()
env = bayes_env.env

SEQUENCE_SIZE = 200
RUNS_PER_CONFIG = 20

def collect_data_random():
    batch_observations = []
    batch_actions = []
    batch_rewards = []

    ENV_SAMPLES = 100

    for _ in tqdm(range(ENV_SAMPLES)):
        obs = env.reset()
        t = 0
        expert = LQRControlCartPole(env)
        observations = []
        actions = []
        rewards = []
        for t in range(SEQUENCE_SIZE):
            a, v = expert.lqr_control(obs)
            obs, r, done, _ = env.step(a)

            observations.append(obs)
            actions.append(a)
            rewards.append(r)
        
        observations, actions, rewards = \
            np.array(observations)[1:,:], np.array(actions), np.array(rewards)
        input_actions = np.expand_dims(actions[:SEQUENCE_SIZE - 1], axis=1)
        input_rewards = np.expand_dims(rewards[:SEQUENCE_SIZE - 1], axis=1)
        data = np.hstack((observations, input_actions, input_rewards))
        batch_observations += [data]
        batch_actions += [actions[1:]]

    batch_observations, batch_actions = \
        np.array(batch_observations), np.array(batch_actions)

    with open("bf_data_rand.pkl", "wb") as f:
        pickle.dump({"data": batch_observations, "output": batch_actions}, f)

    with open("bf_data_rand.pkl", "rb") as f:
        edata = pickle.load(f)
        data = np.array([d[:SEQUENCE_SIZE - 1,:] for d in edata["data"]])
        output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["output"]])
        print ("Data: ", data.shape)
        print ("Output: ", output.shape)

    data = (data - np.mean(data)) / np.std(data)
    train_data = data[:int(len(data)*0.8)]
    train_output = output[:int(len(output)*0.8)]
    test_data = data[int(len(data)*0.8):]
    test_output = output[int(len(output)*0.8):]

def collect_data_linspace():
    env = ContinuousCartPoleEnv(random_param=False)
    param_masscart = np.linspace(0.5, 2, 5)
    param_polelength = np.linspace(0.5, 2, 5)
    batch_observations = []
    batch_actions = []
    batch_rewards = []

    for masscart in param_masscart:
        for polelength in param_polelength:
            env.masscart = masscart
            env.length = polelength
            print ("Polelength: ", env.length)
            expert = LQRControlCartPole(env)
            for _ in range(RUNS_PER_CONFIG):
                observations = []
                actions = []
                rewards = []
                obs = env.reset()
                for t in range(SEQUENCE_SIZE):
                    a, v = expert.lqr_control(obs)
                    obs, r, done, _ = env.step(a)

                    observations.append(obs)
                    actions.append(a)
                    rewards.append(r)
                    # env.render()
                
                observations, actions, rewards = \
                    np.array(observations)[1:,:], np.array(actions), np.array(rewards)
                input_actions = np.expand_dims(actions[:SEQUENCE_SIZE - 1], axis=1)
                input_rewards = np.expand_dims(rewards[:SEQUENCE_SIZE - 1], axis=1)
                data = np.hstack((observations, input_actions, input_rewards))
                batch_observations += [data]
                batch_actions += [actions[1:]]

    batch_observations, batch_actions = \
        np.array(batch_observations), np.array(batch_actions)

    with open("bf_data.pkl", "wb") as f:
        pickle.dump({"data": batch_observations, "output": batch_actions}, f)

    with open("bf_data.pkl", "rb") as f:
        edata = pickle.load(f)
        data = np.array([d[:SEQUENCE_SIZE - 1,:] for d in edata["data"]])
        output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["output"]])
        print ("Data: ", data.shape)
        print ("Output: ", output.shape)

    data = (data - np.mean(data)) / np.std(data)
    train_data = data[:int(len(data)*0.8)]
    train_output = output[:int(len(output)*0.8)]
    test_data = data[int(len(data)*0.8):]
    test_output = output[int(len(output)*0.8):]

# collect_data_linspace()
collect_data_random()