import numpy as np

from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv, LQRControlCartPole
from brl_gym.wrapper_envs.classic_control.wrapper_continuous_cartpole import BayesContinuousCartPoleEnv

import torch
import multiprocessing
import torch.optim as optim
import pickle

from model import BayesFilterNet

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

param_masscart = np.linspace(0.5, 2, 5)
param_polelength = np.linspace(0.5, 2, 5)

belief_dim = 16
model = BayesFilterNet(5, 1, belief_dim).to(device)
model.load_model("/home/rishabh/work/brl_gym/brl_gym/scripts/continuous_cartpole/data/2020-03-09_17-02-14/estimator_xx_checkpoints_mse/098.pt")
bayes_env = BayesContinuousCartPoleEnv()
env = bayes_env.env
# env = ContinuousCartPoleEnv(random_param=False)

ENV_SAMPLES = 100
SEQUENCE_SIZE = 200

with open("bf_data_rand.pkl", "rb") as f:
    edata = pickle.load(f)
    data = np.array([d[:SEQUENCE_SIZE - 1,:] for d in edata["data"]])
    output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["output"]])
    print ("Data: ", data.shape)
    print ("Output: ", output.shape)

mean = np.mean(data)
std = np.std(data)

for _ in range(ENV_SAMPLES):
    obs = env.reset()
    t = 0
    expert = LQRControlCartPole(env)
    a, v = expert.lqr_control(obs)
    obs, r, done, _ = env.step(a)
    a, r = np.array(a), np.array(r)
    hidden = None
    for t in range(SEQUENCE_SIZE):
        data = np.expand_dims(np.hstack((obs, a, r)), axis=0)
        data = (data - mean) / std
        data = np.expand_dims(data, axis=0)
        data = torch.Tensor(data).float().to(device)
        a, hidden = model(data, hidden)
        a = a[0].cpu().data.numpy()[0]
        obs, r, done, _ = env.step(a)
        a, r = np.array(a), np.array(r)
        env.render()

# for masscart in param_masscart:
#     for polelength in param_polelength:
#         env.masscart = masscart
#         env.length = polelength
#         obs = env.reset()
#         t = 0
#         expert = LQRControlCartPole(env)
#         a, v = expert.lqr_control(obs)
#         obs, r, done, _ = env.step(a)
#         a, r = np.array(a), np.array(r)
#         hidden = None
#         for t in range(SEQUENCE_SIZE):
#             data = np.expand_dims(np.hstack((obs, a, r)), axis=0)
#             data = np.expand_dims(data, axis=0)
#             data = torch.Tensor(data).float().to(device)
#             a, hidden = model(data, hidden)
#             a = a[0].cpu().data.numpy()[0]
#             obs, r, done, _ = env.step(a)
#             a, r = np.array(a), np.array(r)
#             env.render()