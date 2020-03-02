import numpy as np
import os.path as osp
from gym import utils

from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler
from brl_gym.envs.mujoco.point_mass import PointMassEnv
from brl_gym.envs.mujoco.point_mass_slow import PointMassSlowEnv
from brl_gym.envs.mujoco.maze10 import Maze10
from brl_gym.envs.mujoco.maze10easy import Maze10Easy
from brl_gym.envs.mujoco.maze10easy_slow import Maze10EasySlow

from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.envs.mujoco.point_mass import GOAL_POSE as GOAL_POSE4
from brl_gym.envs.mujoco.maze10 import GOAL_POSE as GOAL_POSE10

from brl_gym.estimators.bayes_maze_estimator import BayesMazeEstimator, LearnableMazeBF

from gym.spaces import Box, Dict
# from multiprocessing import Pool
from brl_gym.wrapper_envs.util import discount
from brl_gym.envs.mujoco.motion_planner.maze import MotionPlanner
from brl_gym.envs.mujoco.motion_planner.VectorFieldGenerator import VectorField
from brl_gym.wrapper_envs.wrapper_maze import ExplicitBayesMazeEnv, Expert
import gym

from brl_gym.estimators.learnable_bf import BayesFilterDataset
import brl_gym.estimators.learnable_bf.pt_util as pt_util
import brl_gym.estimators.learnable_bf.util as estimator_util
import torch
import multiprocessing
import torch.optim as optim
import pickle

if __name__ == "__main__":
    maze_type = 10

    env = gym.make('Maze10easy-LearnableBF-noent-v0')
    model = env.env.estimator.model
    analytical_bayes_env = gym.make('Maze10easy-v0')
    bf = analytical_bayes_env.env.estimator

    if not osp.exists("estimator_data.pkl"):

        exp = Expert(nenv=1, maze_type=maze_type)
        all_rewards = []


        batch_observations = []
        batch_labels = []
        batch_bf_outputs = []
        batch_size = 48

        for _ in range(batch_size * 20):

            observations = []
            labels = []
            bf_outputs = []

            # Test expert
            o = env.reset()
            target = env.env.env.target
            done = False
            t = 0
            observations += [np.concatenate([np.zeros(3), o['obs'], [int(done)]])]
            labels += [env.env.env.target]

            bf_outputs += [bf.estimate(None, o['obs'])]

            for t in range(500):
                bel = np.zeros(maze_type)
                bel[target] = 1.0

                action = exp.action(np.concatenate([o['obs'], bel]).reshape(1,-1))
                action = action.squeeze() + np.random.normal()
                action[-1] = 1

                if np.random.rand() < 0.2:
                    action = np.random.normal(size=3)

                o, r, done, info = env.step(action)
                bf_outputs += [bf.estimate(action, o['obs'], **info)]

                observations += [np.concatenate([action, o['obs'], [int(done)]])]
                labels += [env.env.env.target]

                if done:
                    print("done at ", t)
                    o = env.reset()
                    target = env.env.env.target
                    observations += [np.concatenate([np.zeros(3), o['obs'], [int(False)]])]
                    labels += [env.env.env.target]
                    bf_outputs += [bf.estimate(None, o['obs'])]


            observations = np.array(observations)[:500,:]
            labels = np.array(labels)[:500]
            bf_outputs = np.array(bf_outputs)[:500]
            batch_observations += [observations]
            batch_labels += [labels]
            batch_bf_outputs += [bf_outputs]

        data = np.array(batch_observations)
        label = np.array(batch_labels)
        bf_output = np.array(batch_bf_outputs)

        with open("estimator_data.pkl", "wb") as f:
            pickle.dump({"data": data, "label": label, "bf":bf_output}, f)

    with open("estimator_data.pkl", "rb") as f:
        edata = pickle.load(f)
        data = np.array([d[:500,:] for d in edata["data"]])
        label = np.array([l[:500] for l in edata["label"]])
        bf_output = np.array([l[:500] for l in edata["bf"]])

    train_data = data[:int(len(data)*0.8)]
    train_label = label[:int(len(label)*0.8)]
    train_bf_output = bf_output[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]
    test_label = label[int(len(label)*0.8):]
    test_bf_output = bf_output[int(len(data)*0.8):]


    belief_dim = maze_type
    sequence_length = 32
    batch_size = 96
    # data_train = BayesFilterDataset(train_data, train_label, belief_dim, sequence_length,
    #     batch_size=batch_size)
    # data_test = BayesFilterDataset(test_data, test_label, belief_dim, sequence_length,
    #     batch_size=batch_size)
    data_train = BayesFilterDataset(train_data, train_bf_output, belief_dim, sequence_length,
        batch_size=batch_size, mse=True)
    data_test = BayesFilterDataset(test_data, test_bf_output, belief_dim, sequence_length,
        batch_size=batch_size, mse=True)
    use_cuda = torch.cuda.is_available()
    num_workers = multiprocessing.cpu_count()

    estimatorlr = 0.001
    estimator_weight_decay = 0.001

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    optimizer = optim.Adam(model.parameters(),
        lr=estimatorlr, weight_decay=estimator_weight_decay)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, **kwargs)

    device = 'cuda'
    estimator_epoch = 100
    estimator_log_path = 'estimator_xx_logs/log.pkl'

    estimator_train_losses, estimator_test_losses, estimator_test_accuracies = pt_util.read_log(estimator_log_path, ([], [], []))

    PRINT_INTERVAL = 10
    for epoch in range(0, estimator_epoch + 1):
        estimator_train_loss = estimator_util.train(
            model, device, optimizer, train_loader, estimatorlr, epoch, PRINT_INTERVAL,
            mse_loss=True)
        estimator_test_loss, estimator_test_accuracy = estimator_util.test(
            model, device, test_loader, mse_loss=True)
        estimator_train_losses.append((epoch, estimator_train_loss))
        estimator_test_losses.append((epoch, estimator_test_loss))
        estimator_test_accuracies.append((epoch, estimator_test_accuracy))
        pt_util.write_log(estimator_log_path, (estimator_train_losses, estimator_test_losses, estimator_test_accuracies))
        print(estimator_test_accuracy, estimator_test_loss)
        model.save_best_model(estimator_test_accuracy, 'estimator_xx_checkpoints/%03d.pt' % epoch)
