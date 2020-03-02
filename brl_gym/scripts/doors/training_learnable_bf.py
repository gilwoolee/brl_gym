import numpy as np
import os.path as osp
from gym import utils

from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.wrapper_doors import Expert, ExplicitBayesDoorsEnv

from gym.spaces import Box, Dict
# from multiprocessing import Pool
from brl_gym.wrapper_envs.util import discount
import gym

from brl_gym.estimators.learnable_bf import BayesFilterDataset
import brl_gym.estimators.learnable_bf.pt_util as pt_util
import brl_gym.estimators.learnable_bf.util as estimator_util
import torch
import multiprocessing
import torch.optim as optim
import pickle


def test(model, device, test_loader, mse_loss=True):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        hidden = None
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, hidden)

            if mse_loss:
                test_loss += model.mse_loss(output, label, reduction='mean').item()
                # correct_mask = pred.eq(torch.round(label))

                x = output.cpu().detach().numpy().reshape(-1, 4)
                mask = np.argmax(np.array([bf.flatten_to_belief(i) for i in x]), axis=1)

                label_x = label.cpu().detach().numpy().reshape(-1, 4)
                correct_mask = np.argmax(np.array([bf.flatten_to_belief(i) for i in label_x]), axis=1)

            else:
                test_loss += model.loss(output, label, reduction='mean').item()

            num_correct = (mask == correct_mask).sum()
            correct += num_correct
            total += len(correct_mask)
            # Comment this out to avoid printing test results

            if batch_idx % 10 == 0:
                # print('Input\t%s\nGT\t%s\npred\t%s\n\n' % (
                    # np.around(data[0].detach().cpu().numpy(),2),
                print('GT\t%s\npred\t%s\n\n' % (
                    label[0,-1],
                    output[0,-1]))

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total

    return test_loss, test_accuracy

if __name__ == "__main__":


    env = gym.make('Door-LearnableBF-noent-v0')
    model = env.env.estimator.model
    analytical_bayes_env = gym.make('Door-v0')
    bf = analytical_bayes_env.env.estimator

    if not osp.exists("estimator_data.pkl"):

        exp = Expert()
        all_rewards = []


        batch_observations = []
        batch_labels = []
        batch_bf_outputs = []
        batch_size = 48

        for _ in range(batch_size*5):

            observations = []
            labels = []
            bf_outputs = []

            # Test expert
            o = env.reset()
            bf.reset()

            open_doors = env.env.env.open_doors.astype(np.float32)
            done = False
            t = 0
            observations += [np.concatenate([np.zeros(3), o['obs'], [int(done)]])]
            labels += [open_doors]

            bf_outputs += [bf.estimate(None, o['obs'])]

            bel = bf.flatten_to_belief(open_doors)

            for t in range(300):

                action = exp.action(np.concatenate([o['obs'], bel]).reshape(1,-1))
                action = action.squeeze() + np.random.normal() * 0.1
                action[-1] = 1

                # if np.random.rand() < 0.1:
                #     action = np.random.normal(size=3)

                o, r, done, info = env.step(action)
                bf_outputs += [bf.estimate(action, o['obs'], **info)]

                observations += [np.concatenate([action, o['obs'], [int(done)]])]
                labels += [open_doors]

                if done:
                    print("done at ", t)
                    o = env.reset()
                    open_doors = env.env.env.open_doors.astype(np.float32)
                    observations += [np.concatenate([np.zeros(3), o['obs'], [int(False)]])]
                    labels += [open_doors]
                    bf.reset()
                    bf_outputs += [bf.estimate(None, o['obs'])]


            observations = np.array(observations)[:300,:]
            labels = np.array(labels)[:300]
            bf_outputs = np.array(bf_outputs)[:300]
            batch_observations += [observations]
            batch_labels += [labels]
            batch_bf_outputs += [bf_outputs]
            # import IPython; IPython.embed(); import sys; sys.exit(0)

        data = np.array(batch_observations)
        label = np.array(batch_labels)
        bf_output = np.array(batch_bf_outputs)

        with open("estimator_data.pkl", "wb") as f:
            pickle.dump({"data": data, "label": label, "bf":bf_output}, f)

    with open("estimator_data.pkl", "rb") as f:
        edata = pickle.load(f)
        data = np.array([d[:300,:] for d in edata["data"]])
        label = np.array([l[:300] for l in edata["label"]])
        bf_output = np.array([l[:300] for l in edata["bf"]])

    # normalize
    data = (data - np.mean(data)) / np.std(data)
    train_data = data[:int(len(data)*0.8)]
    train_label = label[:int(len(label)*0.8)]
    train_bf_output = bf_output[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]
    test_label = label[int(len(label)*0.8):]
    test_bf_output = bf_output[int(len(data)*0.8):]

    belief_dim = 4
    sequence_length = 32
    batch_size = 96

    mse_mode = False

    if not mse_mode:
        data_train = BayesFilterDataset(train_data, train_label, belief_dim, sequence_length,
            batch_size=batch_size)
        data_test = BayesFilterDataset(test_data, test_label, belief_dim, sequence_length,
            batch_size=batch_size)
    else:
        data_train = BayesFilterDataset(train_data, train_bf_output, belief_dim, sequence_length,
            batch_size=batch_size, mse=True)
        data_test = BayesFilterDataset(test_data, test_bf_output, belief_dim, sequence_length,
            batch_size=batch_size, mse=True)
    use_cuda = torch.cuda.is_available()
    num_workers = multiprocessing.cpu_count()

    estimatorlr = 0.001
    estimator_weight_decay = 0.005

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    optimizer = optim.Adam(model.parameters(),
        lr=estimatorlr, weight_decay=estimator_weight_decay)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, **kwargs)

    device = 'cuda'
    estimator_epoch = 100
    estimator_log_path = 'estimator_xx_logs_{}/log.pkl'.format("mse" if mse_mode else "ce")

    checkpoint_path = 'estimator_xx_checkpoints_{}'.format("mse" if mse_mode else "ce")
    # model.load_last_model('estimator_xx_checkpoints')
    estimator_train_losses, estimator_test_losses, estimator_test_accuracies = pt_util.read_log(estimator_log_path, ([], [], []))

    PRINT_INTERVAL = 10
    for epoch in range(0, estimator_epoch + 1):
        estimator_train_loss = estimator_util.train(
            model, device, optimizer, train_loader, estimatorlr, epoch, PRINT_INTERVAL,
            mse_loss=mse_mode)
        estimator_test_loss, estimator_test_accuracy = test(
            model, device, test_loader, mse_loss=mse_loss)
        estimator_train_losses.append((epoch, estimator_train_loss))
        estimator_test_losses.append((epoch, estimator_test_loss))
        estimator_test_accuracies.append((epoch, estimator_test_accuracy))
        pt_util.write_log(estimator_log_path, (estimator_train_losses, estimator_test_losses, estimator_test_accuracies))
        print("train loss", estimator_train_loss)
        print(estimator_test_accuracy, estimator_test_loss)
        model.save_best_model(estimator_test_accuracy, checkpoint_path + '/%03d.pt' % epoch)
