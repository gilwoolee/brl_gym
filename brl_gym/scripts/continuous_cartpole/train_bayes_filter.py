from datetime import datetime
import os
import tqdm
import json

import numpy as np

from brl_gym.estimators.learnable_bf.bf_dataset import BayesFilterDataset
import brl_gym.estimators.learnable_bf.pt_util as pt_util
import brl_gym.estimators.learnable_bf.util as estimator_util
import torch
import multiprocessing
import torch.optim as optim
import pickle

from model import BayesFilterNet

def test(model, device, test_loader, mse_loss=True):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        hidden = None
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, hidden)

            test_loss += model.mse_loss(output, label, reduction='mean').item()

            # if batch_idx % 10 == 0:
            #     # print('Input\t%s\nGT\t%s\npred\t%s\n\n' % (
            #         # np.around(data[0].detach().cpu().numpy(),2),
            #     print('GT\t%s\npred\t%s\n\n' % (
            #         label[0,-1],
            #         output[0,-1]))

    test_loss /= len(test_loader)

    return test_loss

SEQUENCE_SIZE = 200
params = {
    'belief_dim': 8,
    'sequence_length': 32,
    'batch_size': 96,
    'lr': 0.003,
    'num_epochs': 150,
    'data_file': "bf_data_rand.pkl"
}

with open(params['data_file'], "rb") as f:
    edata = pickle.load(f)
    data = np.array([d[:SEQUENCE_SIZE - 1,:] for d in edata["data"]])
    output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["output"]])
    # print ("Data: ", data.shape)
    # print ("Output: ", output.shape)

data = (data - np.mean(data)) / np.std(data)
train_data = data[:int(len(data)*0.8)]
train_output = output[:int(len(output)*0.8)]
test_data = data[int(len(data)*0.8):]
test_output = output[int(len(output)*0.8):]

belief_dim = params['belief_dim']
sequence_length = params['sequence_length']
batch_size = params['batch_size']

mse_mode = True

data_train = BayesFilterDataset(train_data, train_output, sequence_length, batch_size=batch_size, mse=mse_mode)
data_test = BayesFilterDataset(test_data, test_output, sequence_length, batch_size=batch_size, mse=mse_mode)

use_cuda = torch.cuda.is_available()
num_workers = multiprocessing.cpu_count()

estimatorlr = params['lr']
estimator_weight_decay = 0.005

kwargs = {'num_workers': num_workers,
            'pin_memory': True} if use_cuda else {}

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = BayesFilterNet(6, 1, belief_dim).to(device)

optimizer = optim.Adam(model.parameters(),
    lr=estimatorlr, weight_decay=estimator_weight_decay)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, **kwargs)
test_loader  = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, **kwargs)

estimator_epoch = params['num_epochs']
base_path = "./data"
if not os.path.exists(base_path):
    os.makedirs(base_path)

logs_path = os.path.join(base_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(logs_path)

estimator_log_path = os.path.join(logs_path, 'estimator_xx_logs_mse/log.pkl')

checkpoint_path = os.path.join(logs_path, 'estimator_xx_checkpoints_mse')
param_path = os.path.join(logs_path, 'params.json')
with open(param_path, 'w') as fp:
    json.dump(params, fp)
# model.load_last_model('estimator_xx_checkpoints')
estimator_train_losses, estimator_test_losses = pt_util.read_log(estimator_log_path, ([], []))

PRINT_INTERVAL = 5
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
for epoch in tqdm.tqdm(range(0, estimator_epoch + 1)):
    estimator_train_loss = estimator_util.train(
        model, device, optimizer, train_loader, estimatorlr, epoch, PRINT_INTERVAL,
        batch_size, mse_loss=mse_mode)
    estimator_test_loss = test(
        model, device, test_loader, mse_loss=True)
    estimator_train_losses.append((epoch, estimator_train_loss))
    estimator_test_losses.append((epoch, estimator_test_loss))
    pt_util.write_log(estimator_log_path, (estimator_train_losses, estimator_test_losses))
    if epoch % 5 == 0:
        print ("Train loss", estimator_train_loss, " Test loss: ", estimator_test_loss)
    model.save_best_model(estimator_test_loss, checkpoint_path + '/%03d.pt' % epoch)
    # scheduler.step()
