from datetime import datetime
import os
import tqdm
import json

import numpy as np

from tensorboardX import SummaryWriter
from brl_gym.estimators.learnable_bf.bf_dataset_2 import BayesFilterDataset
import brl_gym.estimators.learnable_bf.pt_util as pt_util
import brl_gym.estimators.learnable_bf.util as estimator_util
import torch
import multiprocessing
import torch.optim as optim
import pickle

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from model import BayesFilterNet, BayesFilterNet2, SimpleNN, SimpleNNClassifier

def test(model, device, test_loader, mse_loss=True):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        output, label = None, None
        for batch_idx, (data, label) in enumerate(test_loader):
            hidden = None
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, hidden)

            test_loss += model.mse_loss(output, label, reduction='mean').item()

            # if batch_idx % 10 == 0:
                # print('Input\t%s\nGT\t%s\npred\t%s\n\n' % (
                #     np.around(data[0].detach().cpu().numpy(),2),
        # print('GT\t%s\npred\t%s\n\n' % (
        #     label[-1],
        #     output[-1]))

    test_loss /= len(test_loader)

    return test_loss

SEQUENCE_SIZE = 200
params = {
    'belief_dim': 16,
    'hidden_dim': 256,
    'n_layers': 1,
    'sequence_length': 10,
    'batch_size': 256,
    'lr': 0.000005,
    'num_epochs': 2000,
    'data_file': "bf_data_lin_new.pkl"
}

with open(params['data_file'], "rb") as f:
    edata = pickle.load(f)
    data = np.array([d[:SEQUENCE_SIZE - 1] for d in edata["data"]])
    output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["params"]]) # params
    # output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["output"]]) # actions
    # print ("Data: ", data.shape)
    # print ("Output: ", output.shape)

# shuffle if using linspace data
np.random.seed(172)
perm_id = np.arange(len(data))
np.random.shuffle(perm_id)
data = data[perm_id, :, :5]
output = output[perm_id, 0, :]

new_data = []
for i in range(data.shape[0]):
    features = []
    for j in range(data.shape[1] - 1):
        diff = data[i,j+1,:] - data[i,j,:]
        feat = np.concatenate((data[i,j+1], diff))
        features.extend(feat)
    new_data.extend([features])
data = np.array(new_data)
# print ("feat:", data.shape)
# exit()

# data = np.reshape(data, (-1, data.shape[1] * data.shape[2]))
output = np.reshape(output, (-1, output.shape[-1]))
new_output = []
for i in range(output.shape[0]):
    label1, label2 = np.zeros((1,5)), np.zeros((1,5))
    label1[0,int((output[i,0] - 0.5) // 0.375)] = 1
    label2[0,int((output[i,1] - 0.5) // 0.375)] = 1
    label = np.vstack((label1, label2))
    new_output.extend([label])
output = np.array(new_output)

print ("Output shape: ", output.shape)
# exit()

for i in range(data.shape[-1]):
    data[:,i] = (data[:,i] - np.mean(data[:,i])) / np.std(data[:,i]) # seems to be the most logical thing to do compared to normalizing entire data
    # data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.max(data[:,i])- np.min(data[:,i])) # loss seems to be low with this type, but could be because of the range itself

# data = (data - np.mean(data)) / np.std(data) # works decent
# data /= np.max(data)
# output = (output - np.mean(output)) / np.std(output)
# output /= np.max(output)

# data = (data - np.min(data)) / (np.max(data) - np.min(data))
# output = (output - np.min(output)) / (np.max(output) - np.min(output))

# output[:,:,0] = (output[:,:,0] - np.mean(output[:,:,0])) / np.std(output[:,:,0])
# output[:,:,1] = (output[:,:,1] - np.mean(output[:,:,1])) / np.std(output[:,:,1])

# for i in range(output.shape[-1]):
#     # output[:,i] = (output[:,i] - np.mean(output[:,i])) / np.std(output[:,i])
#     output[:,i] = (output[:,i] - np.min(output[:,i])) / (np.max(output[:,i])- np.min(output[:,i]))

# un, count = np.unique(output[:,0,0], return_counts=True)
# plt.bar(count, un)
# plt.show()

# print ("Output max: ", np.max(output[:,:,1]), " Output min ", np.min(output[:,:,1]))
# print ("Output max: ", np.max(output[:,:,0]), " Output min ", np.min(output[:,:,0]))
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
estimator_weight_decay = 0.0005

kwargs = {'num_workers': num_workers,
            'pin_memory': True} if use_cuda else {}

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if data_train.__feature_len__()[1] == 5:
    model = SimpleNNClassifier(data_train.__feature_len__()[0], data_train.__feature_len__()[1], belief_dim, hidden_dim=params['hidden_dim'], n_layers=params['n_layers'], dropout=0.1).to(device)
elif len(data.shape) == 2:
    model = SimpleNN(data_train.__feature_len__()[0], 2, belief_dim, hidden_dim=params['hidden_dim'], n_layers=params['n_layers'], dropout=0.1).to(device)
elif data_train.__feature_len__()[1] == 2:
    model = BayesFilterNet2(data_train.__feature_len__()[0], 2, belief_dim, hidden_dim=params['hidden_dim'], n_layers=params['n_layers']).to(device)
else:
    model = BayesFilterNet(data_train.__feature_len__()[0], 1, belief_dim, hidden_dim=params['hidden_dim'], n_layers=params['n_layers']).to(device)

# cls 2020-03-24_04-47-06 2020-03-24_04-19-55
# model.load_last_model("/home/rishabh/work/brl_gym/brl_gym/scripts/continuous_cartpole/data/2020-03-24_04-19-55/estimator_xx_checkpoints_mse")
# model.weight_init()
# model.init_hidden(batch_size, device)

optimizer = optim.Adam(model.parameters(),
    lr=estimatorlr) #, weight_decay=estimator_weight_decay)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, **kwargs)
test_loader  = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, **kwargs)

# for data, label in train_loader:
#     print ("Data shape: ", data.shape)

estimator_epoch = params['num_epochs']
base_path = "./data"
if not os.path.exists(base_path):
    os.makedirs(base_path)

logs_path = os.path.join(base_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(logs_path)

estimator_log_path = os.path.join(logs_path, 'estimator_xx_logs_mse/log.pkl')
writer = SummaryWriter(logs_path)
checkpoint_path = os.path.join(logs_path, 'estimator_xx_checkpoints_mse')
param_path = os.path.join(logs_path, 'params.json')
with open(param_path, 'w') as fp:
    json.dump(params, fp)
# model.load_last_model('estimator_xx_checkpoints')
estimator_train_losses, estimator_test_losses = pt_util.read_log(estimator_log_path, ([], []))

PRINT_INTERVAL = 2
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
for epoch in tqdm.tqdm(range(0, estimator_epoch + 1)):
    # model.zero_grad()
    estimator_train_loss = estimator_util.train(
        model, device, optimizer, train_loader, estimatorlr, epoch, PRINT_INTERVAL,
        batch_size, mse_loss=mse_mode)
    
    estimator_train_losses.append((epoch, estimator_train_loss))
    # pt_util.write_log(estimator_log_path, (estimator_train_losses, estimator_test_losses))
    if epoch % PRINT_INTERVAL == 0:
        estimator_test_loss = test(
        model, device, test_loader, mse_loss=True)
        estimator_test_losses.append((epoch, estimator_test_loss))
        print ("Train loss", estimator_train_loss, " Test loss: ", estimator_test_loss)

    model.save_best_model(estimator_test_loss, checkpoint_path + '/%03d.pt' % epoch)
    writer.add_scalar("train_loss", estimator_train_loss, epoch)
    writer.add_scalar("test_loss", estimator_test_loss, epoch)
    # if epoch < 200:
    #     # for param_group in optimizer.param_groups:
    #     #     print ("Learning rate: ", param_group['lr'])
    scheduler.step()
writer.close()