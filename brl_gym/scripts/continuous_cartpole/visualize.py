from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from brl_gym.estimators.learnable_bf.bf_dataset import BayesFilterDataset
import brl_gym.estimators.learnable_bf.pt_util as pt_util
import brl_gym.estimators.learnable_bf.util as estimator_util
import torch
import multiprocessing
import torch.optim as optim
import pickle

from model import BayesFilterNet

SEQUENCE_SIZE = 200
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

belief_dim = 4
sequence_length = 32
batch_size = 96

mse_mode = True

data_train = BayesFilterDataset(train_data, train_output, sequence_length, batch_size=batch_size, mse=mse_mode)
data_test = BayesFilterDataset(test_data, test_output, sequence_length, batch_size=batch_size, mse=mse_mode)

use_cuda = torch.cuda.is_available()
num_workers = multiprocessing.cpu_count()

kwargs = {'num_workers': num_workers,
            'pin_memory': True} if use_cuda else {}

device = "cpu"
model = BayesFilterNet(6, 1, belief_dim)
model.load_last_model("estimator_xx_checkpoints_mse")

test_loader  = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, **kwargs)

belief_data = []
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    hidden = None
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        output, hidden, belief = model.get_belief(data, hidden)
        belief_data.append(belief.data.numpy())

belief_data_final = np.concatenate(belief_data[0].transpose(1,0,2), axis=0)
print ("Belief data shape: ", belief_data_final.shape)
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=5000)
tsne_results = tsne.fit_transform(belief_data_final)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
ax = sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=tsne_results[:,1],
    palette=sns.color_palette("hls", np.unique(tsne_results[:,1]).shape[0]),
    legend="full",
    alpha=0.3
)
plt.show()