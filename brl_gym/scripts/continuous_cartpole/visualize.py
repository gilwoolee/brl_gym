from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# from brl_gym.estimators.learnable_bf.bf_dataset import BayesFilterDataset
from brl_gym.estimators.learnable_bf.bf_dataset_2 import BayesFilterDataset
import brl_gym.estimators.learnable_bf.pt_util as pt_util
import brl_gym.estimators.learnable_bf.util as estimator_util
import torch
import multiprocessing
import torch.optim as optim
import pickle

import gym

from model import BayesFilterNet, BayesFilterNet2

SEQUENCE_SIZE = 100
with open("bf_data_lin_new.pkl", "rb") as f:
    edata = pickle.load(f)
    data = np.array([d[:SEQUENCE_SIZE - 1,:] for d in edata["data"]])
    output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["params"]])
    # output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["output"]])
    print ("Data: ", data.shape)
    print ("Output: ", output.shape)

np.random.seed(172)
perm_id = np.arange(len(data))
np.random.shuffle(perm_id)
data = data[perm_id, :, :5]
output = output[perm_id, 1:, :]

new_data = []
for i in range(data.shape[0]):
    features = []
    for j in range(data.shape[1] - 1):
        # features = []
        diff = data[i,j+1,:] - data[i,j,:]
        feat = np.concatenate((data[i,j+1], diff))
        features.extend([feat])
    new_data.extend([features])
data = np.array(new_data)
print ("Data: ", data.shape)
# print ("mean: ", np.mean(data), " dev: ", np.std(data))
# scaler = StandardScaler()
# data_d = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
# data_d = scaler.fit_transform(data_d)
# data = data_d.reshape((data.shape[0], data.shape[1], data.shape[2]))
# print ("mean: ", np.mean(data[:,:,0]))
for i in range(data.shape[2]):
    data[:,:,i] = (data[:,:,i] - np.mean(data[:,:,i])) / np.std(data[:,:,i])

for i in range(output.shape[-1]):
    output[:, :, i] = (output[:, :, i] - np.min(output[:, :, i])) / (np.max(output[:, :, i])- np.min(output[:, :, i]))

# data = (data - np.min(data)) / (np.max(data) - np.min(data))
# output = (output - np.min(output)) / (np.max(output) - np.min(output))

train_data = data[:int(len(data)*0.8)]
train_output = output[:int(len(output)*0.8)]
test_data = data[int(len(data)*0.8):]
test_output = output[int(len(output)*0.8):]

belief_dim = 16
sequence_length = 3
batch_size = 256

mse_mode = True

data_train = BayesFilterDataset(train_data, train_output, sequence_length, batch_size=batch_size, mse=mse_mode)
data_test = BayesFilterDataset(test_data, test_output, sequence_length, batch_size=batch_size, mse=mse_mode)

use_cuda = torch.cuda.is_available()
num_workers = multiprocessing.cpu_count()

kwargs = {'num_workers': num_workers,
            'pin_memory': True} if use_cuda else {}

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = BayesFilterNet(10, 2, 16, hidden_dim=256, n_layers=1).to(device)
# model = BayesFilterNet2(5, 2, belief_dim).to(device)

model.load_last_model("/home/rishabh/work/learnable_bf/data/2020-03-27_15-49-22/estimator_xx_checkpoints_mse")

test_loader  = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, **kwargs)

belief_data = []
model.eval()
test_loss = 0
correct = 0
total = 0
outputs = []
true_outputs = []

total_samples = 0
correct1 = 0
correct2 = 0
with torch.no_grad():
    hidden = None
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        output, hidden, belief = model.get_belief(data)
        # print ("Output dim: ", output)
        belief_data.append(hidden[0].cpu().data.numpy())
        # print ("Hidden dim: ", hidden.size())
        # print ("Belief dim: ", belief.shape)
        outputs.append(output.cpu().numpy())
        true_outputs.append(label.cpu().numpy())
        total_samples += label.size(0)
        temp = torch.abs(label - output)
        x = temp < 0.1
        correct1 += x[:,:,0].sum().item()
        correct2 += x[:,:,1].sum().item()

accuracy1 = (100 * correct1) / (total_samples * 3)
accuracy2 = (100 * correct2) / (total_samples * 3)
print ("Accuracy1: ", accuracy1, " Accuracy2: ", accuracy2)

outputs = np.array(outputs)
true_outputs = np.array(true_outputs)
print ("oA: ", outputs.shape)
x = np.concatenate(np.array(belief_data), axis=0)
y = np.concatenate(np.array(true_outputs), axis=0)

belief_data_final = np.reshape(x, (x.shape[0], -1))
# belief_data_final = np.reshape(y, (y.shape[0], -1))
# sc = StandardScaler()
# belief_data_final = sc.fit_transform(belief_data_final)
print ("final: ", belief_data_final.shape)
# belief_data_final = np.concatenate(belief_data[0].transpose(1,0,2), axis=0)
# print ("Belief data shape: ", belief_data_final.shape)
time_start = time.time()
pca = PCA(n_components=10)
tsne_results = pca.fit_transform(belief_data_final)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()

PCA_components = pd.DataFrame(tsne_results)

plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

X = PCA_components.iloc[:,:3]
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=3000)
tsne_results = tsne.fit_transform(belief_data_final)

df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

print ("TSNE shape: ", tsne_results.shape)
plt.figure(figsize=(16,10))
ax = sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=tsne_results[:,1],
    palette=sns.color_palette("hls", np.unique(tsne_results[:,1]).shape[0]),
    legend="full",
    alpha=0.3
)
plt.show()