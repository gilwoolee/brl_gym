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

from brl_gym.estimators.learnable_bf.bf_dataset_2 import BayesFilterDataset
import brl_gym.estimators.learnable_bf.pt_util as pt_util
import brl_gym.estimators.learnable_bf.util as estimator_util
import torch
import multiprocessing
import torch.optim as optim
import pickle

from model import BayesFilterNet, BayesFilterNet2, SimpleNN, SimpleNNClassifier


# SEQUENCE_SIZE = 100
with open("bf_data_lin_new.pkl", "rb") as f:
    edata = pickle.load(f)
    data = edata["data"]
    output = edata["params"]
    # output = np.array([l[:SEQUENCE_SIZE - 1] for l in edata["output"]])
    print ("Data: ", data.shape)
    print ("Output: ", output.shape)

params = {
    'belief_dim': 16,
    'hidden_dim': 256,
    'n_layers': 1,
    'sequence_length': 10,
    'batch_size': 512,
    'lr': 0.005,
    'num_epochs': 2000,
    'data_file': "bf_data_lin_new.pkl"
}

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

# data = (data - np.mean(data)) / np.std(data)
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

# model = BayesFilterNet2(5, 2, belief_dim).to(device)

model.load_last_model("/home/rishabh/work/brl_gym/brl_gym/scripts/continuous_cartpole/data/2020-03-25_01-36-56/estimator_xx_checkpoints_mse")
test_loader  = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, **kwargs)

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
        belief_data.append(belief.cpu().data.numpy())
        # print ("Belief dim: ", belief.shape)
        outputs.append(output.cpu().numpy())
        true_outputs.append(label.cpu().numpy())
        _, predicted1 = torch.max(output[:,0].data, 1)
        _, predicted2 = torch.max(output[:,1].data, 1)
        _, label1 = torch.max(label[:,0].data, 1)
        _, label2 = torch.max(label[:,1].data, 1)
        # print ("Pred:", predicted.size())
        total_samples += label.size(0)
        # print ("Pred: ", predicted1.size(), "lab: ", label1.size())
        correct1 += (predicted1 == label1).sum().item()
        correct2 += (predicted2 == label2).sum().item()

accuracy1 = 100 * correct1 / total_samples
accuracy2 = 100 * correct2 / total_samples
print ("Accuracy1: ", accuracy1, " Accuracy2: ", accuracy2)
exit()
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