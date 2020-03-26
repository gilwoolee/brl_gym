import torch
import torch.utils.data as data
import numpy as np

class BayesFilterDataset(data.Dataset):
    def __init__(self, data, label, sequence_length, batch_size, labels=None, mse=False):
        # Data should be numpy B x H x F where
        # B is the number of batches (independent trajectories)
        # H is the size of history
        # F is the size of features

        super(BayesFilterDataset, self).__init__()

        self.data = torch.Tensor(data).float()
        self.label = torch.Tensor(label).float()
        # np.linspace(0.5, 2, 5)

    def __feature_len__(self):
        return self.data.shape[-1], self.label.shape[-1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # print ("Dataset shape: ", self.data[idx].size())
        # return self.data[idx], self.label[idx] # (obs, action, reward)
        return self.data[idx], self.label[idx] # (obs)
        # return np.concatenate((self.data[idx][:,:4], self.data[idx][:,5:]), axis=1) , self.label[idx] # (obs, reward)

    def reshape(self, data, label, labels=None):
        assert data.shape[0] == label.shape[0]
        sequence_length = self.sequence_length

        n_chunks = data.shape[1] // sequence_length
        data = data[:, :n_chunks * sequence_length, :]

        data = data.reshape(data.shape[0], -1, sequence_length, data.shape[2])
        data = np.concatenate(data.transpose(1,0,2,3), axis=0)
        data = torch.Tensor(data).float()

        labels_reshaped = None

        # print ("Label: ", label.shape)
        # label = np.expand_dims(label, axis=2) # Not needed for params as labels
        if not self.mse:
            label = np.repeat(label, n_chunks).reshape(label.shape[0], -1)
            label = np.concatenate(np.transpose(label))
            label = np.repeat(label, sequence_length).reshape(-1, sequence_length)
            label = torch.Tensor(label).long()
        else:
            label = label[:, :n_chunks*sequence_length, :]
            label = label.reshape(label.shape[0], -1, sequence_length, label.shape[2])
            # print ("Label shape: ", label[-1,0,:,:])
            # print ("Rearrange: ", label.shape)
            label = np.concatenate(label.transpose(1,0,2,3), axis=0)
            label = torch.Tensor(label).float()
            # labels = np.repeat(labels, n_chunks).reshape(labels.shape[0], -1)
            # labels = np.concatenate(np.transpose(labels))
            # labels = np.repeat(labels, sequence_length).reshape(-1, sequence_length)
            # labels_reshaped = torch.Tensor(labels).long()

        return data, label

    def add_item(self, data, label):
        data, label = self.reshape(data, label)
        self.data = torch.cat([self.data, data], dim=0)
        self.label = torch.cat([self.label, label], dim=0)

        num_items = (self.data.shape[0] // self.batch_size) * self.batch_size
        self.data = self.data[:num_items]
        self.label = self.label[:num_items]
