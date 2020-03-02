import torch
import torch.utils.data as data
import numpy as np

class BayesFilterDataset(data.Dataset):
    def __init__(self, data, label, output_dim, sequence_length, batch_size, labels=None, mse=False):
        # Data should be numpy B x H x F where
        # B is the number of batches (independent trajectories)
        # H is the size of history
        # F is the size of features

        super(BayesFilterDataset, self).__init__()

        self.output_dim = output_dim
        self.data = data
        self.label = label
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.mse = mse

        assert data.shape[0] == label.shape[0]
        self.data, self.label = self.reshape(data, label)

        num_items = (self.data.shape[0] // self.batch_size) * self.batch_size
        self.data = self.data[:num_items]
        self.label = self.label[:num_items]
        # if self.labels is not None:
        #     self.labels = self.labels[:num_items]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def reshape(self, data, label, labels=None):
        assert data.shape[0] == label.shape[0]
        sequence_length = self.sequence_length

        n_chunks = data.shape[1] // sequence_length
        data = data[:, :n_chunks * sequence_length, :]

        data = data.reshape(data.shape[0], -1, sequence_length, data.shape[2])
        data = np.concatenate(data.transpose(1,0,2,3), axis=0)
        data = torch.Tensor(data).float()

        labels_reshaped = None

        # labels
        if not self.mse:
            label = np.repeat(label, n_chunks).reshape(label.shape[0], -1)
            label = np.concatenate(np.transpose(label))
            label = np.repeat(label, sequence_length).reshape(-1, sequence_length)
            label = torch.Tensor(label).long()
        else:
            label = label[:, :n_chunks*sequence_length, :]
            label = label.reshape(label.shape[0], -1, sequence_length, label.shape[2])
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
