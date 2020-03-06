import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import pt_util

# This code structure is a modification of hw3
class BayesFilterNet(nn.Module):
    def __init__(self, input_dim, output_dim, belief_dim, hidden_dim=64, n_layers=2, dropout=0.0, nonlinear='relu'):

        super(BayesFilterNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.nonlinear = nn.ReLU if nonlinear == 'relu' else nn.Tanh

        self.encoder =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.nonlinear(),
            nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True,
            dropout=dropout)
        self.fc_prefinal = nn.Sequential(
            nn.Linear(hidden_dim, belief_dim))
        self.fc = nn.Sequential(
            self.nonlinear(),
            nn.Linear(belief_dim, output_dim)
            )
        self.best_loss = 100.0

    def forward(self, x, h=None):
        x = self.encoder(x)
        out, h = self.gru(x, h)
        out = self.fc_prefinal(out)
        out = self.fc(out)
        return out, h

    def get_belief(self, x, h=None, temperature=1):
        x = self.encoder(x)
        out, h = self.gru(x, h)
        belief = self.fc_prefinal(out)
        out = self.fc(belief)
        return out, h, belief

    def inference_mse(self, x, hidden_state=None, normalize=True):
        x = x.view(1, 1, -1)
        x, hidden_state = self.forward(x, hidden_state)
        if normalize:
            x = x - torch.min(x)
            x /= torch.max(x)

        return x, hidden_state

    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction.view(-1, self.output_dim,),
            label.view(-1), reduction=reduction)
        return loss_val

    def mse_loss(self, prediction, label, reduction='mean'):
        loss_val = F.mse_loss(prediction.view(-1, self.output_dim,),
            label.view(-1, self.output_dim,), reduction=reduction)
        return loss_val

    # Saves the current model
    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, loss, file_path, num_to_keep=1):
        if loss < self.best_loss:
            self.save_model(file_path, num_to_keep)
            self.best_loss = loss

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
