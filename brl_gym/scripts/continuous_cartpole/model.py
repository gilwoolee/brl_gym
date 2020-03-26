import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import pt_util

# This code structure is a modification of hw3

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, belief_dim, hidden_dim=16, n_layers=2, dropout=0.0, nonlinear='relu'):
        super(SimpleNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.nonlinear = nn.ReLU if nonlinear == 'relu' else nn.Tanh
        
        self.encoder =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.nonlinear(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.nonlinear(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.nonlinear(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        self.fc = nn.Linear(hidden_dim // 4, output_dim)
        self.best_loss = 100.0
    
    def forward(self, x, h=None):
        enc = self.encoder(x)
        out = self.fc(enc)
        return out, h

    def get_belief(self, x, h=None, temperature=1):
        x = self.encoder(x)
        belief, h = self.gru(x, h)
        belief_2 = self.fc_prefinal(belief)
        out = self.fc(belief_2)
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
        # loss_val = F.mse_loss(prediction, label, reduction=reduction)
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
    
    def weight_init(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                if 'gru' in name:
                    continue
                print (name)
                nn.init.xavier_normal(param)

class SimpleNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, belief_dim, hidden_dim=16, n_layers=2, dropout=0.0, nonlinear='relu'):
        super(SimpleNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.nonlinear = nn.ReLU if nonlinear == 'relu' else nn.Tanh
        
        self.encoder =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.nonlinear(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.nonlinear(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.nonlinear(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        self.fc = nn.Linear(hidden_dim // 4, output_dim * 2)
        self.best_loss = 100.0
    
    def forward(self, x, h=None):
        enc = self.encoder(x)
        out = self.fc(enc)
        out = torch.reshape(out, (-1, 2, self.output_dim))
        out = F.softmax(out, dim=2)
        return out, h

    def get_belief(self, x, h=None, temperature=1):
        enc = self.encoder(x)
        out = self.fc(enc)
        out = torch.reshape(out, (-1, 2, self.output_dim))
        out = F.softmax(out, dim=2)
        return out, h, enc

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
        # loss_val = F.mse_loss(prediction, label, reduction=reduction)
        loss_val = F.binary_cross_entropy(prediction,
            label, reduction=reduction)
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
    
    def weight_init(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                if 'gru' in name:
                    continue
                print (name)
                nn.init.xavier_normal(param)

class BayesFilterNet(nn.Module):
    def __init__(self, input_dim, output_dim, belief_dim, hidden_dim=16, n_layers=2, dropout=0.0, nonlinear='relu'):

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
            nn.Linear(hidden_dim, belief_dim),
            # nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            self.nonlinear(),
            nn.Linear(belief_dim, output_dim),
            # self.nonlinear()
            )
        self.best_loss = 100.0

    def forward(self, x, h=None):
        out = self.encoder(x)
        out, h = self.gru(out, h)
        out = self.fc_prefinal(out)
        out = self.fc(out)
        return out, h

    def get_belief(self, x, h=None, temperature=1):
        x = self.encoder(x)
        belief, h = self.gru(x, h)
        belief_2 = self.fc_prefinal(belief)
        out = self.fc(belief_2)
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
        # loss_val = F.mse_loss(prediction, label, reduction=reduction)
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

class BayesFilterNet2(nn.Module):
    def __init__(self, input_dim, output_dim, belief_dim, hidden_dim=16, n_layers=2, dropout=0, nonlinear='relu'):

        super(BayesFilterNet2, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.nonlinear = nn.ReLU if nonlinear == 'relu' else nn.Tanh
        self.nonlinear_fc = nn.Sigmoid

        self.encoder =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.nonlinear(),
            nn.Linear(hidden_dim, hidden_dim),
            self.nonlinear()
            )
        self.gru = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True,
            dropout=dropout)
        # self.gru = nn.LSTMCell(hidden_dim, hidden_dim)
        self.fc_prefinal = nn.Sequential(
            # self.nonlinear(),
            nn.Linear(hidden_dim, belief_dim))
        self.fc1_1 = nn.Sequential(
            self.nonlinear(),
            # nn.Dropout(0.1),
            nn.Linear(belief_dim, belief_dim // 2),
            self.nonlinear())
            
        self.fc1_2 = nn.Sequential(
            # nn.Linear(belief_dim // 2, belief_dim // 2),
            # self.nonlinear(),
            nn.Linear(belief_dim // 2, 1),
            # self.nonlinear()
            # self.nonlinear_fc()
            )
        self.fc2_1 = nn.Sequential(
            self.nonlinear(),
            # nn.Dropout(0.1),
            nn.Linear(belief_dim, belief_dim // 2),
            self.nonlinear()
        )
        self.fc2_2 = nn.Sequential(
            # nn.Linear(belief_dim // 2, belief_dim // 2),
            # self.nonlinear(),
            nn.Linear(belief_dim // 2, 1),
            # self.nonlinear()
            # self.nonlinear_fc()
            )
        self.best_loss = 100.0

    def forward(self, x, h=None):
        x = self.encoder(x)
        out, h = self.gru(x, h)
        out = self.fc_prefinal(out)
        out1 = self.fc1_2(self.fc1_1(out))
        out2 = self.fc2_2(self.fc2_1(out))
        out = torch.cat((out1, out2), dim=2)
        return out, h

    def get_belief(self, x, h=None, temperature=1):
        x = self.encoder(x)
        out, h = self.gru(x, h)
        belief = self.fc_prefinal(out)
        out1 = self.fc1_1(belief)
        out1_1 = self.fc1_2(out1)
        out2 = self.fc2_1(belief)
        out2_1 = self.fc2_2(out2)
        belief_2 = torch.cat((out1, out2), dim=2)
        out = torch.cat((out1_1, out2_1), dim=2)
        return out, h, belief_2

    def inference_mse(self, x, hidden_state=None, normalize=True):
        x = x.view(1, 1, -1)
        x, hidden_state = self.forward(x, hidden_state)
        if normalize:
            x = x - torch.min(x)
            x /= torch.max(x)

        return x, hidden_state

    def loss(self, prediction, label, reduction='mean'):
        loss_val_1 = F.mse_loss(prediction[:,:,0], label[:,:,0], reduction=reduction)
        loss_val_2 = F.mse_loss(prediction[:,:,1], label[:,:,1], reduction=reduction)
        # loss_val = F.mse_loss(prediction.view(-1, self.output_dim,),
        #     label.view(-1, self.output_dim,), reduction=reduction)
        # print ("Loss1: ", loss_val_1.item(), " Loss2: ", loss_val_2.item())
        alpha = 1.0
        return loss_val_1 + alpha * loss_val_2

    def mse_loss(self, prediction, label, reduction='mean'):
        loss_val_1 = F.mse_loss(prediction.view(-1, self.output_dim,)[:,0],
            label.view(-1, self.output_dim,)[:,0], reduction=reduction)
        loss_val_2 = F.mse_loss(prediction.view(-1, self.output_dim,)[:,1],
            label.view(-1, self.output_dim,)[:,1], reduction=reduction)

        # loss_val_1 = F.mse_loss(prediction[:,:,0], label[:,:,0], reduction=reduction)
        # loss_val_2 = F.mse_loss(prediction[:,:,1], label[:,:,1], reduction=reduction)
        # loss_val = F.mse_loss(prediction.view(-1, self.output_dim,),
        #     label.view(-1, self.output_dim,), reduction=reduction)
        # print ("Loss1: ", loss_val_1.item(), " Loss2: ", loss_val_2.item())
        alpha = 0.5
        # return loss_val_1 + alpha * loss_val_2
        return (loss_val_1 + loss_val_2)
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
        self.h = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
    
    def weight_init(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                if 'gru' in name:
                    continue
                print (name)
                nn.init.xavier_normal(param)

def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()

class BayesFilterNet3(nn.Module):
    def __init__(self, input_dim, output_dim, belief_dim, hidden_dim=16, n_layers=2, dropout=0, nonlinear='relu'):

        super(BayesFilterNet2, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.nonlinear = nn.ReLU if nonlinear == 'relu' else nn.Tanh
        self.nonlinear_fc = nn.Sigmoid

        self.encoder =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.nonlinear(),
            nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True,
            dropout=dropout)
        self.fc_prefinal = nn.Sequential(
            # self.nonlinear(),
            nn.Linear(hidden_dim, belief_dim))
        self.fc1_1 = nn.Sequential(
            self.nonlinear(),
            # nn.Dropout(0.1),
            nn.Linear(belief_dim, belief_dim // 2),
            self.nonlinear())
            
        self.fc1_2 = nn.Sequential(
            # nn.Linear(belief_dim // 2, belief_dim // 2),
            # self.nonlinear(),
            nn.Linear(belief_dim // 2, 1),
            # self.nonlinear()
            self.nonlinear_fc()
            )
        self.fc2_1 = nn.Sequential(
            self.nonlinear(),
            # nn.Dropout(0.1),
            nn.Linear(belief_dim, belief_dim // 2),
            self.nonlinear()
        )
        self.fc2_2 = nn.Sequential(
            # nn.Linear(belief_dim // 2, belief_dim // 2),
            # self.nonlinear(),
            nn.Linear(belief_dim // 2, 1),
            # self.nonlinear()
            self.nonlinear_fc()
            )
        self.best_loss = 100.0

    def forward(self, x, h=None):
        x = self.encoder(x)
        out, self.h = self.gru(x, self.h)
        out = self.fc_prefinal(out)
        out1 = self.fc1_2(self.fc1_1(out))
        out2 = self.fc2_2(self.fc2_1(out))
        out = torch.cat((out1, out2), dim=2)
        return out, h

    def get_belief(self, x, h=None, temperature=1):
        x = self.encoder(x)
        out, h = self.gru(x, h)
        belief = self.fc_prefinal(out)
        out1 = self.fc1_1(belief)
        out1_1 = self.fc1_2(out1)
        out2 = self.fc2_1(belief)
        out2_1 = self.fc2_2(out2)
        belief_2 = torch.cat((out1, out2), dim=2)
        out = torch.cat((out1_1, out2_1), dim=2)
        return out, h, belief_2

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
        loss_val_1 = F.mse_loss(prediction[:,:,0], label[:,:,0], reduction=reduction)
        loss_val_2 = F.mse_loss(prediction[:,:,1], label[:,:,1], reduction=reduction)
        # loss_val = F.mse_loss(prediction.view(-1, self.output_dim,),
        #     label.view(-1, self.output_dim,), reduction=reduction)
        # print ("Loss1: ", loss_val_1.item(), " Loss2: ", loss_val_2.item())
        alpha = 0.2
        # return 2 * ((1 - alpha) * loss_val_1 + alpha * loss_val_2)
        return (loss_val_1 + loss_val_2) / 2
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
        self.h = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
