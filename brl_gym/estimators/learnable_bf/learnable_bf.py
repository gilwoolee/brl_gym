import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from brl_gym.estimators.estimator import Estimator
from brl_gym.estimators.learnable_bf.bf_dataset import BayesFilterDataset
from . import pt_util

# This code structure is a modification of hw3
class BayesFilterNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_layers=2, dropout=0.0, nonlinear='relu'):

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
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.nonlinear(),
            nn.Linear(hidden_dim, output_dim)
            )
        self.best_accuracy = -1

    def forward(self, x, h=None):
        x = self.encoder(x)
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

    def inference(self, x, hidden_state=None, temperature=1):
        x = x.view(1, 1, -1)
        x, hidden_state = self.forward(x, hidden_state)
        x = x / max(temperature, 1e-20)
        x = F.softmax(x, dim=2)

        return x, hidden_state

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
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)


class LearnableBF(Estimator):
    def __init__(self, action_space, observation_space, belief_space, device=None, mse=True,
        nonlinear='relu', normalize=True):
        input_dim = observation_space.shape[0] + action_space.shape[0] + 1
        output_dim = belief_space.shape[0]
        self.hidden_state = None
        self.belief_dim = output_dim
        use_cuda = torch.cuda.is_available()
        self.device = "cuda"
        self.model = BayesFilterNet(input_dim, output_dim, nonlinear=nonlinear)
        self.model = self.model.to(self.device)
        self.temperature = 1
        self.mse = mse
        self.normalize = normalize
        print(self.model)

    def reset(self):
        self.hidden_state = None
        self.belief = np.ones(self.belief_dim) / self.belief_dim
        return self.belief.copy()

    def estimate(self, action, observation, **kwargs):
        if action is None:
            return self.reset()
        inp = np.concatenate([action, observation, [int(kwargs['done'])]], axis=0).ravel()
        inp = torch.Tensor(inp).float().to(self.device)
        if not self.mse:
            output, self.hidden_state = self.model.inference(inp, self.hidden_state, self.temperature)
        if self.mse:
            output, self.hidden_state = self.model.inference_mse(inp, self.hidden_state, normalize=self.normalize)
        self.belief = output.detach().cpu().numpy().ravel()
        self.belief /= np.sum(self.belief)

        return self.belief.copy()

    def get_belief(self):
        return self.belief.copy()

    def forward(self, action, observation, **kwargs):
        if action is None:
            return self.reset()
        inp = np.concatenate([action, observation, [int(kwargs['done'])]], axis=0).ravel()
        inp = torch.Tensor(inp).float().to(self.device)

        inp = inp.reshape(1, 1, -1)
        output, self.hidden_state = self.model(inp, self.hidden_state)

        x = output / max(self.temperature, 1e-20)
        x = F.softmax(x, dim=2)

        self.belief = x.detach().cpu().numpy().ravel()
        return x, output, self.hidden_state

    def set_bayes_filter(self, model_path):
        self.model.load_last_model(model_path)

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)


def max_sampling_strategy(model, output):
    return torch.argmax(output, dim=2).cpu().numpy()

def sample_sampling_strategy(model, output):
    output = output.cpu().numpy().squeeze()
    choices = np.arange(output.shape[-1])

    labels = []
    for data in output:
        data = data / np.sum(data)
        x = np.random.choice(choices, size=1, p=data)
        labels += [x]
    return labels

def generate_labels(model, device, seed_sequence, sampling_strategy='max', temperature=1):
    model.eval()

    with torch.no_grad():
        # Computes the initial hidden state from the prompt (seed words).
        hidden = None

        outputs = []
        for ind in seed_sequence:
            data = ind.to(device)
            output, hidden = model.inference(data, hidden, temperature=temperature)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)

        if sampling_strategy == 'max':
            outputs = max_sampling_strategy(model, outputs)

        elif sampling_strategy == 'sample':
            outputs = sample_sampling_strategy(model, outputs)
        outputs = np.concatenate(outputs).ravel()

        return outputs