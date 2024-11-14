import os
import numpy as np
import torch
import torch.nn as nn
import operator
from functools import reduce
from utils.util import ZFilter

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

import logging
log = logging.getLogger('root')


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        self.max_logvar = torch.tensor(-3 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)
        self.min_logvar = torch.tensor(-7 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)

        # Create or load networks
        self.networks = nn.ModuleList([self.create_network(n) for n in range(self.num_nets)]).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

    def get_output(self, output):
        """
        Argument:
          output: the raw output of a single ensemble member
        Return:
          mean and log variance
        """
        # print(f"Passed to get_output:{output}")
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_loss(self, targ, mean, logvar):
        """
        Compute negative log prediction probability.
        Args:
            targ: Target next state. (batch_size * state_dim)
            mean: predicted mean. (batch_size * state_dim)
            logvar: predicted log variance. (batch_size * state_dim)
        returns:
            loss (scalar)
        """
        negative_log_loss = ((mean - targ) ** 2) / torch.exp(logvar) + logvar
        return torch.sum(negative_log_loss) # sum across all observations and states

    def create_network(self, n):
        layer_sizes = [self.state_dim + self.action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, HIDDEN3_UNITS]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
        Argument:
          inputs: state and action inputs. Assumes that inputs are standardized.
          targets: resulting states
        Return:
            List containing the average loss of all the networks at each train iteration

        """
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
        assert inputs.shape[0] == targets.shape[0]

        num_samples = inputs.shape[0] 
        losses = []

        for i in range(num_train_itrs):
            print(f"Train iter {i} of {num_train_itrs}:")
            for n in range(self.num_nets):
                # sample minibatch of size B from D
                indices = torch.randint(0, num_samples, (batch_size,))
                batch_inputs = inputs[indices]
                batch_targets = targets[indices]
                self.opt.zero_grad()
                mean, logvar = self.forward(batch_inputs)[n]
                loss = self.get_loss(batch_targets, mean, logvar)
                loss.backward()
                self.opt.step()
                print(f"Train model {n} of {self.num_nets}, Loss: {loss.item()}")
                losses.append(loss.item())
        return losses
        # raise NotImplementedError