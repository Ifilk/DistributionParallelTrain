from collections import OrderedDict

import numpy as np
import torch
from torch import optim, nn

from hp_dml.config import HyperNetworkConfiguration, ModelConfiguration
from hp_dml.dist import Node

from enum import IntEnum

from hp_dml.hyper_network import HyperNetwork
from hp_dml.mr_sgd import MR_SGD
from hp_dml.regularization import Regularization
from hp_dml.utils import clone_parameters, model_power, weighted_sum


class MetaMessageType(IntEnum):
    GROUP = 0
    P_AGG = 1
    P_SYN = 2


class ModuleProxy(Node):
    def __init__(self,
                 rank: int,
                 node_amount: int,
                 group_density: float,
                 model: nn.Module,
                 train_loader,
                 model_config: ModelConfiguration,
                 hn_config: HyperNetworkConfiguration):
        # lr, iter_method, model
        super().__init__(rank, node_amount, group_density, model)
        # self.layer_name = list(self.model.get_weights().keys())
        self.layer_name = {k: v.cpu() for k, v in model.state_dict().items()}.keys()
        self.hyper_network = HyperNetwork(hn_config, self.layer_name, rank)

        self.train_loader = train_loader
        self.model_config = model_config
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.model_config.learning_rate,
                                   momentum=0.9, weight_decay=5e-4)
        self.mr_optimizer = MR_SGD(self.model.parameters(), lr=self.model_config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.round = 0
        self.agged_model = {}
        if self.model_config.iter_method == 'iteration': # Iterator for 'iteration' method
            self.data_iteration = iter(self.train_loader)

    @property
    def p(self):
        return self.hyper_network(self._rank)

    def update_hyper_network(self):
        self.hyper_network.update(self.delta, self.agged_model.values())

    def train(self, meta_message):
        self.model.train()

        frz_model_params = clone_parameters(self.model)

        # epoch method: train all mini-batch each round
        if self.model_config.iter_method == 'epoch':
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # obtain data
                data, target = data.cuda(), target.cuda()

                # train
                self.optimizer.zero_grad()
                if self.model_config.model_name == 'LR' or self.model_config.model_name == 'LeNet':
                    output = self.model(data)
                else:
                    output, embedding = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                self.optimizer.step()
                # iteration mode: train 1 mini-batch each round
        elif self.model_config.iter_method == 'iteration':
            # obtain data
            try:
                inputs, targets = next(self.data_iteration)
            except StopIteration:
                self.data_iteration = iter(self.train_loader)
                inputs, targets = next(self.data_iteration)

            inputs, targets = inputs.cuda(), targets.cuda()

            # train
            self.optimizer.zero_grad()
            if self.model_config.model_name == 'LR' or self.model_config.model_name == 'LeNet':
                outputs = self.model(inputs)
            else:
                outputs, embedding = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if self.model_config.model_name == 'Lasso':
                # L1 Regularization
                loss += Regularization(self.model, 0.001, p=1)(self.model)
            elif self.model_config.model_name == 'RR':
                # L2 Regularization
                loss += Regularization(self.model, 0.001, p=0)(self.model)

            loss.backward()
            self.optimizer.step()

            # _, predicted = outputs.max(1)

        self.delta = OrderedDict(
            {k: p1 - p0 for (k, p1), p0 in
             zip(self.model.state_dict(keep_vars=True).items(), frz_model_params.values())})

        # TODO P Power
        p1 = ...
        p2 = ...
        powered_model = model_power(self.model.get_weights(), p1, p2)
        grad = self.model.get_gradients()

        return powered_model, grad

    def gradients_step(self, gradients, scale_factor, p_loc):
        # no usage parameter: scale_factor
        self.mr_optimizer.zero_grad()
        self.model.set_gradients(gradients)
        # TODO p1 p2
        p1 = ...
        p2 = ...
        scale_factor = pow(10, -(np.mean(p1, p2))) if p1 != 1 and p2 != 1 else 1

        self.mr_optimizer.step(scale_factor, self.layer_name)

        powered_model = model_power(self.model.get_weights(), 1 / p1, 1 / p2)

        return powered_model

    def aggregation(self, models_list, weight_matrix):
        alpha = {}
        # ['module.0.weight', 'module.0.bias', ...]
        names = set(n.split(".")[0][:-1] for n in self.layer_name)
        for name in names:
            # TODO consistency don't need to calculate dynamically
            alpha[name] = torch.tensor([1 / self._node_amount
                                        for _ in range(self._node_amount)])
        for layer in self.layer_name:
            # meaningless slice [:-1]
            layer_alpha = alpha[layer.split(".")[0][:-1]] * torch.tensor(weight_matrix)
            weight = layer_alpha / sum(layer_alpha)
            # meaningless slice [:-1]
            alpha[layer.split(".")[0][:-1]] = weight
            layer_param = [models_list[i][layer] for i in range(len(models_list))]
            if len(layer_param) == len(weight):
                raise ValueError("The number of tensors must match the number of weights.")
            # N^{(t)}_i + 1 is the number of devices neighboring device i
            # e^{(t)}_{i,j} = (N^{(t)}_i + 1) ^ {-1}
            # a^{(t)}_{i,j} = min{e^{(t)}_{i,j}, e^{(t)}_{i,j}}
            # Aggregate with a^{t}_{i,j}
            # The shape of weight is models_list * 1
            self.agged_model[layer] = weighted_sum(layer_param, weight)

        return alpha, self.agged_model

    def update_model(self) -> torch.Tensor:
        pass