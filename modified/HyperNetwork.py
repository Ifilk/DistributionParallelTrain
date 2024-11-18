# -*- coding: utf-8 -*-
import os
import pickle
from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn import functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class HyperNetwork(nn.Module):
    def __init__(self, args, layer_names, client_id):
        super(HyperNetwork, self).__init__()
        self.args = args
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.client_num = self.args.world_size

        self.embedding = nn.Embedding(self.client_num, self.args.embedding_dim, device=self.device)
        
        self.blocks_name = set(n.split(".")[0][:-1] for n in layer_names)
        
        self.cache_dir = self.args.cache_dir
        if not os.path.isdir(self.cache_dir):
            # os.system(f"mkdir -p {self.cache_dir}")
            os.makedirs(self.cache_dir)

        if os.listdir(self.cache_dir) != self.client_num:
            for client_id in range(self.client_num):
                with open(f'{self.cache_dir}/world_size_{self.args.world_size}/Client_{self.client_id}.pkl', "wb") as f:
                    pickle.dump(
                        {
                            "mlp": nn.Sequential(
                                nn.Linear(self.args.embedding_dim, self.args.hidden_dim),
                                nn.ReLU(),
                                nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
                                nn.ReLU(),
                                nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
                                nn.ReLU(),
                            ),
                            
                            "fc": {
                                name: Linear(self.args.hidden_dim, self.client_num)
                                for name in self.blocks_name
                            },
                        },
                        f,
                    )
                    
        # for tracking the current client's hn parameters
        self.current_client_id: int = None
        self.mlp: nn.Sequential = None
        self.fc_layers: Dict[str, Linear] = {}

    def mlp_parameters(self) -> List[nn.Parameter]:
        return list(filter(lambda p: p.requires_grad, self.mlp.parameters()))


    def fc_layer_parameters(self) -> List[nn.Parameter]:
        params_list = []
        for block, fc in self.fc_layers.items():
            params_list += list(filter(lambda p: p.requires_grad, fc.parameters()))
        return params_list

    def emd_parameters(self) -> List[nn.Parameter]:
        return list(self.embedding.parameters())

    def forward(self, client_id):
        self.current_client_id = client_id

        emd = self.embedding(torch.tensor(client_id, dtype=torch.long, device=self.device))
        self.load_hn()
        feature = self.mlp(emd)
        ps = {block: F.softmax(self.fc_layers[block](feature)) for block in self.blocks_name}

        return [torch.max(p)[1] for p in ps ]
    

    def save_hn(self):
        for block, param in self.fc_layers.items():
            self.fc_layers[block] = param.cpu()
        with open(f'{self.cache_dir}/world_size_{self.args.world_size}/Client_{self.client_id}.pkl', "wb") as f:
            pickle.dump(
                {"mlp": self.mlp.cpu(), "fc": self.fc_layers}, f,
            )
        self.mlp = None
        self.fc_layers = {}
        self.current_client_id = None

    def load_hn(self):
        with open(f'{self.cache_dir}/world_size_{self.args.world_size}/Client_{self.client_id}.pkl', "rb") as f:
            parameters = pickle.load(f)
        self.mlp = parameters["mlp"].to(self.device)
        for block, param in parameters["fc"].items():
            self.fc_layers[block] = param.to(self.device)

    def clean_models(self):
        if os.path.isdir(self.cache_dir):
            del_files(self.cache_dir)
            
    def update(self, diff, model_params):    

        hn_grads = torch.autograd.grad(
            outputs=list(filter(lambda param: param.requires_grad, model_params)),
            inputs=self.mlp_parameters() + self.fc_layer_parameters() + self.emd_parameters(),
            grad_outputs=list(map(lambda tup: tup[1], filter(lambda tup: tup[1].requires_grad, diff.items()))),
            # allow_unused=True,
            retain_graph=True
        )
        
        
        mlp_grads = hn_grads[: len(self.mlp_parameters())]
        fc_grads = hn_grads[
            len(self.mlp_parameters()) : len(
                self.mlp_parameters() + self.fc_layer_parameters()
            )
        ]
        emd_grads = hn_grads[
            len(self.mlp_parameters() + self.fc_layer_parameters()) :
        ]

        for param, grad in zip(self.fc_layer_parameters(), fc_grads):
            if grad is not None:
                param.data -= self.args.hn_lr * grad

        for param, grad in zip(self.mlp_parameters(), mlp_grads):
            param.data -= self.args.hn_lr * grad

        for param, grad in zip(self.emd_parameters(), emd_grads):
            param.data -= self.args.hn_lr * grad
            
        self.save_hn()
            
def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)
            
def get_model(model):
    param = []
    for p in model.parameters():
        param.append(p.detach().clone())
    return param     

