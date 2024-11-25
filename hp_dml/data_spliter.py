import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from cjltest.divide_data import partition_dataset, select_dataset

def pic(args, class_num, indices, labels, flag):
    num_each_class = []
    for i in range(args.world_size):
        num = np.zeros(class_num)
        for index in indices[i]:
            num[labels[index]]+=1
        num_each_class.append(num)
    num_each_class = np.array(num_each_class) 
    
    t = {}
    for label in range(class_num):
        t[label] = num_each_class[:, label]
    name = [f'Device {i+1}' for i in range(args.world_size)]
    df=pd.DataFrame(t,index=name)

    plt.figure(figsize=(5,5),dpi=200)    
    df.plot(kind="bar",stacked=True,figsize=(10,5))
    plt.legend(loc="upper left", fontsize=10, ncol=10)
    
    path = f'{args.stdout}/{args.model}-{args.dataset}/partion_{args.partion_method}-test_{args.test_method}/'
    if not os.path.exists(path): os.mkdir(path)  
        
    title = 'Train-Set' if flag else 'Test-Set'
    plt.title(title)
    plt.savefig(path + f'/{title}-Distribution.png')

class Idxs2Tensor(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)

class DataSpliter():

    IID = 'iid'
    NONIID = 'noniid'

    TYPE_MNIST = 'mnist'
    TYPE_CIFAR10 = 'cifar10'
    TYPE_CIFAR100 = 'cifar100'
    TYPE_TINY_IMAGENET = 'TinyImageNet'

    def __init__(self, args, dataset, split_mode=IID, overlap=0.1,
                 total_class=10, data_type=TYPE_MNIST, num_nodes=5, bsz=128):

        self.args = args
        self.dataset = dataset

        self.total_class = total_class

        if data_type == DataSpliter.TYPE_MNIST or data_type == DataSpliter.TYPE_CIFAR10:
            self.total_class = 10
        elif data_type == DataSpliter.TYPE_CIFAR10:
            self.total_class = 100
        elif data_type == DataSpliter.TYPE_TINY_IMAGENET:
            self.total_class = 200

        if self.total_class > num_nodes:
            self.each_node_class = int(self.total_class / num_nodes)
        else:
            self.each_node_class = 1

        self.overlap = overlap
        self.split_mode = split_mode
        self.data_type = data_type
        self.num_nodes = num_nodes
        self.bsz = bsz

    def get_loaders(self):
        if self.split_mode == DataSpliter.IID:
            return self._get_loaders_iid()

        elif self.split_mode == DataSpliter.NONIID:
            loaders, _  = self._get_loaders_noniid()
            return loaders


    def _get_loaders_iid(self):
        loaders = []
        nodes = [v + 1 for v in range(self.num_nodes)]

        data = partition_dataset(self.dataset, nodes)
        for i in nodes:
            loaders.append(select_dataset(nodes, i, data, batch_size=self.bsz))

        return loaders

    def _get_loaders_noniid(self):
        loaders = []
        num = []
        if self.data_type == DataSpliter.TYPE_MNIST or \
                self.data_type == DataSpliter.TYPE_CIFAR10 or \
                self.data_type == DataSpliter.TYPE_CIFAR100 or \
                self.data_type == DataSpliter.TYPE_TINY_IMAGENET:

            # index of each node's data
            idxs_list = self._gen_noniid_idxs_list()
            
            pic(self.args, self.total_class, idxs_list, self.dataset.targets, 1)

            for idxs in idxs_list.values():
                loaders.append(DataLoader(Idxs2Tensor(self.dataset, list(idxs)),
                                          batch_size=self.bsz, shuffle=True))

                num.append(len(idxs))

        return loaders, num

    def _split_list_n_list(self, origin_list, n):
        if len(origin_list) % n == 0:
            cnt = len(origin_list) // n
        else:
            cnt = len(origin_list) // n + 1
        for i in range(0, n):
            # return generator
            yield origin_list[i * cnt: (i + 1) * cnt]

    def _gen_noniid_idxs_list(self):

        # split origin data's idx as list with num_nodes size
        current_idx = 0

        # idxs_list for each label
        label_idxs_list = [[] for _ in range(self.total_class)]
        for data in self.dataset:
            # data[1] is the label of data
            label_idxs_list[data[1]].append(current_idx)
            current_idx += 1

        # shuffle
        for i in range(len(label_idxs_list)):
            random.shuffle(label_idxs_list[i])

        # labels_each_node[i] contains the main labels for node i
        labels_each_node = [[] for _ in range(self.num_nodes)]
        # Record each label divided into several copies
        label_dict = {i: 0 for i in range(self.total_class)}
        x = self.each_node_class
        for i in range(self.num_nodes):
            for j in range(x):
                tmp_label = (i * x + j) % self.total_class
                labels_each_node[i].append(tmp_label)
                label_dict[tmp_label] += 1

        generator_list = []
        for i in range(len(label_idxs_list)):
            generator_list.append(self._split_list_n_list(label_idxs_list[i], label_dict[i]))

        # Assign data idxs to each node
        node_idxs_list = {i: np.array([]) for i in range(self.num_nodes)}
        for i in range(self.num_nodes):
            idxs = []
            for j in range(len(labels_each_node[i])):
                idxs.extend(next(generator_list[labels_each_node[i][j]]))
            random.shuffle(idxs)

            # add other labels idxs to node i
            main_labels_len = len(idxs)
            remain_class = self.total_class - self.each_node_class
            other_labels_count = int(main_labels_len * self.overlap / remain_class)
            for k in range(self.total_class):
                if k not in labels_each_node[i]:
                    if other_labels_count > len(label_idxs_list[k]):
                        other_labels_count = len(label_idxs_list[k])
                    idxs.extend(np.random.choice(label_idxs_list[k], other_labels_count))
            node_idxs_list[i] = np.array(idxs)
            

        return node_idxs_list












