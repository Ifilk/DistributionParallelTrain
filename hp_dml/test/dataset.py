import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class Loader_data(Dataset):
    def __init__(self,x,y):
        super(Loader_data,self).__init__()
        self.x=x
        self.y=y
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,idx):
        return(self.x[idx],self.y[idx])


def init_dataset(args):
    if args.dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(root=f'{args.data_dir}', train=True,
                                                download=False, transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(root=f'{args.data_dir}', train=False,
                                               download=True, transform=torchvision.transforms.ToTensor())
        return ssp_data_spliter(args, train_data, test_data)
    elif args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=f'{args.data_dir}', train=True,
                                                download=True, transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.CIFAR10(root=f'{args.data_dir}', train=False,
                                               download=True, transform=torchvision.transforms.ToTensor())
        return ssp_data_spliter(args, train_data, test_data)

    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=f'{args.data_dir}', train=True,
                                                download=True, transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.CIFAR100(root=f'{args.data_dir}', train=False,
                                               download=True, transform=torchvision.transforms.ToTensor())
        return ssp_data_spliter(args, train_data, test_data)


def ssp_data_spliter(args, train_data, test_data):
    if args.iid == 0:
        train_spliter = DataSpliter(args, train_data, split_mode='noniid',
                                    overlap=args.data_overlap, data_type=args.dataset,
                                    num_nodes=args.world_size, bsz=args.batch_size)
    else:
        train_spliter = DataSpliter(args, train_data, split_mode='iid',
                                    overlap=args.data_overlap, data_type=args.dataset,
                                    num_nodes=args.world_size, bsz=args.batch_size)
    train_loaders = train_spliter.get_loaders()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    data_size_partition = [len(loader.dataset) for loader in train_loaders]
    return train_loaders, test_loader

def init_dataset_new(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(root=f'{args.data_dir}', train=True,
                                                download=True, transform=trans_mnist)
        test_data = datasets.MNIST(root=f'{args.data_dir}', train=False,
                                               download=True, transform=trans_mnist)
        class_num = len(train_data.classes)
        return data_spliter(args, class_num, train_data, test_data)
    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR10(root=f'{args.data_dir}', train=True,
                                                download=True, transform=trans_cifar)
        test_data = datasets.CIFAR10(root=f'{args.data_dir}', train=False,
                                               download=True, transform=trans_cifar)
        class_num = len(train_data.classes)
        return data_spliter(args, class_num, train_data, test_data)
    elif args.dataset == 'cifar100':
        train_data = datasets.CIFAR100(root=f'{args.data_dir}', train=True,
                                                download=True, transform=transforms.ToTensor())
        test_data = datasets.CIFAR100(root=f'{args.data_dir}', train=False,
                                               download=True, transform=transforms.ToTensor())
        class_num = len(train_data.classes)
        return data_spliter(args, class_num, train_data, test_data)

def dirichlet_spliter(train_labels, alpha, n_clients):

    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)


    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def label_spliter(args, train_data, test_data):
    all_data = torch.cat([torch.as_tensor(train_data.data, dtype=torch.float32),
                              torch.as_tensor(test_data.data, dtype=torch.float32)])

    all_labels = torch.cat([torch.as_tensor(train_data.targets),
                                  torch.as_tensor(test_data.targets)])
    
    splited_idx = dirichlet_spliter(all_labels, alpha=args.alpha, n_clients=args.world_size)
    
    train_labels_idx = []
    test_labels_idx = []
    for single in splited_idx:
        train_idx, test_idx = train_test_split(single, test_size=0.2, random_state=2025)
        train_labels_idx.append(train_idx)
        test_labels_idx.append(test_idx)

    return all_data, all_labels, train_labels_idx, test_labels_idx

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

    # 画图
    plt.figure(figsize=(5,5),dpi=200)    
    df.plot(kind="bar",stacked=True,figsize=(10,5))
    plt.legend(loc="upper left", fontsize=10, ncol=10)
    
    path = f'{args.stdout}/{args.model}-{args.dataset}/partion_{args.partion_method}-test_{args.test_method}/'
    if not os.path.exists(path): os.mkdir(path)    
        
    title = 'Train-Set' if flag else 'Test-Set'
    plt.title(title)
    plt.savefig(path + f'{args.dataset}-{title}-Distribution.png')
    
    
def data_spliter(args, class_num, train_data, test_data):
    train_loaders = []
    test_loaders = []
    all_data, all_labels, train_labels_idx, test_labels_idx = label_spliter(args, train_data, test_data)
    pic(args, class_num, train_labels_idx, all_labels, 1)
    pic(args, class_num, test_labels_idx, all_labels, 0)
      
    for client_idx in range(args.world_size):

        single_train_data = torch.stack([all_data[idx].unsqueeze(0) for idx in train_labels_idx[client_idx]], dim=0) if args.dataset == 'mnist' else \
            torch.stack([all_data[idx].permute(2, 0, 1) for idx in train_labels_idx[client_idx]], dim=0) 
        single_train_label = torch.tensor([all_labels[idx] for idx in train_labels_idx[client_idx]])
        single_test_data = torch.stack([all_data[idx].unsqueeze(0) for idx in test_labels_idx[client_idx]], dim=0) if args.dataset == 'mnist' else \
            torch.stack([all_data[idx].permute(2, 0, 1) for idx in test_labels_idx[client_idx]], dim=0) 
        single_test_label = torch.tensor([all_labels[idx] for idx in test_labels_idx[client_idx]])
        
        train_loader=DataLoader(
            Loader_data(single_train_data, single_train_label),
            batch_size=args.batch_size,
            shuffle=True)    
        test_loader=DataLoader(
            Loader_data(single_test_data, single_test_label),
            batch_size=args.batch_size,
            shuffle=True)    
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        
    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
        
    return train_loaders, test_loaders, global_test_loader




