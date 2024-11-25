import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn as nn
from datetime import timedelta

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# world_size = the amount of nodes * the amount of process per a node
# process - (local_rank, global_rank) local_rank regard as the number of GPU on a Node approximately
def setup(global_rank, world_size):
    # 配置Master Node的信息
    os.environ['MASTER_ADDR'] = 'XXX.XXX.XXX.XXX'
    os.environ['MASTER_PORT'] = 'XXXX'

    # 初始化Process Group
    # 关于init_method, 参数详见https://pytorch.org/docs/stable/distributed.html#initialization
    dist.init_process_group("nccl", init_method='env://', rank=global_rank, world_size=world_size, timeout=timedelta(seconds=5))

def cleanup():
    dist.destroy_process_group()

def run_demo_checkpoint(local_rank, args):
    # 计算global_rank和world_size
    global_rank = local_rank + args.node_rank * args.nproc_per_node
    world_size = args.nnode * args.nproc_per_node
    setup(global_rank=global_rank, world_size=world_size)
    print(f"Running DDP checkpoint example on rank {global_rank}.")

    # 设置seed
    torch.manual_seed(args.seed)

    model = ToyModel().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"

    if global_rank == 0:
        # 只在Process0中保存模型
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # barrier(): 可以理解为只有当所有Process都到达了这一步才能继续往下执行
    # 以此保证其他Process只有在Process0完成保存后才可能读取模型
    dist.barrier()

    # 配置`map_location`.
    map_location = torch.device(f'cuda:{local_rank}')

    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(local_rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    print(outputs)

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if global_rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--nproc_per_node', type=int)
    parser.add_argument('--nnode', type=int)
    parser.add_argument('--node_rank', type=int)
    args = parser.parse_args()

    # mp.spawn(run_demo, args=(args,), nprocs=args.nproc_per_node)