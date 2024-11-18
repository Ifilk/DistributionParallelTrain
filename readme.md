# Distribution Parallel Train

## 迁移
ray 代码
```python
import ray

ray.init()

@ray.remote
def compute(x):
    return x ** 2

results = ray.get([compute.remote(i) for i in range(10)])
print(results)
```

torch.distributed 代码

```python
import torch
import torch.distributed as dist

def compute(rank, x):
    return x ** 2

def main():
    dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=0)
    rank = dist.get_rank()
    if rank == 0:
        data = [i for i in range(10)]
        for i, value in enumerate(data):
            dist.send(torch.tensor(value), dst=1)
    elif rank == 1:
        results = []
        for _ in range(10):
            tensor = torch.zeros(1)
            dist.recv(tensor, src=0)
            results.append(compute(rank, tensor.item()))
        print(results)

if __name__ == "__main__":
    main()
```

分布式模型训练
```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

def train(rank, world_size):
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    model = nn.Linear(10, 1)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 模拟数据
    data = torch.randn(10, 10)
    labels = torch.randn(10, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(10):  # 模拟训练
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Rank {rank} finished training.")

def main():
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
```


## NCCL配置

```bash
export NCCL_SOCKET_IFNAME=eth0  # 替换为你的网络接口名称，例如 eth0, enp0s3
export NCCL_DEBUG=INFO          # 启用 NCCL 调试信息（可选）
export NCCL_IB_DISABLE=0        # 如果使用 InfiniBand，启用此项（可选）
export NCCL_P2P_DISABLE=0       # 如果需要禁用 P2P 通信，设置为 1（可选）
export OMP_NUM_THREADS=4        # 限制线程数以避免资源争抢（可选）

export MASTER_ADDR="主节点IP地址"
export MASTER_PORT=29500  # 默认端口

```
测试nccl
```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make CUDA_HOME=/usr/local/cuda
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# nccl性能调优
export NCCL_TOPO_FILE=/path/to/topo.xml  # 如果有自定义拓扑文件
export NCCL_ALGO=ring                   # 指定算法（如 ring, tree, collnet）
export NCCL_CHECKS_DISABLE=1            # 禁用 NCCL 自检（若性能较低）
```

train.py
```python
import torch.distributed as dist
import torch

def setup_distributed(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)  # 设置当前进程使用的 GPU
    print(f"Rank {rank} initialized.")

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup_distributed(rank, world_size)
    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor)
    print(f"Rank {rank} has tensor value: {tensor.item()}")

if __name__ == "__main__":
    main()
```

使用torchrun启动分布式训练
```bash
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="主节点IP" train.py
```