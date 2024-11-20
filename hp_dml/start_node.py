import torch
from torch.distributed import isend
import random

def start_node(rank, world_size, beta):
    """开始节点的逻辑实现"""
    # 生成向量 O，长度为 n + 1 (假设 n = world_size)
    O = torch.zeros(world_size + 1, dtype=torch.int)

    # 设置 O[m] 为上线的 GPU 数量（≤ 8）
    gpu_count = min(torch.cuda.device_count(), 8)
    O[rank] = gpu_count

    # 随机生成 j，满足 0 < j <= beta * n
    n = world_size
    j = random.randint(1, int(beta * n))  # 0 < j <= beta * n
    O[n] = j

    # 随机选择一个邻居并发送 O
    neighbors = [i for i in range(world_size) if i != rank]  # 除去自身的节点
    neighbor = random.choice(neighbors)
    isend(tensor=O, dst=neighbor)

    print(f"Start Node {rank}: Sent vector O {O.tolist()} to Node {neighbor}")
