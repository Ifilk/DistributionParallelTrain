import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import random

# 初始化分布式环境
def init_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 主进程 - 任务调度器
def master(rank, world_size, tasks):
    init_distributed(rank, world_size)

    print(f"[Master] Rank {rank} initialized.")

    # 循环分发任务
    for task_id, task_data in enumerate(tasks):
        target_worker = task_id % (world_size - 1) + 1  # 分配到具体 worker
        print(f"[Master] Assigning Task ID {task_id} to Worker {target_worker}.")
        dist.send(torch.tensor([task_id]), dst=target_worker)
        dist.send(torch.tensor(task_data), dst=target_worker)

    # 发送结束信号
    for worker_rank in range(1, world_size):
        dist.send(torch.tensor([-1]), dst=worker_rank)
    print("[Master] All tasks dispatched.")

# 工作进程 - 等待任务执行
def worker(rank, world_size):
    init_distributed(rank, world_size)

    print(f"[Worker] Rank {rank} initialized.")

    while True:
        # 接收任务 ID
        task_id = torch.zeros(1, dtype=torch.long)
        dist.recv(task_id, src=0)

        if task_id.item() == -1:  # 结束信号
            print(f"[Worker] Rank {rank} received termination signal.")
            break

        # 接收任务数据
        task_data = torch.zeros(1)
        dist.recv(task_data, src=0)

        print(f"[Worker] Rank {rank} received Task ID {task_id.item()} with data {task_data.item()}.")

        # 执行任务
        result = task_data.item() ** 2  # 简单平方运算
        time.sleep(random.uniform(0.5, 1.5))  # 模拟任务执行时间

        print(f"[Worker] Rank {rank} completed Task ID {task_id.item()} with result {result}.")

# 主入口
def main():
    world_size = 4  # 总进程数，包括 1 个主进程和 3 个工作进程
    tasks = [i for i in range(10)]  # 示例任务列表

    # 使用多进程模拟分布式机器，若处于真实环境则通过给定不同的rank区分主从机器
    mp.spawn(
        fn=lambda rank: master(rank, world_size, tasks) if rank == 0 else worker(rank, world_size),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    main()