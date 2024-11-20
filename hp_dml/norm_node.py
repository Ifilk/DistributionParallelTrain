import torch
from torch.distributed import init_process_group, recv, isend


# 初始化通信环境
def init_environment(backend="nccl", rank=None, world_size=None):
    init_process_group(backend=backend, rank=rank, world_size=world_size)


# 伪代码中定义的全局变量
is_trained = False
is_updated = False
step = 0


# 主要逻辑函数
def node_main(rank, world_size):
    global is_trained, is_updated, step
    while True:
        # 接收向量 e
        e = torch.empty(5)
        recv(tensor=e)
        if e[0].item() == 1:
            # 接收向量 O
            O = torch.empty(world_size + 1)  # 假设 n=world_size
            recv(tensor=O)

            # 如果 O 全是 -1，退出程序
            if torch.all(O == -1):
                break

            # 如果 O[m] 是 -1，跳转到起始逻辑
            if O[rank].item() == -1:
                continue

            # 如果已更新
            if is_updated:
                O[rank] = -1
                continue

            # 如果 O[m] 是 0，进行相应逻辑
            if O[rank].item() == 0:
                gpu_count = min(torch.cuda.device_count(), 8)
                O[rank] = gpu_count

                valid_indices = (0 < O[:world_size]) & (O[:world_size] <= 8)
                valid_count = torch.sum(valid_indices).item()

                if valid_count == O[-1].item():
                    # 节点 m 变为 root
                    neighbors = torch.nonzero(valid_indices).flatten().tolist()
                    O[:world_size] += 9
                elif valid_count == world_size - 1:
                    # 定义 m 为开始节点
                    pass
                else:
                    # 随机邻居转发
                    neighbor = torch.randint(0, world_size, (1,)).item()
                    isend(tensor=O, dst=neighbor)

            # 如果尚未训练
            if not is_trained:
                model_update(rank)

        elif e[0].item() == 2:
            # 接收模型参数
            param_shape = e[2:4].long().tolist()
            M_t = torch.empty(param_shape)
            recv(tensor=M_t, src=torch.distributed.WORLD)

            if torch.equal(M_t[rank], M_t):
                is_updated = True
            else:
                step = handle_step(M_t, step)


def model_update(rank):
    """模拟训练逻辑"""
    print(f"Node {rank}: Performing model update...")
    # 示例代码，用实际模型替换
    pass


def handle_step(M_t, step):
    """处理不同的 step 阶段"""
    if step == 0:
        # 聚合模型
        print(f"Step {step}: Aggregating model...")
        step += 1
    elif step == 1:
        # 接受并更新模型
        print(f"Step {step}: Updating model...")
        step += 1
    elif step == 2:
        # 重置 step
        print(f"Step {step}: Resetting step...")
        step = 0
    return step


# 主程序入口
if __name__ == "__main__":
    rank = int(input("Enter rank: "))  # 节点编号
    world_size = int(input("Enter world size: "))  # 总节点数

    init_environment(rank=rank, world_size=world_size)
    try:
        node_main(rank, world_size)
    finally:
        torch.distributed.destroy_process_group()
