# 分布式任务调度框架 Job 使用说明

## 概述

`Job` 是一个轻量级的分布式任务调度框架，用于管理和执行多进程中的复杂任务流水线。通过 `Job`，可以方便地在主进程和工作进程中定义各自的任务，支持数据的分发与聚合，以及多进程之间的通信。

---

## 特性

- **灵活的任务管理**：支持主进程、工作进程和公共任务的定义与调度。
- **易于扩展的任务流水线**：通过装饰器的方式轻松注册不同的任务。
- **支持 Tensor 通信**：内置了基于 `torch.distributed` 的通信功能。
- **统一的任务构建接口**：将任务分步骤组织并构建为可执行的流水线。

---

## 环境依赖

在使用 `Job` 之前，请确保满足以下环境依赖：
- Python 3.7+
- PyTorch (建议版本：1.10+)
- 配置完成的分布式环境 (`torch.distributed` 支持的 backend)

---

## 快速开始

以下是一个简单的示例，展示如何使用 `Job` 定义和执行任务流水线：

### 1. 初始化 Job
```python
from dpt.job_manager import Job

world_size = 4  # 假设有 4 个进程参与
job = Job(world_size)
```

### 2. 定义任务流水线

通过装饰器定义主进程、工作进程或公共任务：

```python
@job.public_progress()
def initialize():
    print("初始化任务")

@job.worker_progress()
def worker_compute():
    local_result = 42  # 模拟计算
    return local_result

@job.master_progress([world_size - 1])
def aggregate_results(worker_results):
    print("接收到的工作进程结果:", worker_results)
    final_result = sum(worker_results)
    return final_result
```

### 3. 构建并运行流水线
完成任务定义后，调用 `build` 方法生成可执行的流水线函数，并在分布式环境中运行：

```python
if __name__ == "__main__":
    dist.init_process_group("gloo", rank=0, world_size=world_size)
    
    pipeline = job.build()
    pipeline()  # 执行任务流水线
```

---

## 装饰器说明

### 1. `@job.master_progress`
- **功能**：用于注册仅由主进程执行的任务。
- **参数**：
  - `recv_shape_list`：指定主进程从工作进程接收的数据形状列表。

### 2. `@job.worker_progress`
- **功能**：用于注册仅由工作进程执行的任务。
- **参数**：
  - `recv_shape_list`：指定工作进程从主进程接收的数据形状列表。

### 3. `@job.public_progress`
- **功能**：用于注册所有进程都会执行的任务。

---

## 进阶用法

### 1. 数据广播
主进程的返回结果会自动广播到工作进程，无需额外编码：

```python
@job.master_progress()
def broadcast_data():
    return [1, 2, 3, 4]  # 广播到所有工作进程
```

### 2. 数据聚合
工作进程的计算结果会自动收集到主进程：

```python
@job.worker_progress()
def worker_calculate():
    return torch.tensor([rank])  # 每个工作进程返回自己的 rank 值
```

---

## 注意事项

1. 确保在多进程环境下运行，正确配置 `torch.distributed` 的 backend 和进程组。
2. 所有 `recv_shape_list` 参数需要与实际传输的张量形状一致。
3. 调用 `pipeline()` 前，请确保已初始化分布式环境 (`dist.init_process_group`)。

---

## 联系方式

如有问题，请通过以下方式联系我们：
- 邮箱：support@example.com
- 文档主页：[Job Framework Docs](https://example.com)

---

通过 `Job` 框架，您可以更高效地构建和管理分布式任务流水线，为复杂的分布式计算提供可靠的支持！
