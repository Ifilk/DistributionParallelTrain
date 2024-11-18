### PyTorch分布式通信 `torch.distributed`

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 定义模型
model = ...

# 检查分布式api是否可用
assert dist.is_available()
# 检查nccl后端是否可用
assert dist.is_nccl_available()
# 初始化分布式环境
# 初始化方法 如env://，tcp://或file://
dist.init_process_group(backend='nccl', init_method='env://')

# 检查是否初始化
assert dist.is_initialized()
# 将模型封装在DistributedDataParallel中
ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])


from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# 定义数据集和分布式采样器
dataset = ...
sampler = DistributedSampler(dataset)

# 使用DataLoader加载数据
data_loader = DataLoader(dataset, sampler=sampler)

tensor = ...
# 发送张量数据
dist.send(tensor, dst=1)

# 接收张量数据
dist.recv(tensor, src=1)
```