# TorchTitan API

## JobConfig

### Arguments

| 名称                                            | 类型                  | 默认值                                                 | 描述                                                     |
|-----------------------------------------------|---------------------|-----------------------------------------------------|--------------------------------------------------------|
| `--job.config_file`                           | `str`               | `None`                                              | Job config file                                        |
| `--job.dump_folder`                           | `str`               | `"./torchtitan/outputs"`                            | Folder to dump job outputs                             |
| `--job.description`                           | `str`               | `"default job"`                                     | Description of the job                                 |
| `--job.use_for_integration_test`              | `store_true`        | `False`                                             | Add this config to the integration test suite          |
| `--profiling.enable_profiling`                | `store_true`        | `False`                                             | Whether to enable pytorch profiler                     |
| `--profiling.save_traces_folder`              | `str`               | `"profile_traces"`                                  | Trace files location                                   |
| `--profiling.profile_freq`                    | `int`               | `10`                                                | How often to collect profiler traces, in iterations    |
| `--profiling.enable_memory_snapshot`          | `store_true`        | `False`                                             | Whether to dump memory snapshot                        |
| `--profiling.save_memory_snapshot_folder`     | `str`               | `"memory_snapshot"`                                 | Memory snapshot files location                         |
| `--metrics.log_freq`                          | `int`               | `10`                                                | How often to log metrics to TensorBoard, in iterations |
| `--metrics.enable_color_printing`             | `store_true`        | `False`                                             | Whether to enable color printing                       |
| `--metrics.enable_tensorboard`                | `store_true`        | `False`                                             | Whether to log metrics to TensorBoard                  |
| `--metrics.save_tb_folder`                    | `str`               | `"tb"`                                              | Folder to dump TensorBoard states                      |
| `--metrics.rank_0_only`                       | `store_true`        | `True`                                              | Whether to save TensorBoard metrics only for rank 0    |
| `--model.name`                                | `str`               | `"llama"`                                           | Which model to train                                   |
| `--model.flavor`                              | `str`               | `"debugmodel"`                                      | Which model config to train                            |
| `--model.norm_type`                           | `str`               | `"rmsnorm"`                                         | Type of layer normalization to use                     |
| `--model.tokenizer_path`                      | `str`               | `"./torchtitan/datasets/tokenizer/tokenizer.model"` | Tokenizer path                                         |
| `--optimizer.name`                            | `str`               | `"AdamW"`                                           | Optimizer to use                                       |
| `--optimizer.lr`                              | `float`             | `8e-4`                                              | Learning rate to use                                   |
| `--optimizer.fused`                           | `store_true`        | `False`                                             | Whether the fused implementation (CUDA only) is used   |
| `--training.dataset`                          | `str`               | `"c4_mini"`                                         | Dataset to use                                         |
| `--training.dataset_path`                     | `str`               | `None`                                              | Path to the dataset in the file system                 |
| `--training.batch_size`                       | `int`               | `8`                                                 | Batch size                                             |
| `--training.seq_len`                          | `int`               | `2048`                                              | Sequence length                                        |
| `--training.warmup_steps`                     | `int`               | `200`                                               | Steps for lr scheduler warmup                          |
| `--training.max_norm`                         | `Union[float, int]` | `1.0`                                               | Max norm for gradient clipping                         |
| `--training.steps`                            | `int`               | `10000`                                             | How many train steps to run                            |
| `--training.data_parallel_replicate_degree`   | `int`               | `1`                                                 | Data parallelism degree for weight replication         |
| `--training.data_parallel_shard_degree`       | `int`               | `-1`                                                | Data parallelism degree for weight sharding            |
| `--training.enable_cpu_offload`               | `bool`              | `False`                                             | Whether to apply CPU offloading                        |
| `--training.tensor_parallel_degree`           | `int`               | `1`                                                 | Tensor Parallelism degree                              |
| `--training.enable_loss_parallel`             | `store_true`        | `True`                                              | Whether to apply loss parallel                         |
| `--experimental.enable_async_tensor_parallel` | `store_true`        | `False`                                             | Whether to apply async tensor parallel                 |
| `--checkpoint.enable_checkpoint`              | `store_true`        | `False`                                             | Whether to enable checkpoint                           |
| `--checkpoint.folder`                         | `str`               | `"checkpoint"`                                      | The folder to store checkpoints                        |
| `--checkpoint.interval_type`                  | `str`               | `"steps"`                                           | Checkpointing interval unit of measurement             |
| `--checkpoint.interval`                       | `int`               | `500`                                               | Checkpointing interval, in steps or seconds            |
| `--checkpoint.model_weights_only`             | `store_true`        | `False`                                             | Save only model weights at the end of training         |
| `--memory_estimation.enabled`                 | `store_true`        | `False`                                             | Whether to estimate memory usage for FSDP.             |
| `--memory_estimation.disable_fake_mode`       | `store_true`        | `False`                                             | Whether to estimate memory under FakeTensorMode.       |

### Configuration File

```toml
# torchtitan Config.toml

[job]
dump_folder = "./outputs"
description = "Llama 3 debug training"
use_for_integration_test = true

[profiling]
enable_profiling = true
save_traces_folder = "profile_trace"
profile_freq = 10
enable_memory_snapshot = false
save_memory_snapshot_folder = "memory_snapshot"

[metrics]
log_freq = 1
enable_color_printing = true
enable_tensorboard = true
save_tb_folder = "tb"

[model]
name = "llama3"
flavor = "debugmodel"
norm_type = "rmsnorm"  # layernorm / np_layernorm / rmsnorm / fused_rmsnorm
# test tokenizer.model, for debug purpose only
tokenizer_path = "./test/assets/test_tiktoken.model"

[optimizer]
name = "AdamW"
lr = 8e-4

[training]
batch_size = 8
seq_len = 2048
warmup_steps = 2  # lr scheduler warm up, normally 20% of the train steps
max_norm = 1.0  # grad norm clipping
steps = 10
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
tensor_parallel_degree = 1
compile = false
dataset = "c4_test"  # supported datasets: c4_test (2K), c4 (177M)

[experimental]
context_parallel_degree = 1
pipeline_parallel_degree = 1
enable_async_tensor_parallel = false

[checkpoint]
enable_checkpoint = false
folder = "checkpoint"
interval_type = "steps"
interval = 5
model_weights_only = false
export_dtype = "float32"
async_mode = "disabled"  # ["disabled", "async", "async_with_pinned_mem"]

[activation_checkpoint]
mode = 'selective'  # ['none', 'selective', 'full']
selective_ac_option = '2'  # 'int' = ac every positive int layer or 'op', ac based on ops policy

[float8]
enable_float8_linear = false

```