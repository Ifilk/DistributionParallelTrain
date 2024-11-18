import time
import threading



class DistributedFramework:
    """模拟 Ray.init 的初始化框架"""
    def __init__(self):
        self.tasks = []  # 存储任务线程
        # TODO 初始化

    def remote(self, cls):
        """
        模拟 Ray 的 @ray.remote 装饰器
        返回一个分布式对象代理
        """
        return DistributedProxy(cls)


class DistributedProxy:
    """代理类，用于调用分布式对象的远程方法"""
    def __init__(self, cls):
        self.instance = cls()

    def train(self, *args, **kwargs):
        task_id = f"task_{time.time_ns()}"  # 任务 ID
        thread = threading.Thread(target=self._execute, args=(task_id, *args), kwargs=kwargs)
        thread.start()
        return task_id

    def _execute(self, task_id, *args, **kwargs):
        result = self.instance.train(*args, **kwargs)
        # TODO 标记任务完成
        # TODO 存储结果


class Client:
    """模拟远程任务类"""
    def train(self):
        time.sleep(2)  # 模拟训练
        return f"Model_{time.time_ns()}", f"Gradient_{time.time_ns()}", f"Loss_{time.time_ns()}"


def ray_wait(task_ids, num_returns=1, timeout=None):
    """模拟 ray.wait"""
    start_time = time.time()
    ready = set()

    while len(ready) < num_returns:
        # TODO 查询已完成的任务，更新ready

        if timeout and (time.time() - start_time > timeout):
            break
        time.sleep(0.1)  # 减少查询频率

    not_ready = task_ids.difference(ready)
    return list(ready), list(not_ready)


def ray_get(task_ids):
    """模拟 ray.get"""
    results = []
    for task_id in task_ids:
        # TODO 获取task的result
        result = ...
        results.append(result)
    return results


# 主程序
def main():
    # 初始化框架
    framework = DistributedFramework()

    # 创建分布式客户端
    clients = [framework.remote(Client) for _ in range(5)]

    # 提交训练任务
    process_ids = [client.train() for client in clients]

    # 等待所有任务完成
    ready, not_ready = ray_wait(set(process_ids), num_returns=len(process_ids), timeout=10)

    # 获取任务结果
    results = ray_get(ready)

    # 打印结果
    for model, gradient, loss in results:
        print(f"Model: {model}, Gradient: {gradient}, Loss: {loss}")


if __name__ == "__main__":
    main()
