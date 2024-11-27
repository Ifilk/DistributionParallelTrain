```python
import hp_dml.dist as dist

manager = dist.ProcessManager(
    state=3,
    max_worker=8,
    queue_size=8
)

@manager.handler(
    state=0,
    mutex=True
)
def handler_for_state0(meta_message: dist.MetaMessage):
    result = dist.last_recv()
    

dist.recv()

manager.handle(dist.get_meta_state())
```