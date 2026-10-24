"""Project-local Torch 2.9.1 DCP writer: avoid TORCH_SAVE CPU-copy retention.

No torch installation files or filesystem module globals are changed. Source
hashes fail closed on an unreviewed upstream implementation.
"""
import ast
import gc
import hashlib
import inspect
import textwrap

QUEUE_SHA256 = '1f5f8015828f8606bc9d7dd1d8e2477bda4b1f1c8e913c3265b35b2a46d18287'
DISPATCH_SHA256 = 'a16804371a37b032c453825ea56f26ac0e150bbb36fbc70b6f49ee4ab1afc178'


def streaming_writer_class(event=None):
    import torch
    import torch.distributed.checkpoint.filesystem as fs
    if torch.__version__.split('+')[0] != '2.9.1':
        raise RuntimeError('Streaming DCP adapter requires reviewed Torch 2.9.1')
    source = inspect.getsource(fs._write_files_from_queue)
    dispatch = inspect.getsource(fs.FileSystemWriter._write_data)
    if hashlib.sha256(source.encode()).hexdigest() != QUEUE_SHA256:
        raise RuntimeError('DCP queue source changed; review before saving')
    if hashlib.sha256(dispatch.encode()).hexdigest() != DISPATCH_SHA256:
        raise RuntimeError('DCP dispatcher source changed; review before saving')
    tree = ast.parse(source)
    changes = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        indexes = [i for i, n in enumerate(node.body)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript)
                   and isinstance(n.targets[0].value, ast.Name)
                   and n.targets[0].value.id in ('tensor_dict', 'metadata_dict')]
        if indexes:
            if len(indexes) != 2 or indexes[1] != indexes[0] + 1:
                raise RuntimeError('Unexpected DCP tensor collection structure')
            start = indexes[0]
            condition = ast.parse('serialization_format == SerializationFormat.SAFETENSORS', mode='eval').body
            node.body[start:start + 2] = [ast.If(test=condition, body=node.body[start:start + 2], orelse=[])]
            changes += 1
    if changes != 1:
        raise RuntimeError('Expected exactly one tensor collection loop')
    ast.fix_missing_locations(tree)
    namespace = fs.__dict__.copy()
    exec(compile(tree, '<project-streaming-dcp-queue>', 'exec'), namespace)
    exec(compile(ast.parse(textwrap.dedent(dispatch)), '<project-streaming-dcp-dispatch>', 'exec'), namespace)
    local_dispatch = namespace['_write_data']

    class StreamingFileSystemWriter(fs.FileSystemWriter):
        def __init__(self, *args, **kwargs):
            kwargs.update(thread_count=1, sync_files=True, per_thread_copy_ahead=0)
            super().__init__(*args, **kwargs)
            if self.serialization_format != fs.SerializationFormat.TORCH_SAVE:
                raise ValueError('Only reviewed TORCH_SAVE is enabled for this project')

        def _write_data(self, planner, file_queue):
            if event:
                event('dcp_tensor_write_begin', str(self.path))
            result = local_dispatch(self, planner, file_queue)
            # local_dispatch has returned: loader, staging tensors and its frame
            # are now out of scope. This does not promise allocator RSS release.
            gc.collect()
            if event:
                event('dcp_tensor_write_end', str(self.path))
            return result

    return StreamingFileSystemWriter
