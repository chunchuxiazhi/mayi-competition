"""CPU roundtrip + clone-lifetime regression for the exact project writer."""
import gc
from pathlib import Path
import tempfile
import unittest
import weakref


class StreamingWriterTest(unittest.TestCase):
    def test_real_dcp_roundtrip_and_no_collection_of_past_cpu_copies(self):
        import torch
        import torch.distributed.checkpoint as dcp
        import torch.distributed.checkpoint.filesystem as fs
        from dcp_stream_writer import streaming_writer_class
        original = fs._write_files_from_queue
        refs, observations = [], []
        original_write = fs._write_item
        def observe(*args, **kwargs):
            data = args[2]
            if isinstance(data, torch.Tensor):
                refs.append(weakref.ref(data))
                observations.append(sum(r() is not None for r in refs))
            return original_write(*args, **kwargs)
        # Only intercept _write_item while constructing the isolated namespace.
        # Reset the upstream global before invoking writer code.
        fs._write_item = observe
        try:
            writer_class = streaming_writer_class()
        finally:
            fs._write_item = original_write
        tensors = {f'w_{i}': torch.full((2 * 1024 * 1024,), float(i + 1))[:1024 * 1024] for i in range(4)}
        root = Path('/home/zzj/competitions/lingbot/cache/tmp').resolve()
        with tempfile.TemporaryDirectory(dir=root, prefix='stream-writer-cpu-') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(root))
            writer = writer_class(folder)
            self.assertEqual(writer.thread_count, 1)
            self.assertEqual(writer.per_thread_copy_ahead, 0)
            self.assertTrue(writer.sync_files)
            dcp.save(tensors, storage_writer=writer)
            restored = {k: torch.zeros_like(v) for k, v in tensors.items()}
            dcp.load(restored, storage_reader=dcp.FileSystemReader(folder))
            self.assertTrue(all(torch.equal(v, restored[k]) for k, v in tensors.items()))
        gc.collect()
        self.assertEqual(observations, [1, 1, 1, 1])
        self.assertEqual(sum(r() is not None for r in refs), 0)
        self.assertIs(fs._write_files_from_queue, original)


if __name__ == '__main__':
    unittest.main()
