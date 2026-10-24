import hashlib,tempfile,unittest
from pathlib import Path
from download_model import restore_tensor

class Restore(unittest.TestCase):
    def fixture(self,root):
        content=b'abcde';pieces=[b'abc',b'de'];offset=0;parts=[]
        for index,data in enumerate(pieces):
            part={'asset_name':'part'+str(index),'offset':offset,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};parts.append(part);offset+=len(data)
        record={'path':'model.safetensors','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'parts':parts}
        calls=[]
        def downloader(base,part,cache):
            calls.append(part['asset_name']);cache.mkdir(exist_ok=True);path=cache/part['asset_name'];path.write_bytes(pieces[int(part['asset_name'][-1])]);return path
        return record,downloader,calls
    def test_reassemble_and_skip_verified_complete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);record,downloader,calls=self.fixture(root)
            restore_tensor(record,root,'',root/'cache',downloader);restore_tensor(record,root,'',root/'cache',downloader)
            self.assertEqual((root/'model.safetensors').read_bytes(),b'abcde');self.assertEqual(calls,['part0','part1'])
    def test_partial_restore_reuses_verified_part(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);record,downloader,calls=self.fixture(root);(root/'model.safetensors.incomplete').write_bytes(b'abcx')
            restore_tensor(record,root,'',root/'cache',downloader)
            self.assertEqual(calls,['part1']);self.assertEqual((root/'model.safetensors').read_bytes(),b'abcde')
    def test_different_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);record,downloader,calls=self.fixture(root);(root/'model.safetensors').write_bytes(b'other')
            with self.assertRaises(ValueError):restore_tensor(record,root,'',root/'cache',downloader)
            self.assertEqual((root/'model.safetensors').read_bytes(),b'other')

if __name__=='__main__':unittest.main()
