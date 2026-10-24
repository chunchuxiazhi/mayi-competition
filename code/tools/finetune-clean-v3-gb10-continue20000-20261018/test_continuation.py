import copy,json,tempfile,unittest
from pathlib import Path
from continuation import ContinuationSchedule,learning_rate,prune_chain,select_checkpoint,signature
from storage_policy import commit

class FakeOptimizer:
    def __init__(self,lr=1e-5):self.param_groups=[{'lr':lr},{'lr':lr}]

class ScheduleTests(unittest.TestCase):
    def migrated(self):
        opt=FakeOptimizer();scheduler=ContinuationSchedule(opt)
        scheduler.load_state_dict({'last_epoch':10000,'_last_lr':[1e-5,1e-5],'base_lrs':[2e-5,2e-5]})
        return opt,scheduler
    def test_first_stage_migration_is_continuous(self):
        opt,s=self.migrated();self.assertEqual(s.last_epoch,10000)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-5)
        s.step();self.assertEqual(s.last_epoch,10001)
        self.assertLess(s.get_last_lr()[0],1e-5)
    def test_all_10000_updates_decay_to_exact_endpoint(self):
        _,s=self.migrated();rates=[]
        for _ in range(10000):s.step();rates.append(s.get_last_lr()[0])
        self.assertEqual(s.last_epoch,20000);self.assertAlmostEqual(rates[-1],5e-6)
        self.assertTrue(all(a>=b for a,b in zip(rates,rates[1:])))
        with self.assertRaises(RuntimeError):s.step()
    def test_checkpoint_resume_has_same_next_lr_as_uninterrupted(self):
        _,s=self.migrated()
        for _ in range(1000):s.step()
        state=copy.deepcopy(s.state_dict());opt=FakeOptimizer();restored=ContinuationSchedule(opt)
        for g in opt.param_groups:g['lr']=learning_rate(11000)
        restored.load_state_dict(state);s.step();restored.step()
        self.assertEqual(s.state_dict(),restored.state_dict())
    def test_reject_wrong_initial_step_or_optimizer_rate(self):
        for step,rate in ((9000,1e-5),(10000,2e-5)):
            opt=FakeOptimizer();s=ContinuationSchedule(opt)
            for g in opt.param_groups:g['lr']=rate
            with self.assertRaises(RuntimeError):s.load_state_dict({'last_epoch':step,'_last_lr':[1e-5]*2,'base_lrs':[2e-5]*2})
    def test_reject_unknown_or_changed_recipe(self):
        _,s=self.migrated();saved=s.state_dict();saved['lr_end']=0
        with self.assertRaises(RuntimeError):s.load_state_dict(saved)
        with self.assertRaises(RuntimeError):ContinuationSchedule(FakeOptimizer()).step()

class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp_parent=Path(__file__).resolve().parent/'test-tmp';self.temp_parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=self.temp_parent);self.root=Path(self.tmp.name).resolve()
        if not self.root.is_relative_to(self.temp_parent.resolve()):raise RuntimeError('Test fixture escaped its directory')
        self.old=self.root/'old';self.new=self.root/'new'
        self.config={'train':{'output_dir':str(self.new)}}
        self.control={'keep_checkpoints':2,'continuation':{'source_run':str(self.old),'source_signature':'old'}}
    def tearDown(self):
        if not self.root.is_relative_to(self.temp_parent.resolve()):raise RuntimeError('Unsafe test cleanup')
        self.tmp.cleanup()
    def make(self,run,step,complete=True):
        path=run/'checkpoints'/f'global_step_{step}'
        for name in ('model/.metadata','model/__0_0.distcp','optimizer/.metadata','optimizer/__0_0.distcp','extra_state/extra_state_rank_0.pt'):
            p=path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic-small-test-state')
        if complete:commit(path.parent,path,'old' if run==self.old else signature(self.config))
        return path
    def test_retains_latest_two_across_source_and_continuation(self):
        a=self.make(self.old,9000);b=self.make(self.old,10000);c=self.make(self.new,11000)
        self.assertEqual(prune_chain(self.config,self.control),[str(a)])
        self.assertTrue(b.exists() and c.exists())
        self.make(self.new,12000)
        self.assertEqual(prune_chain(self.config,self.control),[str(b)])
        self.assertTrue(c.exists())
    def test_incomplete_or_corrupt_save_never_deletes_recovery_points(self):
        a=self.make(self.old,9000);b=self.make(self.old,10000);c=self.make(self.new,11000,False)
        with self.assertRaises((RuntimeError,FileNotFoundError)):prune_chain(self.config,self.control)
        self.assertTrue(a.exists() and b.exists() and c.exists())
        with self.assertRaises((RuntimeError,FileNotFoundError)):select_checkpoint(self.config,self.control)
    def test_unexpected_source_checkpoint_prevents_deletion(self):
        a=self.make(self.old,8000);self.make(self.new,11000)
        with self.assertRaises(RuntimeError):prune_chain(self.config,self.control)
        self.assertTrue(a.exists())

if __name__=='__main__':unittest.main(verbosity=2)
