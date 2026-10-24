"""Regression checks for persistence/resource guards and portable result export."""
import io,json,os,subprocess,sys,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from test_preparation import Fixture
import common,runtime_resources,export_results
from bounded_log import drain_to_file,start_drain

class Persistence(Fixture):
    def test_atomic_rename_is_directory_synced(self):
        path=self.root/'nested/result.json'
        with patch('common.fsync_directory',wraps=common.fsync_directory) as sync:
            common.atomic_json(path,{'valid':1});self.assertGreaterEqual(sync.call_count,1)
        self.assertEqual(json.loads(path.read_text()),{'valid':1})
    def test_failed_replace_preserves_previous_summary(self):
        path=self.root/'result.json';common.atomic_json(path,{'old':True});before=path.read_bytes()
        with patch('common.os.replace',side_effect=OSError('simulated storage error')):
            with self.assertRaises(OSError):common.atomic_json(path,{'new':True})
        self.assertEqual(path.read_bytes(),before)
    def test_final_rebuild_recovers_a_committed_round_missing_from_stale_summary(self):
        cfg=self.cfg();directory=common.run_path(cfg)/'tasks/lift_pot'
        common.summarize(cfg,'lift_pot')
        common.append_event(directory/'episodes.jsonl',self.event(0,'policy_completed',success=False))
        self.assertEqual(common.overall(cfg)['completed_valid_policy_episodes'],0)
        self.assertEqual(common.overall(cfg,rebuild=True)['completed_valid_policy_episodes'],1)
    def test_capped_subprocess_log_continues_draining(self):
        child=subprocess.Popen([sys.executable,'-c',"import sys;sys.stdout.buffer.write(b'x'*1048576);sys.stdout.flush()"],stdout=subprocess.PIPE)
        errors=[];path=self.root/'capped.log';thread=start_drain(child.stdout,path,512,errors.append)
        try:self.assertEqual(child.wait(timeout=5),0);thread.join(timeout=5)
        finally:
            if child.poll() is None:child.kill();child.wait()
        self.assertFalse(errors);self.assertEqual(path.stat().st_size,512);self.assertFalse(thread.is_alive())
    def test_log_write_error_is_reported(self):
        errors=[];stream=io.BytesIO(b'pending bytes')
        drain_to_file(stream,self.root,32,errors.append)
        self.assertEqual(len(errors),1);self.assertTrue(stream.closed)

class Resources(Fixture):
    def snapshot(self):
        return {'unified_mem_available_bytes':100*2**30,'disk_free_bytes':100*2**30,'free_inodes':100000,'effective_cpus':20,'load_average':[0,0,0]}
    def check(self,snapshot,cache=100):
        with patch('runtime_resources.available_memory',return_value=100*2**30),patch('runtime_resources.host_snapshot',return_value=snapshot),patch('runtime_resources.cache_bytes',return_value=cache):
            guard=runtime_resources.ResourceGuard(self.root,baseline_cache_bytes=100);return guard.check()
    def test_low_memory_stops_before_more_work(self):
        snapshot=self.snapshot();snapshot['unified_mem_available_bytes']=15*2**30
        with self.assertRaises(MemoryError):self.check(snapshot)
    def test_low_disk_and_inodes_stop(self):
        for key,value in [('disk_free_bytes',7*2**30),('free_inodes',9999)]:
            snapshot=self.snapshot();snapshot[key]=value
            with self.assertRaises(OSError):self.check(snapshot)
    def test_cache_growth_limit_covers_the_shared_baseline(self):
        with self.assertRaises(OSError):self.check(self.snapshot(),5*2**30)
    def test_cpu_and_cache_environment_is_bounded(self):
        service=runtime_resources.thread_env(self.root,'inference');worker=runtime_resources.thread_env(self.root,'sim')
        self.assertEqual(int(service['OMP_NUM_THREADS'])+2*int(worker['OMP_NUM_THREADS']),16)
        self.assertEqual(worker['OPENBLAS_NUM_THREADS'],'1')
        self.assertTrue(Path(service['TMPDIR']).is_relative_to(self.root))
        self.assertEqual(int(service['CUDA_CACHE_MAXSIZE']),512*1024**2)
    def test_suite_baseline_is_not_reset_between_settings(self):
        suite=self.root/'suite'
        with patch('runtime_resources.cache_bytes',return_value=123):self.assertEqual(runtime_resources.suite_cache_baseline(self.root,suite),123)
        with patch('runtime_resources.cache_bytes',return_value=999):self.assertEqual(runtime_resources.suite_cache_baseline(self.root,suite),123)
    def test_scoped_cpu_quota_is_observed(self):
        group=self.root/'group';group.mkdir();(group/'cpu.max').write_text('200000 100000')
        with patch('runtime_resources.cgroup_paths',return_value=[group]):self.assertLessEqual(runtime_resources.effective_cpus(),2)

class ResultExport(Fixture):
    def prepare(self):
        cfgs=[]
        for name,setting in [('clean','demo_clean'),('randomized','demo_randomized')]:
            cfg=self.cfg(setting);cfg.update(setting_name=name,suite_relative='suite',run_relative='suite/'+name,
                                            model_source_relative='suite/preparation/model-source.json')
            slots=[]
            for index in range(10):
                slots.append({'task':'lift_pot','setting':setting,'instruction_type':'seen','episode_index':index,
                              'seed':100000+index,'policy_seed':1000000+index,'instruction_seed':2000000+index,
                              'baseline_instruction':None})
            common.atomic_json(common.run_path(cfg)/'episodes-plan.json',{'episodes':slots})
            cfgs.append(cfg)
        common.atomic_json(self.root/'suite/preparation/model-source.json',{'synthetic_test':True})
        path=common.run_path(cfgs[0])/'tasks/lift_pot/episodes.jsonl'
        for index,success in [(0,True),(1,False)]:
            event=self.event(index,'policy_completed',success=success,instruction='official test instruction',policy_seed=1000000+index,
                             instruction_seed=2000000+index,execution_steps=50,elapsed_seconds=1.0,exception=None)
            common.append_event(path,event)
        return cfgs
    def test_export_keeps_actual_denominator_and_unrun_slots(self):
        data=export_results.build_results(self.prepare())
        self.assertEqual(data['valid_rounds'],2);self.assertEqual(data['successes'],1);self.assertEqual(data['success_rate_on_valid'],0.5)
        self.assertEqual(data['planned_rounds'],20);self.assertFalse(data['standard_100_episode_score'])
        self.assertEqual(data['tasks_and_settings'][1]['rounds'][0]['status'],'not_started')
    def test_portable_archive_checksums_roundtrip_and_corruption_detection(self):
        cfgs=self.prepare()
        with patch('export_results.assert_no_runtime'),patch('export_results.validate_inputs'),patch('model_source.verify_source',return_value='synthetic-source'):
            result=export_results.export(cfgs);directory=Path(result['exported'])
            manifest=export_results.verify_archive(directory)
            self.assertFalse(json.loads((directory/'submission_readiness.json').read_text())['official_submission_ready'])
            self.assertFalse(any(name.endswith('.safetensors') for name in manifest['files']))
            again=export_results.export(cfgs);self.assertTrue(again['reused_identical_export'])
            (directory/'quick_results.json').write_text('{}')
            with self.assertRaises(ValueError):export_results.verify_archive(directory)

if __name__=='__main__':unittest.main(verbosity=2)
