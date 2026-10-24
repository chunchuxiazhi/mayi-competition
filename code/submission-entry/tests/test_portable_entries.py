"""CPU-only deployment/call-graph tests. No real subprocess, model or simulator."""
import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import portable_eval as evaluation
import portable_train as training
import model_source
from common import fingerprint,seeds
from validation import validate_inputs

class Entries(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'replay'
    def args(self,*extra):return ['--workdir',str(self.root),*extra]
    def test_two_settings_same_weights_distinct_outputs_and_pools(self):
        configs=evaluation.build_configs(evaluation.parser().parse_args(self.args()))
        self.assertEqual([(c['task_config'],c['instruction_type']) for c in configs],[('demo_clean','seen'),('demo_randomized','unseen')])
        self.assertEqual(configs[0]['model_relative'],configs[1]['model_relative'])
        self.assertNotEqual(configs[0]['run_relative'],configs[1]['run_relative'])
        self.assertTrue(all(len(set(c['tasks']))==50 and c['test_num']==10 for c in configs))
        self.assertTrue(all('training_relative' not in c and c['source_mode']=='submitted_hf' for c in configs))
    def test_default_never_spawns_or_writes_output(self):
        with patch.object(evaluation.subprocess,'Popen',side_effect=AssertionError('Must not spawn')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluation.main(self.args()),0)
        self.assertFalse(self.root.exists())
        self.assertNotIn('torch',sys.modules);self.assertNotIn('sapien',sys.modules)
    def test_reviewer_calls_both_drivers_sequentially(self):
        calls=[]
        class Child:
            def __init__(self,argv,**kwargs):calls.append(argv)
            def wait(self,**kwargs):return 0
            def poll(self):return 0
        with patch.object(evaluation.sys,'platform','linux'),patch.object(evaluation,'preflight',return_value={'passed':True}),\
             patch.object(evaluation,'prepare_layout',return_value=[Path('clean.json'),Path('randomized.json')]),\
             patch('port_release.wait_for_port_release'),patch.object(evaluation.subprocess,'Popen',Child),\
             patch.object(evaluation,'export_replay_results') as export,contextlib.redirect_stdout(io.StringIO()):
            evaluation.main(self.args('--execute','--vla-python',str(Path(self.temp.name)/'vla/bin/python')))
        self.assertEqual(len(calls),2)
        self.assertIn('clean',calls[0][calls[0].index('--config')+1]);self.assertIn('randomized',calls[1][calls[1].index('--config')+1])
        export.assert_called_once()
    def test_global_driver_failure_stops_next_setting(self):
        calls=[]
        class Child:
            def __init__(self,argv,**kwargs):calls.append(argv)
            def wait(self,**kwargs):return 1
            def poll(self):return 1
        with patch.object(evaluation.sys,'platform','linux'),patch.object(evaluation,'preflight',return_value={}),\
             patch.object(evaluation,'prepare_layout',return_value=[Path('a'),Path('b')]),\
             patch('port_release.wait_for_port_release'),patch.object(evaluation.subprocess,'Popen',Child),\
             patch.object(evaluation,'export_replay_results') as export,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):evaluation.main(self.args('--execute','--vla-python',str(Path(self.temp.name)/'vla/bin/python')))
        self.assertEqual(len(calls),1);export.assert_not_called()
    def test_unseen_only_single_worker_and_actual_count_configurable(self):
        cfg=evaluation.build_configs(evaluation.parser().parse_args(self.args('--setting','randomized','--workers','1','--test-num','100')))[0]
        self.assertEqual((cfg['instruction_type'],cfg['workers'],cfg['test_num']),('unseen',1,100))
    def test_episode_plan_rejects_seed_drift(self):
        cfg=evaluation.build_configs(evaluation.parser().parse_args(self.args('--setting','clean')))[0]
        folder=self.root/cfg['run_relative'];folder.mkdir(parents=True)
        plan={'configuration_sha256':fingerprint(cfg),'episodes':[{'task':t,'episode_index':i,'setting':cfg['task_config'],
            'instruction_type':cfg['instruction_type'],**seeds(cfg,j,i)} for j,t in enumerate(cfg['tasks']) for i in range(10)]}
        path=folder/'episodes-plan.json';path.write_text(json.dumps(plan));self.assertEqual(validate_inputs(cfg)['planned_slots'],500)
        plan['episodes'][1]['seed']+=1;path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):validate_inputs(cfg)
    def test_submission_HF_hash_verification_without_DCP_or_mtime(self):
        import hashlib
        model=Path(self.temp.name)/'hf';model.mkdir();shard=model/'part.safetensors';shard.write_bytes(b'cpu-fixture')
        (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':shard.name}}))
        manifest=Path(self.temp.name)/'manifest.json';manifest.write_text(json.dumps({'step':20000,'files':[
            {'path':shard.name,'bytes':11,'sha256':hashlib.sha256(shard.read_bytes()).hexdigest()}]}))
        cfg={'model_relative':str(model),'model_manifest':str(manifest),'training_step':20000}
        self.assertEqual(len(model_source.verify_source(cfg,full_hash=True)),64)
        shard.write_bytes(b'bad-fixture')
        with self.assertRaises(ValueError):model_source.verify_source(cfg,full_hash=True)
    def test_training_recipe_only_relocates_paths_and_explicit_second_stage_fields(self):
        first=training.make_config(self.root,'10000');second=training.make_config(self.root,'20000')
        expected=json.loads(json.dumps(first));expected['train'].update(output_dir=str(self.root/'runs/finetune-clean-v3-gb10/clean50-20000-from10000-20261018'),
            max_steps=20000,lr='1e-05',lr_min='5e-06',lr_warmup_ratio=0.0)
        self.assertEqual(second,expected);self.assertNotIn(training.OLD,json.dumps(first))
        self.assertEqual((first['train']['save_steps'],first['data']['num_workers']),(1000,0))

if __name__=='__main__':unittest.main()
