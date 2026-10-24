"""CPU-only accounting/launch-control tests: no Torch, simulator or model imports."""
import ast,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import common,driver,run_suite
from common import HERE,atomic_json,append_event,read_events,summarize,overall,seeds
from model_source import expected_episode,verify_source

class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp_parent=HERE/'test-tmp';self.temp_parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=self.temp_parent);self.root=Path(self.tmp.name).resolve()
        if not self.root.is_relative_to(self.temp_parent.resolve()):raise RuntimeError('Test fixture escaped')
    def tearDown(self):
        if not self.root.is_relative_to(self.temp_parent.resolve()):raise RuntimeError('Unsafe test cleanup')
        self.tmp.cleanup()
    def cfg(self,setting='demo_clean'):
        return {'label':'test','project_root':str(self.root),'run_relative':setting,'task_config':setting,
                'instruction_type':'seen','training_step':20000,'tasks':['lift_pot'],'test_num':10}
    def event(self,index,kind,**kw):
        return {'event':kind,'task':'lift_pot','setting':'demo_clean','instruction_type':'seen',
                'episode_index':index,'attempt_id':f'attempt-{index}','seed':100000+index,**kw}

class Accounting(Fixture):
    def test_actual_denominator_keeps_failure_and_blocked_separate(self):
        cfg=self.cfg();path=common.run_path(cfg)/'tasks/lift_pot/episodes.jsonl'
        for index,success in ((0,True),(1,False)):
            append_event(path,self.event(index,'policy_started'))
            append_event(path,self.event(index,'policy_completed',success=success))
        append_event(path,self.event(2,'episode_blocked',success=None))
        row=summarize(cfg,'lift_pot')
        self.assertEqual(row['completed_valid_policy_episodes'],2);self.assertEqual(row['successes'],1)
        self.assertEqual(row['success_rate'],0.5);self.assertEqual(row['expert_blocked_episode_slots'],1)
        self.assertEqual(row['not_validly_completed_slots'],8);self.assertFalse(row['complete'])
        self.assertFalse(row['standard_100_episode_score']);self.assertEqual(overall(cfg)['success_rate'],0.5)
    def test_duplicate_terminal_is_rejected_instead_of_counted_twice(self):
        cfg=self.cfg();path=common.run_path(cfg)/'tasks/lift_pot/episodes.jsonl'
        for _ in range(2):append_event(path,self.event(0,'policy_completed',success=False))
        with self.assertRaises(RuntimeError):summarize(cfg,'lift_pot')
    def test_cross_setting_ledger_is_rejected(self):
        cfg=self.cfg('demo_randomized');path=common.run_path(cfg)/'tasks/lift_pot/episodes.jsonl'
        append_event(path,self.event(0,'policy_completed',success=False))
        with self.assertRaises(ValueError):summarize(cfg,'lift_pot')
    def test_zero_valid_does_not_become_zero_or_success_score(self):
        row=summarize(self.cfg(),'lift_pot')
        self.assertIsNone(row['success_rate']);self.assertFalse(row['complete'])
    def test_interruption_is_not_a_completed_or_successful_episode(self):
        cfg=self.cfg();path=common.run_path(cfg)/'tasks/lift_pot/episodes.jsonl'
        append_event(path,self.event(0,'policy_started'))
        append_event(path,self.event(0,'policy_interrupted',success=None))
        row=summarize(cfg,'lift_pot');self.assertEqual(row['interrupted_attempts'],1)
        self.assertEqual(row['completed_valid_policy_episodes'],0);self.assertEqual(row['open_policy_attempts'],0)
    def test_ledger_cap_preserves_existing_bytes(self):
        path=self.root/'ledger.jsonl';append_event(path,{'event':'small'})
        before=path.read_bytes()
        with patch('common.LEDGER_LIMIT_BYTES',len(before)+8):
            with self.assertRaises(RuntimeError):append_event(path,{'event':'too-large'})
        self.assertEqual(path.read_bytes(),before)
    def test_damaged_partial_line_is_not_silently_ignored(self):
        path=self.root/'broken.jsonl';path.write_text('{"x":')
        with self.assertRaises(RuntimeError):read_events(path)

class Inputs(Fixture):
    def test_fixed_plan_rejects_setting_seed_and_instruction_drift(self):
        cfg=self.cfg();ss=seeds({'seed_base':100000,'policy_seed_base':1000000,'instruction_seed_base':2000000},0,0)
        plan={'task':'lift_pot','episode_index':0,'setting':'demo_clean','baseline_instruction':'lift the pot',**ss}
        atomic_json(common.run_path(cfg)/'episodes-plan.json',{'episodes':[plan]})
        expected_episode(cfg,'lift_pot',0,ss,'lift the pot')
        with self.assertRaises(ValueError):expected_episode(cfg,'lift_pot',0,{**ss,'seed':1},'lift the pot')
        with self.assertRaises(ValueError):expected_episode(cfg,'lift_pot',0,ss,'wrong')
        cfg['task_config']='demo_randomized'
        with self.assertRaises(ValueError):expected_episode(cfg,'lift_pot',0,ss,'lift the pot')
    def test_wrong_model_step_is_rejected_without_loading_weights(self):
        cfg=self.cfg();cfg.update(training_relative='training',model_relative='training/exports/global_step_20000/hf_ckpt',
                                 model_wrapper_relative='training/exports/global_step_20000/hf_ckpt',model_source_relative='source.json')
        atomic_json(self.root/'source.json',{'training_step':19000,'hf_path':str(self.root/cfg['model_relative'])})
        with self.assertRaises(ValueError):verify_source(cfg)
    def test_both_configs_have_exactly_1000_combined_slots_and_separate_outputs(self):
        cfgs=run_suite.configs();self.assertEqual(sum(len(c['tasks'])*c['test_num'] for c in cfgs),1000)
        self.assertEqual({c['task_config'] for c in cfgs},{'demo_clean','demo_randomized'})
        self.assertNotEqual(cfgs[0]['run_relative'],cfgs[1]['run_relative'])
        self.assertEqual(cfgs[0]['model_relative'],cfgs[1]['model_relative'])
        self.assertTrue(all(c['workers']==2 and c['training_step']==20000 and c['instruction_type']=={'clean':'seen','randomized':'unseen'}[c['setting_name']] for c in cfgs))

class Gates(unittest.TestCase):
    def test_no_runtime_stage_without_explicit_flag(self):
        for argv in (['run_suite.py','--mode','prepare-model'],['run_suite.py','--mode','smoke'],['run_suite.py','--mode','full']):
            with self.subTest(argv=argv),patch('sys.argv',argv),patch('subprocess.Popen') as spawn:
                with self.assertRaises(RuntimeError):run_suite.main()
                spawn.assert_not_called()
    def test_default_mode_only_calls_static_checks(self):
        with patch('sys.argv',['run_suite.py']),patch('static_check.check',return_value={'static_checks_passed':True}) as check,patch('subprocess.Popen') as spawn:
            run_suite.main();self.assertEqual(check.call_count,2);spawn.assert_not_called()
    def test_driver_requires_explicit_smoke_or_full_mode(self):
        for argv in (['driver.py'],['driver.py','--start-approved'],['driver.py','--smoke-only','--full-suite','--start-approved']):
            with self.subTest(argv=argv),patch('sys.argv',argv),patch('subprocess.Popen') as spawn:
                with self.assertRaises(SystemExit):driver.main()
                spawn.assert_not_called()
    def test_session_wrapper_hash_is_unchanged_from_verified_implementation(self):
        expected='3e8d119d66b6c44d9a74fc2f51e8003f10ab91e2f65ab46c20708e8c0196f631'
        self.assertEqual(hashlib.sha256((HERE/'session_policy.py').read_bytes()).hexdigest(),expected)
    def test_native_arguments_and_legacy_score_export_are_controlled(self):
        tree=ast.parse((HERE/'evaluate_task.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        user=next(n.value for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='usr' for t in n.targets))
        values={ast.literal_eval(k):ast.unparse(v) for k,v in zip(user.keys,user.values)}
        self.assertEqual(values['task_config'],"CONFIG['task_config']")
        self.assertEqual(values['test_num'],"CONFIG['test_num']")
        self.assertEqual(values['instruction_type'],"CONFIG['instruction_type']")
        self.assertEqual(values['eval_video_log'],'False')
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval_policy')
        self.assertFalse(any(isinstance(n,ast.Return) for n in ast.walk(func)))

if __name__=='__main__':unittest.main(verbosity=2)
