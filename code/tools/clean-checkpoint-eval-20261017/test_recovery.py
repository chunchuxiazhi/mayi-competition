"""Control-flow regression tests with fake environment/model, not simulation."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys
import numpy as np
import evaluate_task as client
from common import load_config,read_events,append_event,summarize,seeds,fingerprint
from recovery import TaskIncomplete,TaskSliceCompleted,validate_revision,recorded_expert,validate_metadata,global_resource_fault

class FakeEnvironment:
    def __init__(self,info=None,setup_error=False):
        self.info={'{A}':'060_kitchenpot/base1'} if info is None else info
        self.plan_success=False;self.calls=[];self.setup_error=setup_error
        self.step_lim=2;self.take_action_cnt=0;self.eval_success=False
    def setup_demo(self,now_ep_num,seed,**args):
        self.calls.append(('setup',now_ep_num,seed));self.take_action_cnt=0
        if self.setup_error:raise ValueError('invalid scene')
    def play_once(self):self.calls.append(('expert',));return {'info':self.info}
    def close_env(self,**kw):pass
    def check_success(self):return False
    def set_instruction(self,instruction):self.instruction=instruction
    def get_instruction(self):return self.instruction
    def get_obs(self):return {'observation':{k:{'rgb':np.zeros((2,2,3),dtype=np.uint8)} for k in ('head_camera','left_camera','right_camera')},'joint_action':{'vector':np.zeros(14)}}
    def take_action(self,act):self.take_action_cnt+=1

class FakeModel:
    def __init__(self,mismatch=False):
        self._ws=types.SimpleNamespace(close=lambda:None);self.calls=[];self.mismatch=mismatch
    def get_server_metadata(self):return {'session_id':'fake-session','training_step':10000,'model_source_sha256':'fake-source'}
    def infer(self,obs):
        self.calls.append(obs)
        if obs.get('reset'):self.step=0;self.seed=obs['episode_seed']
        else:self.step+=1
        return {'session_id':'wrong' if self.mismatch else 'fake-session',
            '_session_state':{'global_step':self.step,'episode_seed':self.seed},'action':np.zeros((1,14))}

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.source_patch=patch('model_source.verify_source',return_value='fake-source');self.source_patch.start();self.addCleanup(self.source_patch.stop)
        self.plan_patch=patch('model_source.expected_episode');self.plan_patch.start();self.addCleanup(self.plan_patch.stop)
        self.temp=tempfile.TemporaryDirectory();self.config=load_config()
        sys.path.insert(0,str(Path(self.config['project_root'])/'repos/RoboTwin/description/utils'))
        self.config['run_relative']=self.temp.name;self.config['tasks']=['lift_pot'];self.config['test_num']=2
        client.CONFIG=self.config;client.WORKER_ID=0;client.RETRY_INTERRUPTED=True;client.EPISODE_INDICES=None
        self.ledger=Path(self.temp.name)/'tasks/lift_pot/episodes.jsonl'
        self.expert=self.ledger.with_name('expert.jsonl')
    def tearDown(self):self.temp.cleanup()
    def run_client(self,env=None,model=None):
        return client.eval_policy('lift_pot',env or FakeEnvironment(),{'render_freq':0,'clear_cache_freq':10},
            model or FakeModel(),100000,test_num=2,instruction_type='seen')
    def test_expert_failure_runs_same_seeds_and_preserves_failed_policy_on_resume(self):
        env=FakeEnvironment();self.run_client(env)
        self.assertEqual([c for c in env.calls if c[0]=='setup'],[('setup',0,100000),('setup',0,100000),('setup',1,100001),('setup',1,100001)])
        before=self.ledger.read_bytes();second=FakeEnvironment();self.run_client(second)
        self.assertEqual(second.calls,[]);self.assertEqual(self.ledger.read_bytes(),before)
        row=summarize(self.config,'lift_pot');self.assertEqual(row['completed_valid_policy_episodes'],2)
        self.assertEqual(row['success_rate'],0);self.assertEqual(row['expert_failed_or_exception'],2)
    def test_existing_failed_expert_and_historical_block_are_reused(self):
        base={'task':'lift_pot','episode_index':1,'setting':'demo_clean','instruction_type':'seen',**seeds(self.config,0,1),'attempt_id':'old-expert'}
        append_event(self.expert,{**base,'event':'expert_started'})
        append_event(self.expert,{**base,'event':'expert_failed','expert_success':False,'episode_info':{'{A}':'060_kitchenpot/base1'}})
        append_event(self.ledger,{**base,'event':'episode_blocked'})
        client.EPISODE_INDICES=[1];env=FakeEnvironment()
        with self.assertRaises(TaskSliceCompleted):self.run_client(env)
        self.assertFalse(any(c[0]=='expert' for c in env.calls))
        self.assertEqual(len(read_events(self.expert)),2)
        self.assertEqual(summarize(self.config,'lift_pot')['expert_blocked_episode_slots'],0)
    def test_invalid_metadata_leaves_slots_incomplete_without_policy(self):
        model=FakeModel()
        with self.assertRaises(TaskIncomplete):self.run_client(FakeEnvironment({'{A}':'missing/object'}),model)
        self.assertEqual(model.calls,[])
        self.assertEqual(summarize(self.config,'lift_pot')['completed_valid_policy_episodes'],0)
        self.assertEqual(len([e for e in read_events(self.expert) if e['event']=='expert_started']),2)
    def test_session_mismatch_is_global(self):
        with self.assertRaisesRegex(RuntimeError,'session mismatch'):self.run_client(model=FakeModel(mismatch=True))
        self.assertEqual(read_events(self.ledger)[-1]['failure_scope'],'global')
    def test_interrupted_slot_resume_same_instruction_and_count_once(self):
        self.run_client();events=read_events(self.ledger)
        # Separate fixture preserving one interrupted attempt, rather than editing a real ledger.
        second=Path(self.temp.name)/'interrupted';self.config['run_relative']=str(second)
        directory=second/'tasks/lift_pot'
        for e in read_events(self.expert):append_event(directory/'expert.jsonl',e)
        for e in events:
            if e['episode_index']!=0:continue
            if e['event']=='policy_completed':e={**e,'event':'policy_interrupted','success':None}
            append_event(directory/'episodes.jsonl',e)
        self.run_client()
        row=summarize(self.config,'lift_pot')
        self.assertEqual(row['interrupted_attempts'],1);self.assertEqual(row['completed_valid_policy_episodes'],2)
    def test_provenance_mismatch_and_protected_revision_rejected(self):
        base={'task':'lift_pot','episode_index':1,'setting':'demo_clean','instruction_type':'seen','seed':100001}
        with self.assertRaisesRegex(RuntimeError,'provenance'):recorded_expert([{**base,'task':'other','event':'expert_failed','expert_success':False}],base)
        original=load_config();current=copy.deepcopy(original)
        current['control_revision']={'version':2,'parent_config_sha256':fingerprint(original)}
        current['expert_failure_policy']='validate_metadata_then_policy_same_seed';validate_revision(original,current)
        current['test_num']=100
        with self.assertRaisesRegex(RuntimeError,'protected'):validate_revision(original,current)
    def test_bad_arm_parameter_rejected(self):
        with self.assertRaises(TaskIncomplete):validate_metadata({'{a}':None},Path(self.config['project_root'])/'repos/RoboTwin')
    def test_global_resource_failure_not_classified_as_scene_problem(self):
        self.assertTrue(global_resource_fault(RuntimeError('CUDA out of memory')))
        self.assertTrue(global_resource_fault(MemoryError()))
        self.assertTrue(global_resource_fault(ImportError('missing native module')))
        self.assertFalse(global_resource_fault(ValueError('invalid scene')))

if __name__=='__main__':unittest.main(verbosity=2)
