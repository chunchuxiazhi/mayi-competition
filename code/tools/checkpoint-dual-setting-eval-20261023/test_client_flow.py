"""Exercise the real client control flow with tiny fakes; no simulator/model loads."""
import json,sys,types,unittest
from unittest.mock import patch
import numpy as np
import evaluate_task as client
from common import atomic_json,read_events,run_path,seeds,summarize
from recovery import TaskIncomplete,TaskSliceCompleted
from test_preparation import Fixture

class FakeEnvironment:
    def __init__(self,expert_exception=False):
        self.expert_exception=expert_exception;self.setup_seeds=[];self.expert_calls=0;self.step_lim=3
    def setup_demo(self,now_ep_num,seed,is_test,**kwargs):
        self.index=now_ep_num;self.seed=seed;self.setup_seeds.append(seed)
        self.take_action_cnt=0;self.eval_success=False;self.plan_success=True
    def play_once(self):
        self.expert_calls+=1
        if self.expert_exception and self.index==0:raise ValueError('synthetic expert failure')
        return {'info':{}}
    def close_env(self,**kwargs):pass
    def check_success(self):return True
    def set_instruction(self,instruction):self.instruction=instruction
    def get_instruction(self):return self.instruction
    def get_obs(self):
        return {'observation':{'head_camera':{'rgb':None},'left_camera':{'rgb':None},'right_camera':{'rgb':None}},
                'joint_action':{'vector':np.zeros(14)}}
    def take_action(self,action):
        self.take_action_cnt+=1
        if self.index==1:self.eval_success=True

class FakePolicy:
    def __init__(self,interrupt_once=False):
        self._ws=types.SimpleNamespace(close=lambda:None);self.interrupt_once=interrupt_once;self.reset_seeds=[]
    def get_server_metadata(self):return {'training_step':20000,'model_source_sha256':'synthetic-source','session_id':'fake-session'}
    def infer(self,observation):
        if observation.get('reset'):
            self.seed=observation['episode_seed'];self.step=0;self.reset_seeds.append(self.seed)
        else:
            if self.interrupt_once:self.interrupt_once=False;raise KeyboardInterrupt('synthetic interrupted client')
            self.step+=1
        return {'session_id':'fake-session','_session_state':{'global_step':self.step,'episode_seed':self.seed},
                'action':None if observation.get('reset') else np.zeros((2,14))}

class ClientFlow(Fixture):
    def configure(self):
        cfg=self.cfg();cfg.update(test_num=2,seed_base=100000,policy_seed_base=1000000,instruction_seed_base=2000000,
                                 instruction_candidates=100,expert_attempt_limit_per_scene=5,use_length=50)
        plan=[{'task':'lift_pot','episode_index':i,'setting':'demo_clean','baseline_instruction':None,**seeds(cfg,0,i)} for i in range(2)]
        atomic_json(run_path(cfg)/'episodes-plan.json',{'episodes':plan})
        return cfg
    def call(self,cfg,env,policy,retry=False):
        generator=types.SimpleNamespace(generate_episode_descriptions=lambda *args:[])
        # Torch is unused inside eval_policy; forbid its real import, while the
        # NumPy arrays here contain only 28 zeros and never call any ML policy.
        with patch.object(client,'CONFIG',cfg),patch.object(client,'WORKER_ID',0),patch.object(client,'RETRY_INTERRUPTED',retry),patch.object(client,'EPISODE_INDICES',None),\
             patch.dict(sys.modules,{'torch':types.SimpleNamespace(),'generate_episode_instructions':generator}),\
             patch.object(client,'generate_seen',return_value='do the task'),patch('model_source.verify_source',return_value='synthetic-source'):
            client.eval_policy('lift_pot',env,{'render_freq':0,'clear_cache_freq':5},policy,0,test_num=2,instruction_type='seen')
    def test_real_control_flow_records_failures_and_success_and_honors_step_cap(self):
        cfg=self.configure();env=FakeEnvironment();policy=FakePolicy()
        with self.assertRaises(TaskSliceCompleted):self.call(cfg,env,policy)
        events=read_events(run_path(cfg)/'tasks/lift_pot/episodes.jsonl');done=[e for e in events if e['event']=='policy_completed']
        self.assertEqual([(e['success'],e['execution_steps']) for e in done],[(False,3),(True,1)])
        self.assertEqual(policy.reset_seeds,[1000000,1000001]);self.assertEqual(env.expert_calls,2)
        row=summarize(cfg,'lift_pot');self.assertEqual(row['success_rate'],0.5)
    def test_resume_skips_completed_failures_and_does_not_duplicate_records(self):
        cfg=self.configure()
        with self.assertRaises(TaskSliceCompleted):self.call(cfg,FakeEnvironment(),FakePolicy())
        path=run_path(cfg)/'tasks/lift_pot/episodes.jsonl';before=path.read_bytes();env=FakeEnvironment();policy=FakePolicy()
        with self.assertRaises(TaskSliceCompleted):self.call(cfg,env,policy)
        self.assertEqual(path.read_bytes(),before);self.assertEqual(env.expert_calls,0);self.assertEqual(policy.reset_seeds,[])
    def test_expert_exception_preserves_slot_and_does_not_substitute_a_new_seed(self):
        cfg=self.configure();env=FakeEnvironment(expert_exception=True);policy=FakePolicy()
        with self.assertRaises(TaskIncomplete):self.call(cfg,env,policy)
        row=summarize(cfg,'lift_pot');self.assertEqual(row['completed_valid_policy_episodes'],1);self.assertEqual(row['expert_blocked_episode_slots'],1)
        self.assertEqual(set(env.setup_seeds),{100000,100001});self.assertEqual(policy.reset_seeds,[1000001])
    def test_interrupted_round_requires_explicit_same_seed_retry(self):
        cfg=self.configure()
        with self.assertRaises(KeyboardInterrupt):self.call(cfg,FakeEnvironment(),FakePolicy(interrupt_once=True))
        with self.assertRaises(RuntimeError):self.call(cfg,FakeEnvironment(),FakePolicy())
        policy=FakePolicy()
        with self.assertRaises(TaskSliceCompleted):self.call(cfg,FakeEnvironment(),policy,retry=True)
        self.assertEqual(policy.reset_seeds,[1000000,1000001]);row=summarize(cfg,'lift_pot')
        self.assertEqual(row['completed_valid_policy_episodes'],2);self.assertEqual(row['interrupted_attempts'],1)

if __name__=='__main__':unittest.main(verbosity=2)
