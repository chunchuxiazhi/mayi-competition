"""Unseen must reach the real eval control flow, not only a JSON label."""
import copy,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import recovery,evaluate_task as client,validation
from common import atomic_json,read_events,run_path,seeds
from test_preparation import Fixture
from test_client_flow import FakeEnvironment,FakePolicy

class Unseen(Fixture):
    def test_wrong_randomized_instruction_rejected_before_runtime_import(self):
        cfg=json.loads((Path(__file__).parent/'config-randomized.json').read_text())
        cfg['project_root']=str(self.root);cfg['instruction_type']='seen'
        with self.assertRaisesRegex(ValueError,'setting/instruction'):validation.validate_config(cfg)
    def test_official_unseen_pool_reaches_policy_observation_and_terminal(self):
        cfg=self.cfg('demo_randomized');cfg.update(instruction_type='unseen',test_num=2,seed_base=100000,policy_seed_base=1000000,
          instruction_seed_base=2000000,instruction_candidates=100,expert_attempt_limit_per_scene=5,use_length=50)
        plan=[{'task':'lift_pot','episode_index':i,'setting':'demo_randomized','instruction_type':'unseen',
               'baseline_instruction':None,**seeds(cfg,0,i)} for i in range(2)]
        atomic_json(run_path(cfg)/'episodes-plan.json',{'episodes':plan})
        generator=types.SimpleNamespace(generate_episode_descriptions=lambda *args:[{'seen':['SEEN ONLY'],'unseen':['UNSEEN ONLY']}])
        class Policy(FakePolicy):
            def __init__(self):super().__init__();self.instructions=[]
            def infer(self,obs):
                if not obs.get('reset'):self.instructions.append(obs['task'])
                return super().infer(obs)
        policy=Policy()
        with patch.object(client,'CONFIG',cfg),patch.object(client,'WORKER_ID',0),patch.object(client,'RETRY_INTERRUPTED',False),patch.object(client,'EPISODE_INDICES',None),\
             patch.dict(sys.modules,{'torch':types.SimpleNamespace(),'generate_episode_instructions':generator}),\
             patch('model_source.verify_source',return_value='synthetic-source'):
            with self.assertRaises(recovery.TaskSliceCompleted):
                client.eval_policy('lift_pot',FakeEnvironment(),{'render_freq':0,'clear_cache_freq':5},policy,0,test_num=2,instruction_type='unseen')
        self.assertTrue(policy.instructions);self.assertEqual(set(policy.instructions),{'UNSEEN ONLY'})
        done=[e for e in read_events(run_path(cfg)/'tasks/lift_pot/episodes.jsonl') if e['event']=='policy_completed']
        self.assertEqual(len(done),2)
        for row in done:
            self.assertEqual(row['instruction_type'],'unseen');self.assertEqual(row['instruction'],'UNSEEN ONLY')
            self.assertEqual(row['instruction_evidence']['pool'],'unseen')
    def test_missing_unseen_does_not_fall_back_to_seen(self):
        generator=types.SimpleNamespace(generate_episode_descriptions=lambda *args:[{'seen':['SEEN ONLY'],'unseen':[]}])
        with patch.dict(sys.modules,{'generate_episode_instructions':generator}):
            with self.assertRaises(recovery.TaskIncomplete):recovery.generate_instruction('lift_pot',{},1,100,self.root,instruction_type='unseen')

if __name__=='__main__':unittest.main(verbosity=2)
