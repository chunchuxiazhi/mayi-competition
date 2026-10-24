"""Verify wrapper state/RNG interleaving with a tiny fake policy, not VLA inference."""
import json
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch
from session_policy import SessionPolicy

class FakePolicy:
    def __init__(self):
        self.vla=SimpleNamespace(feature_transform=None);self.global_step=0
        self.last_action_chunk=None;self.last_normalized_action_chunk=None
    def infer(self,obs):
        if obs.get('reset'):
            self.global_step=0;self.last_action_chunk=None
            self.vla.feature_transform={'name':obs['episode_seed']}
            return {'action':None}
        self.global_step+=1
        value=(random.random(),float(np.random.rand()),float(torch.rand(())),float(torch.rand((),device='cuda')))
        self.last_action_chunk=value
        return {'action':value,'transform_seed':self.vla.feature_transform['name']}

def main():
    def reset(s,c,seed):return s.infer(c,{'reset':True,'robo_name':'robotwin','episode_seed':seed})
    interleaved=SessionPolicy(FakePolicy(),torch,np);interleaved.open('A');interleaved.open('B')
    reset(interleaved,'A',101);reset(interleaved,'B',202)
    a1=interleaved.infer('A',{});b1=interleaved.infer('B',{})
    reset(interleaved,'A',303)
    b2=interleaved.infer('B',{});a_new=interleaved.infer('A',{})
    isolated=SessionPolicy(FakePolicy(),torch,np);isolated.open('B')
    reset(isolated,'B',202);expected_b1=isolated.infer('B',{});expected_b2=isolated.infer('B',{})
    assert b1==expected_b1 and b2==expected_b2
    assert b2['_session_state']=={'global_step':2,'episode_seed':202}
    assert b2['transform_seed']==202 and a_new['transform_seed']==303
    assert interleaved.sessions['A']['wrapper'].vla is interleaved.sessions['B']['wrapper'].vla
    assert interleaved.sessions['A']['wrapper'] is not interleaved.sessions['B']['wrapper']
    assert interleaved.policy.vla.feature_transform is None
    blocked=False
    try:interleaved.infer('A',{'path_to_pi_model':'different-checkpoint'})
    except ValueError:blocked=True
    assert blocked
    report={'passed':True,'checks':['private wrapper/shared VLA','reset does not reset other session',
        'Python NumPy Torch CPU/CUDA RNG match isolated sequence','FeatureTransform restored','checkpoint change refused'],
        'real_model_loaded':False,'simulation_episodes':0,'scope':'wrapper behavioral check; actual model checked by real first-action session assertions'}
    out=Path('/home/zzj/competitions/lingbot/runs/finetuned-clean-step10000-50x10-20261017/preparation/session-isolation-check.json')
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
