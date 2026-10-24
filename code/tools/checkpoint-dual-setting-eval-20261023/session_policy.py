"""Per-connection policy state and RNG, around ONE resident weight object.

No inference occurs on import. Calls are synchronous and serialized; no batching.
"""
import copy
import random
import threading

class SessionPolicy:
    def __init__(self,policy,torch,np,max_sessions=2):
        self.policy=policy;self.torch=torch;self.np=np
        self.max_sessions=max_sessions;self.sessions={};self.lock=threading.RLock()
        self.initial_fields=dict(policy.__dict__)
        self.initial_transform=policy.vla.feature_transform

    def capture_rng(self):
        return (random.getstate(),self.np.random.get_state(),self.torch.get_rng_state(),self.torch.cuda.get_rng_state_all())

    def restore_rng(self,state):
        random.setstate(state[0]);self.np.random.set_state(state[1]);self.torch.set_rng_state(state[2]);self.torch.cuda.set_rng_state_all(state[3])

    def open(self,connection_id):
        with self.lock:
            if connection_id in self.sessions or len(self.sessions)>=self.max_sessions:raise RuntimeError('Session capacity exceeded')
            # Copy the small policy wrapper, NEVER model tensors/processor/weights.
            wrapper=copy.copy(self.policy)
            wrapper.__dict__=dict(self.initial_fields)
            self.sessions[connection_id]={'wrapper':wrapper,'transform':None,'rng':None,'initialized':False,'episode_seed':None}

    def close(self,connection_id):
        with self.lock:self.sessions.pop(connection_id,None)

    def infer(self,connection_id,observation):
        with self.lock:
            ctx=self.sessions[connection_id];policy=ctx['wrapper']
            if observation.get('path_to_pi_model') is not None:raise ValueError('Checkpoint changes are prohibited')
            resetting=observation.get('reset') is True
            if resetting:
                if observation.get('robo_name')!='robotwin':raise ValueError('Only robotwin mapping allowed')
                if type(observation.get('episode_seed')) is not int:raise ValueError('Reset requires recorded policy RNG seed')
            elif not ctx['initialized']:raise RuntimeError('Reset this connection before inference')
            original_rng=self.capture_rng()
            original_transform=self.policy.vla.feature_transform
            try:
                if resetting:
                    seed=observation['episode_seed']
                    random.seed(seed);self.np.random.seed(seed%(2**32))
                    self.torch.manual_seed(seed);self.torch.cuda.manual_seed_all(seed)
                else:self.restore_rng(ctx['rng'])
                # FeatureTransform is stored on shared VLA by upstream reset().
                # Swap it ONLY inside this lock; wrapper caches and counters are private.
                policy.vla.feature_transform=ctx['transform']
                with self.torch.inference_mode():result=dict(policy.infer(dict(observation)))
                # Complete CUDA work before allowing another session to enter or restoring RNG.
                self.torch.cuda.synchronize()
                ctx['transform']=policy.vla.feature_transform
                ctx['rng']=self.capture_rng();ctx['initialized']=True
                if resetting:ctx['episode_seed']=observation['episode_seed']
                result['_session_state']={'global_step':policy.global_step,'episode_seed':ctx['episode_seed']}
                return result
            except BaseException:
                ctx['initialized']=False
                raise
            finally:
                self.policy.vla.feature_transform=original_transform
                self.restore_rng(original_rng)
