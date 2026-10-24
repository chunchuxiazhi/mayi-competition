"""Adapters around the unchanged official trainer: one rank, bounded resources, strict recovery."""
from pathlib import Path
import gc,hashlib,json,math,os,time
from storage_policy import GIB,atomic_json,check_space,commit,prune,validate

def available_memory():
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024

def signature(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

def is_restore_initialization(state,optimizer):
    if not state.get('restoring_optimizer'):return False
    if any(float(group['lr'])!=0.0 for group in optimizer.param_groups):
        raise RuntimeError('Unexpected nonzero-LR optimizer step during state restoration')
    state['restore_initialization_calls']=state.get('restore_initialization_calls',0)+1
    return True

def append(path,event):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>64*1024**2:raise RuntimeError('Structured training log cap exceeded')
    with path.open('a') as f:f.write(json.dumps({'unix_time':time.time(),'launch_attempt':os.environ.get('LINGBOT_TRAIN_ATTEMPT'),**event},allow_nan=False)+'\n');f.flush()

def install(official,config,control):
    import torch
    import torch.distributed as dist
    import numpy as np
    import random
    import safe_video
    from lingbotvla.checkpoint.checkpointer import DistributedCheckpointer,ModelState,OptimizerState
    import lingbotvla.checkpoint.checkpointer as checkpoint_module
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    safe_video.install()
    run=Path(config['train']['output_dir']);run.mkdir(parents=True,exist_ok=True)
    expected=signature(config);state={'step':0,'losses':[],'last_time':time.monotonic()}
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise RuntimeError('Exactly one training rank is supported')
    props=torch.cuda.get_device_properties(0)
    cap=int(control['cuda_allocator_limit_gib']*GIB)
    torch.cuda.set_per_process_memory_fraction(min(cap/props.total_memory,0.80),0)
    torch.set_num_threads(control['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.multiprocessing.set_start_method('spawn',force=True)

    def memory_check(checkpoint=False):
        floor=control['checkpoint_memory_floor_gib'] if checkpoint else control['memory_free_floor_gib']
        if available_memory()<floor*GIB:
            raise MemoryError('Unified-memory reserve reached; previous committed checkpoint is retained')

    original_clip=torch.nn.utils.clip_grad_norm_
    def checked_clip(*args,**kwargs):
        kwargs['error_if_nonfinite']=True
        result=original_clip(*args,**kwargs)
        state['gradient_norm']=float(result)
        return result
    torch.nn.utils.clip_grad_norm_=checked_clip

    def save_phase(phase,path=None):
        status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
        append(run/'logs/save-memory.jsonl',{'phase':phase,'step':state['step'],'checkpoint_path':path,
            'unified_mem_available_gib':available_memory()/GIB,'process_rss_kib':int(status['VmRSS'].split()[0]),
            'process_peak_rss_kib':int(status['VmHWM'].split()[0]),'cuda_allocated_gib':torch.cuda.memory_allocated()/GIB,
            'cuda_reserved_gib':torch.cuda.memory_reserved()/GIB})
    from dcp_stream_writer import streaming_writer_class
    original_writer=streaming_writer_class(save_phase)
    def bounded_writer(*args,**kwargs):
        return original_writer(*args,**kwargs)
    checkpoint_module.FileSystemWriter=bounded_writer

    class ManagedCheckpointer:
        @classmethod
        def save(cls,path,training_state,global_steps=None,save_async=False):
            if save_async:raise ValueError('Async DCP is disabled for this unified-memory budget')
            save_phase('save_begin',str(path))
            memory_check(checkpoint=True);check_space(control)
            destination=Path(path)/f'global_step_{global_steps}'
            if global_steps!=state['step']:raise RuntimeError('Optimizer/checkpoint step mismatch')
            if destination.exists():raise RuntimeError('Refusing to overwrite an existing checkpoint')
            extra=training_state['extra_state']
            extra['python_random_state']=random.getstate();extra['numpy_random_state']=np.random.get_state()
            extra['cuda_random_states']=torch.cuda.get_rng_state_all();extra['gb10_config_signature']=expected
            DistributedCheckpointer.save(path,training_state,global_steps=global_steps,save_async=False)
            save_phase('all_components_written',str(destination))
            dist.barrier()
            commit(path,destination,expected)
            from continuation import prune_chain
            removed=prune_chain(config,control)
            append(run/'logs/checkpoints.jsonl',{'event':'committed','step':global_steps,'path':str(destination),'removed':removed})
            torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
            save_phase('save_cleanup_complete',str(destination))
            memory_check()
            state['first_forward_after_save']=True

        @classmethod
        def load(cls,path,training_state,allow_partial_load=False,**kwargs):
            if allow_partial_load:raise ValueError('Partial resume is prohibited')
            path=Path(path)
            from continuation import validate_resume_source
            load_signature=validate_resume_source(path,config,control)
            # The upstream loader swallows optimizer errors; require full restoration here.
            dcp.load({'state':ModelState(training_state['model'])},storage_reader=FileSystemReader(path/'model'))
            # PyTorch initializes an empty Adam state using zero gradients and
            # zero LR. This is state allocation, not a training update; exclude
            # it from the training-only micro-batch/step telemetry hooks.
            state['restoring_optimizer']=True
            try:
                dcp.load({'state':OptimizerState(training_state['model'],training_state['optimizer'])},storage_reader=FileSystemReader(path/'optimizer'))
            finally:state['restoring_optimizer']=False
            extra=torch.load(path/'extra_state/extra_state_rank_0.pt',map_location='cpu',weights_only=False)
            if extra.get('gb10_config_signature')!=load_signature:raise ValueError('Resume configuration changed')
            training_state['extra_state']=extra;state['step']=int(extra['global_step'])
            random.setstate(extra['python_random_state']);np.random.set_state(extra['numpy_random_state'])
            torch.cuda.set_rng_state_all(extra['cuda_random_states'])
            append(run/'logs/checkpoints.jsonl',{'event':'resumed','step':state['step'],'path':str(path),
                'zero_lr_optimizer_state_initializations':state.get('restore_initialization_calls',0)})
            return training_state

    def factory(dist_backend,ckpt_manager):
        if dist_backend!='fsdp2' or ckpt_manager!='dcp':raise ValueError('Unreviewed training/checkpoint backend')
        return ManagedCheckpointer
    official.build_checkpointer=factory

    original_model=official.build_foundation_model
    def build_model(*args,**kwargs):
        memory_check()
        model=original_model(*args,**kwargs)
        def before_forward(module,inputs):
            if state.pop('first_forward_after_save',False):
                save_phase('first_forward_after_save');memory_check()
        model.register_forward_pre_hook(before_forward)
        def capture_loss(module,inputs,output):
            loss=float(output[0].detach())
            if not math.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            state['losses'].append(loss)
        model.register_forward_hook(capture_loss)
        return model
    official.build_foundation_model=build_model

    original_optimizer=official.build_optimizer
    def build_optimizer(model,*args,**kwargs):
        trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
        if any('.qwenvl.' in n for n in trainable):raise RuntimeError('Vision-language backbone unexpectedly trainable')
        if not any('qwen_expert' in n for n in trainable):raise RuntimeError('No action expert is trainable')
        atomic_json(run/'trainable-parameters.json',{'total_trainable':sum(trainable.values()),'tensors':trainable})
        optimizer=original_optimizer(model,*args,**kwargs)
        optimizer.register_step_pre_hook(lambda opt,a,k:memory_check())
        def after_step(opt,args,kwargs):
            if is_restore_initialization(state,opt):return
            now=time.monotonic();state['step']+=1
            if len(state['losses'])!=config['train']['gradient_accumulation_steps']:
                raise RuntimeError('Unexpected number of micro-batches in optimizer update')
            append(run/'logs/updates.jsonl',{'optimizer_step':state['step'],'micro_batches':len(state['losses']),
                'mean_loss':sum(state['losses'])/len(state['losses']),'seconds_since_previous_update':now-state['last_time'],
                'cuda_allocated_gib':torch.cuda.memory_allocated()/GIB,'cuda_reserved_gib':torch.cuda.memory_reserved()/GIB,
                'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/GIB,
                'unified_mem_available_gib':available_memory()/GIB,'gradient_norm':state.get('gradient_norm'),
                'learning_rates':[g['lr'] for g in opt.param_groups]})
            state['losses'].clear();state['last_time']=now
        optimizer.register_step_post_hook(after_step)
        return optimizer
    official.build_optimizer=build_optimizer

    original_dataset=official.build_vla_dataset
    def build_dataset(*args,**kwargs):
        dataset=original_dataset(*args,**kwargs)
        records=json.loads(Path(control['task_episode_map']).read_text())
        selected=control['selected_tasks'];allowed=set(selected)
        if control['purpose']=='joint_50_tasks' and len(allowed)!=50:raise ValueError('Joint run must include all 50 tasks')
        if len(dataset)!=control['expected_frames']:raise ValueError('Dataset frame count changed')
        rows=[r for r in records if r['task'] in allowed]
        if len(rows)!=50*len(allowed) or any(sum(r['task']==t for r in rows)!=50 for t in allowed):
            raise ValueError('Task/episode selection mismatch')
        atomic_json(run/'selected-episodes.json',rows)
        if len(allowed)!=50:
            from torch.utils.data import Subset
            indices=[i for row in rows for i in range(row['from'],row['to'])]
            return Subset(dataset,indices)
        return dataset
    official.build_vla_dataset=build_dataset

    # num_workers=0 avoids a reproduced native shutdown failure in the normalized
    # StatefulDataLoader. Do not call private shutdown methods or suppress it.
    if config['data']['num_workers']!=0:raise RuntimeError('Unverified multiprocessing data-loader profile')
    return lambda:None
