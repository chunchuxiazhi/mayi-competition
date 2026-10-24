"""Guard the unchanged trainer's entire resume block; optional zero-update verification.

The AST transform changes only the resume exception handler and adds a boundary
after that block, before any training iteration. The upstream file is not edited.
"""
import ast,copy,hashlib,inspect,json,os,textwrap
from pathlib import Path
from storage_policy import atomic_json

def transform_source(source):
    tree=ast.parse(textwrap.dedent(source));function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    attempts=[n for n in ast.walk(function) if isinstance(n,ast.Try) and any(isinstance(c,ast.Call)
        and isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name)
        and c.func.value.id=='Checkpointer' and c.func.attr=='load' for stmt in n.body for c in ast.walk(stmt))]
    if len(attempts)!=1 or len(attempts[0].handlers)!=1:raise RuntimeError('Trainer resume structure changed')
    handler=attempts[0].handlers[0]
    if not isinstance(handler.type,ast.Name) or handler.type.id!='Exception' or not handler.name:
        raise RuntimeError('Unexpected upstream resume handler')
    replacement=ast.parse("raise SystemExit('Full checkpoint restoration failed; no fallback: '+repr("+handler.name+")) from "+handler.name).body
    handler.body=replacement
    boundaries=[i for i,n in enumerate(function.body) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='candidates']
    if len(boundaries)!=1:raise RuntimeError('Cannot find post-resume/pre-training boundary')
    function.body.insert(boundaries[0]+1,ast.parse('__gb10_resume_boundary(locals())').body[0])
    ast.fix_missing_locations(tree)
    return tree

def trees_equal(a,b):
    import torch,numpy as np
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(trees_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,type(a)):return len(a)==len(b) and all(trees_equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

def install(official,config,control):
    original=inspect.getsource(official.main)
    tree=transform_source(original)
    project=Path(control['project_root']);run=Path(config['train']['output_dir'])
    probe=os.environ.get('LINGBOT_VERIFY_RESUME_ONLY')=='1'
    def boundary(scope):
        import torch,random,numpy as np
        import torch.distributed as dist
        step=int(scope['global_step']);candidates=scope['candidates']
        if candidates and not scope.get('loaded'):raise SystemExit('Checkpoint candidates exist but none fully restored')
        if not candidates and step!=0:raise SystemExit('Unexpected fresh-run step')
        if not probe:
            atomic_json(run/'resume-boundary.json',{'restored':bool(candidates),'global_step':step,
                'full_resume_error_fallback_disabled':True,'launch_attempt':os.environ.get('LINGBOT_TRAIN_ATTEMPT')})
            return
        if not candidates or step!=20:raise RuntimeError('The zero-update probe requires the completed 20-step pilot')
        extra=scope['state']['extra_state'];loader=scope['train_dataloader'];optimizer=scope['optimizer']
        if not trees_equal(scope['lr_scheduler'].state_dict(),extra['lr_scheduler']):raise RuntimeError('LR scheduler state mismatch')
        if not torch.equal(torch.get_rng_state(),extra['torch_rng_state']):raise RuntimeError('CPU RNG mismatch')
        if not trees_equal(torch.cuda.get_rng_state_all(),extra['cuda_random_states']):raise RuntimeError('CUDA RNG mismatch')
        if not trees_equal(random.getstate(),extra['python_random_state']) or not trees_equal(np.random.get_state(),extra['numpy_random_state']):
            raise RuntimeError('Python/NumPy RNG mismatch')
        steps=[int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
        if not steps or min(steps)<1 or max(steps)!=step:raise RuntimeError('Optimizer steps were not restored')
        moments=[v['exp_avg'] for v in optimizer.state.values() if 'exp_avg' in v]
        if not moments:raise RuntimeError('Adam moments missing')
        sentinel='model.action_out_proj.weight';parameter=next(p for n,p in scope['model'].named_parameters() if n==sentinel)
        actual=parameter.detach();actual=actual.to_local() if hasattr(actual,'to_local') else actual
        from safetensors import safe_open
        expected=None
        for path in (run/'exports/global_step_20/hf_ckpt').glob('*.safetensors'):
            with safe_open(path,framework='pt',device='cpu') as f:
                if sentinel in f.keys():expected=f.get_tensor(sentinel);break
        if expected is None or not torch.equal(actual.cpu(),expected):raise RuntimeError('Restored model differs from committed/exported model')
        # Check the actual next normalized batch twice after restoring the same
        # loader and RNG state. No model forward/backward or optimizer.step.
        def restore_rng():
            torch.set_rng_state(extra['torch_rng_state']);torch.cuda.set_rng_state_all(extra['cuda_random_states'])
            random.setstate(extra['python_random_state']);np.random.set_state(extra['numpy_random_state'])
        saved=extra['train_dataloader'];loader.load_state_dict(copy.deepcopy(saved));restore_rng();first=next(iter(loader))
        loader.load_state_dict(copy.deepcopy(saved));restore_rng();second=next(iter(loader))
        if not trees_equal(first,second):raise RuntimeError('Restored DataLoader next batch is not deterministic')
        if len(first)!=config['train']['gradient_accumulation_steps']:raise RuntimeError('Restored micro-batch count mismatch')
        evidence={'passed':True,'checkpoint_signature':extra['gb10_config_signature'],'restored_global_step':step,
            'optimizer_state_tensors':len(optimizer.state),'adam_moment_groups':len(moments),'optimizer_step_range':[min(steps),max(steps)],
            'lr_scheduler_restored':True,'python_numpy_cpu_cuda_rng_restored':True,'model_projection_exact_match':True,
            'dataloader_next_batch_reproduced':True,'optimizer_steps_executed':0,'model_forward_backward_executed':False,
            'source_checkpoint':scope['load_checkpoint_path'],'resume_guard_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        destination=Path(os.environ['LINGBOT_RESUME_PROBE_RESULT'])
        if not destination.resolve().is_relative_to(project/'runs/finetune-clean-v3-gb10/resume-verification'):
            raise RuntimeError('Probe result escaped reviewed directory')
        atomic_json(destination,evidence)
        scope['writer'].close();dist.barrier();dist.destroy_process_group()
        raise SystemExit(0)
    namespace=official.__dict__.copy();namespace['__gb10_resume_boundary']=boundary
    exec(compile(tree,official.__file__,'exec'),namespace)
    official.main=namespace['main']

