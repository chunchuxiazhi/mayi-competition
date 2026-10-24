"""Pinned full-state continuation, explicit second-stage LR and cross-run retention.

The source run/config/commit records remain immutable. No tensor files are copied.
Only the old run's two known checkpoint folders and this run's checkpoints may
be pruned, and only after a new checkpoint is committed and all candidates pass.
"""
import ast,copy,hashlib,json,math,os,shutil
from pathlib import Path
from storage_policy import atomic_json,validate,checked_checkpoint

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def signature(config):return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

def learning_rate(step):
    if not 10000<=step<=20000:raise ValueError('LR step outside approved continuation')
    return 5e-6+5e-6*0.5*(1+math.cos(math.pi*(step-10000)/10000))

class ContinuationSchedule:
    """Absolute optimizer step index; no LambdaLR closure/base-LR ambiguity."""
    def __init__(self,optimizer):
        self.optimizer=optimizer;self.last_epoch=10000;self._restored=False
        self._set(10000)
    def _set(self,step):
        rate=learning_rate(step)
        for group in self.optimizer.param_groups:group['lr']=rate
        self._last_lr=[rate for _ in self.optimizer.param_groups]
    def step(self):
        if not self._restored:raise RuntimeError('Scheduler cannot update before full-state resume')
        if self.last_epoch>=20000:raise RuntimeError('Continuation budget exceeded')
        self.last_epoch+=1;self._set(self.last_epoch)
    def get_last_lr(self):return self._last_lr.copy()
    def state_dict(self):
        return {'kind':'clean50-continuation-cosine-v1','origin':10000,'end':20000,
                'lr_start':1e-5,'lr_end':5e-6,'last_epoch':self.last_epoch,'_last_lr':self.get_last_lr()}
    def load_state_dict(self,saved):
        step=int(saved['last_epoch'])
        rates=saved.get('_last_lr',[])
        if len(rates)!=len(self.optimizer.param_groups):raise RuntimeError('Scheduler parameter-group mismatch')
        if saved.get('kind')=='clean50-continuation-cosine-v1':
            expected=self.state_dict()
            for key in ('origin','end','lr_start','lr_end'):
                if saved.get(key)!=expected[key]:raise RuntimeError('Continuation scheduler recipe changed')
            target=learning_rate(step)
        elif 'kind' not in saved:
            # One explicitly authorized migration from the finished first cosine.
            if step!=10000:raise RuntimeError('Only the completed first-stage scheduler can migrate')
            if any(not math.isclose(float(x),2e-5,rel_tol=0,abs_tol=1e-14) for x in saved.get('base_lrs',[])):
                raise RuntimeError('First-stage scheduler base learning rate changed')
            target=1e-5
        else:raise RuntimeError('Unknown checkpoint scheduler')
        if any(not math.isclose(float(x),target,rel_tol=0,abs_tol=1e-14) for x in rates):
            raise RuntimeError('Saved scheduler rates disagree with approved stage/step')
        if any(not math.isclose(float(g['lr']),target,rel_tol=0,abs_tol=1e-14) for g in self.optimizer.param_groups):
            raise RuntimeError('Restored optimizer and scheduler rates disagree')
        self.last_epoch=step;self._set(step);self._restored=True

def validate_plan(config,control):
    plan=control['continuation'];source_run=Path(plan['source_run']);run=Path(config['train']['output_dir'])
    base=json.loads((source_run/'frozen-config.json').read_text())
    if digest(source_run/'frozen-config.json')!=plan['source_config_sha256'] or signature(base)!=plan['source_signature']:
        raise RuntimeError('Pinned source config changed')
    expected=copy.deepcopy(base)
    expected['train'].update(output_dir=str(run),max_steps=20000,lr='1e-05',lr_min='5e-06',lr_warmup_ratio=0.0)
    if expected!=config:raise RuntimeError('Unapproved training changes outside the continuation plan')
    root=Path(control['project_root']).resolve();training=root/'runs/finetune-clean-v3-gb10'
    if source_run!=training/'clean50-10000-fixed1000' or run!=training/'clean50-20000-from10000-20261018':
        raise RuntimeError('Run roots differ from the reviewed competition paths')
    if plan['source_step']!=10000 or plan['target_step']!=20000 or control['keep_checkpoints']!=2:
        raise RuntimeError('Unapproved budget or retention')
    if not (source_run/'exports/global_step_10000/hf_ckpt/model.safetensors.index.json').is_file():
        raise RuntimeError('Preserve the verified 10000-step inference export as the comparison model')
    return {'passed':True,'source_step':10000,'target_step':20000,'additional_updates':10000,
            'changed_train_fields':['output_dir','max_steps','lr','lr_min','lr_warmup_ratio'],
            'lr_stage':'1e-5 to 5e-6 cosine, no new warmup','source_config_immutable':True,
            'tensor_files_copied':False,'complete_checkpoints_across_both_runs':2,
            'new_save_steps':list(range(11000,20001,1000))}

def validate_resume_source(path,config,control):
    path=Path(path);plan=control['continuation'];source=Path(plan['source_run'])/'checkpoints/global_step_10000'
    if path==source:
        if digest(path/'complete.json')!=plan['source_commit_sha256']:raise RuntimeError('Source commit record changed')
        validate(source.parent,path,plan['source_signature']);return plan['source_signature']
    run=Path(config['train']['output_dir']);step=int(path.name.removeprefix('global_step_'))
    if not 10000<step<=20000 or step%1000:raise RuntimeError('Unexpected continuation checkpoint step')
    validate(run/'checkpoints',path,signature(config));return signature(config)

def select_checkpoint(config,control):
    folder=Path(config['train']['output_dir'])/'checkpoints'
    candidates=sorted(folder.glob('global_step_*'),key=lambda p:int(p.name.removeprefix('global_step_')),reverse=True)
    selected=candidates[0] if candidates else Path(control['continuation']['source_run'])/'checkpoints/global_step_10000'
    validate_resume_source(selected,config,control)
    return selected

def prune_chain(config,control):
    plan=control['continuation'];current=Path(config['train']['output_dir'])/'checkpoints'
    source=Path(plan['source_run'])/'checkpoints';entries=[]
    for folder,sig in ((source,plan['source_signature']),(current,signature(config))):
        for p in folder.glob('global_step_*'):
            step=int(p.name.removeprefix('global_step_'))
            if folder==source and step not in (9000,10000):raise RuntimeError('Unexpected old-run checkpoint; do not prune')
            if folder==current and (step<=10000 or step>20000 or step%1000):raise RuntimeError('Unexpected new-run checkpoint')
            # Never delete an incomplete save; stop cleanup for manual inspection.
            validate(folder,p,sig);entries.append((step,folder,p,sig))
    entries.sort(reverse=True,key=lambda row:row[0])
    if len({e[0] for e in entries})!=len(entries):raise RuntimeError('Duplicate checkpoints across runs')
    if not entries or entries[0][1]!=current:raise RuntimeError('No newer committed continuation checkpoint')
    removed=[]
    for _,folder,path,sig in entries[control['keep_checkpoints']:]:
        target=checked_checkpoint(folder,path)
        validate(folder,target,sig)
        shutil.rmtree(target);removed.append(str(target))
    return removed

def install(official,config,control,resume_guard):
    validate_plan(config,control)
    selected=select_checkpoint(config,control)
    original_parse=official.parse_args
    def parse(rootclass):
        args=original_parse(rootclass)
        args.train.load_checkpoint_path=str(selected)
        return args
    official.parse_args=parse
    def build(optimizer,**kwargs):
        if kwargs['train_steps']!=20000 or kwargs['lr_warmup_ratio']!=0.0:
            raise RuntimeError('Unexpected native scheduler construction')
        return ContinuationSchedule(optimizer)
    official.build_lr_scheduler=build
    original_transform=resume_guard.transform_source
    def transform(source):
        tree=original_transform(source)
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        matches=[i for i,n in enumerate(function.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                 and isinstance(n.value.func,ast.Name) and n.value.func.id=='__gb10_resume_boundary']
        if len(matches)!=1:raise RuntimeError('Strict resume boundary changed')
        function.body.insert(matches[0]+1,ast.parse("__import__('continuation').resume_boundary(locals())").body[0])
        return ast.fix_missing_locations(tree)
    resume_guard.transform_source=transform
    global _context
    _context=(config,control,selected)

def resume_boundary(scope):
    import torch,random,numpy as np
    from resume_guard import trees_equal
    config,control,selected=_context
    step=int(scope['global_step']);extra=scope['state']['extra_state']
    if not scope.get('loaded') or scope['candidates']!=[str(selected)] or Path(scope['load_checkpoint_path'])!=selected:
        raise RuntimeError('Exact approved full checkpoint was not restored')
    if step!=int(selected.name.removeprefix('global_step_')) or not 10000<=step<20000:
        raise RuntimeError('Unexpected resume step')
    if not scope['args'].train.resume_dataloader_state or scope['start_step']==0:
        raise RuntimeError('Native mid-epoch DataLoader restoration is required')
    scheduler=scope['lr_scheduler']
    if scheduler.last_epoch!=step or not scheduler._restored:raise RuntimeError('Scheduler did not restore')
    if step>10000 and not trees_equal(scheduler.state_dict(),extra['lr_scheduler']):raise RuntimeError('Scheduler resume mismatch')
    for actual,saved in ((torch.get_rng_state(),extra['torch_rng_state']),
                         (torch.cuda.get_rng_state_all(),extra['cuda_random_states']),
                         (random.getstate(),extra['python_random_state']),(np.random.get_state(),extra['numpy_random_state'])):
        if not trees_equal(actual,saved):raise RuntimeError('RNG restoration mismatch')
    optimizer=scope['optimizer'];steps=[int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
    if not steps or min(steps)!=step or max(steps)!=step:raise RuntimeError('Adam step/moment restoration mismatch')
    atomic_json(Path(config['train']['output_dir'])/'continuation-resume-verified.json',
                {'passed':True,'source_checkpoint':str(selected),'restored_global_step':step,
                 'target_step':20000,'adam_state_groups':len(steps),'rng_restored':True,
                 'native_dataloader_restored':True,'scheduler_migrated':step==10000,
                 'next_learning_rates':scheduler.get_last_lr(),'launch_attempt':os.environ['LINGBOT_TRAIN_ATTEMPT']})

def verify_completion(run,config,control,attempt):
    import torch
    checkpoint=Path(run)/'checkpoints/global_step_20000'
    record=validate(checkpoint.parent,checkpoint,signature(config))
    extra=torch.load(checkpoint/'extra_state/extra_state_rank_0.pt',map_location='cpu',weights_only=False)
    if extra['global_step']!=20000 or extra['lr_scheduler']['last_epoch']!=20000:
        raise RuntimeError('Final checkpoint/update count mismatch')
    if extra['lr_scheduler']['kind']!='clean50-continuation-cosine-v1':raise RuntimeError('Final LR recipe mismatch')
    rows=[json.loads(s) for s in (Path(run)/'logs/updates.jsonl').read_text().splitlines() if s.strip()]
    boundary=json.loads((Path(run)/'continuation-resume-verified.json').read_text())
    actual=[r['optimizer_step'] for r in rows if r.get('launch_attempt')==attempt]
    if actual!=list(range(boundary['restored_global_step']+1,20001)):
        raise RuntimeError('Final update ledger has gaps or duplicate updates')
    return {'state':'segment_complete','stopped_at_optimizer_step':20000,'new_optimizer_updates':len(actual),
            'checkpoint':str(checkpoint),'checkpoint_bytes':sum(record['files'].values()),
            'automatic_export':False,'evaluation_started':False,'automatic_restart':False}
