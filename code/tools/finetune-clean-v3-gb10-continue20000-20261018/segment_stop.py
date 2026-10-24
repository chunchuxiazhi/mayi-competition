"""Stop a resumed segment after a native save, without changing its LR plan."""
import ast
import json
import os
from pathlib import Path
from storage_policy import atomic_json,validate


def validate_budget(config,stop_at,resume_at):
    if type(stop_at) is not int or type(resume_at) is not int:
        raise ValueError('Both segment boundaries must be integers')
    train=config['train']
    if not 0<resume_at<stop_at<=train['max_steps']:
        raise ValueError('Invalid resumed segment boundaries')
    if stop_at%train['save_steps'] or resume_at%train['save_steps']:
        raise ValueError('Segment boundaries must use native scheduled checkpoints')
    return {'expected_resume_step':resume_at,'stop_after_committed_step':stop_at,
            'original_max_steps':train['max_steps'],'original_save_steps':train['save_steps'],
            'original_lr_schedule_preserved':True,'automatic_export':False}


def transform_tree(tree,stop_at):
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    boundaries=[i for i,n in enumerate(function.body) if isinstance(n,ast.Expr)
                and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                and n.value.func.id=='__gb10_resume_boundary']
    if len(boundaries)!=1:
        raise RuntimeError('Expected exactly one guarded resume boundary')
    function.body.insert(boundaries[0]+1,ast.parse("__import__('segment_stop').resume_boundary(locals())").body[0])
    candidates=[]
    expected=ast.dump(ast.parse('args.train.max_steps is not None and global_step >= args.train.max_steps',mode='eval').body)
    for node in ast.walk(function):
        if isinstance(node,ast.If) and ast.dump(node.test)==expected:
            candidates.append(node)
    if len(candidates)!=1:
        raise RuntimeError('Expected exactly one native max_steps termination check')
    node=candidates[0]
    node.test=ast.parse(f'args.train.max_steps is not None and global_step >= min(args.train.max_steps,{stop_at})',mode='eval').body
    # Replace only the native termination message; retain its break/flags and
    # the trainer's final native save/barrier/group cleanup logic.
    if not node.body or not isinstance(node.body[0],ast.Expr):
        raise RuntimeError('Unexpected termination body')
    node.body[0]=ast.parse("logger.info_rank0(f'Reached requested segment stop at optimizer step {global_step}; original max_steps and LR schedule unchanged.')").body[0]
    ast.fix_missing_locations(tree)
    return tree


def install(resume_guard,config,control):
    stop=os.environ.get('LINGBOT_SEGMENT_STOP_AT')
    start=os.environ.get('LINGBOT_EXPECT_RESUME_STEP')
    if stop is None and start is None:
        return
    if stop is None or start is None:
        raise ValueError('Both stop and expected-resume controls are required')
    budget=validate_budget(config,int(stop),int(start))
    original=resume_guard.transform_source
    resume_guard.transform_source=lambda source:transform_tree(original(source),int(stop))
    global _context
    _context=(config,control,budget)


def resume_boundary(scope):
    import torch,random,numpy as np
    from resume_guard import trees_equal
    config,control,budget=_context
    step=int(scope['global_step']);extra=scope['state']['extra_state']
    if not scope.get('loaded') or not scope['candidates'] or step!=budget['expected_resume_step']:
        raise RuntimeError('Segment requires the exact approved complete checkpoint')
    expected=Path(config['train']['output_dir'])/'checkpoints'/f'global_step_{step}'
    if Path(scope['load_checkpoint_path']).resolve()!=expected.resolve():
        raise RuntimeError('Unexpected resume checkpoint path')
    args=scope['args']
    if args.train.max_steps!=config['train']['max_steps'] or args.train.save_steps!=config['train']['save_steps']:
        raise RuntimeError('Original training/save schedule changed')
    if not args.train.resume_dataloader_state or scope['start_step']==0:
        raise RuntimeError('Expected native mid-epoch DataLoader restoration')
    if not trees_equal(scope['lr_scheduler'].state_dict(),extra['lr_scheduler']):
        raise RuntimeError('Scheduler restoration mismatch')
    if not torch.equal(torch.get_rng_state(),extra['torch_rng_state']):
        raise RuntimeError('CPU RNG restoration mismatch')
    for actual,saved in ((torch.cuda.get_rng_state_all(),extra['cuda_random_states']),
                         (random.getstate(),extra['python_random_state']),
                         (np.random.get_state(),extra['numpy_random_state'])):
        if not trees_equal(actual,saved):raise RuntimeError('RNG restoration mismatch')
    optimizer=scope['optimizer'];steps=[int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
    if not steps or min(steps)!=step or max(steps)!=step:
        raise RuntimeError('Adam states not fully restored to approved step')
    atomic_json(Path(config['train']['output_dir'])/'segment-resume-verified.json',
        {'passed':True,**budget,'restored_global_step':step,'adam_state_groups':len(steps),
         'scheduler_and_rng_restored':True,'native_dataloader_restore_enabled':True,
         'next_learning_rates':[g['lr'] for g in optimizer.param_groups],
         'launch_attempt':os.environ.get('LINGBOT_TRAIN_ATTEMPT')})


def verify_completion(run,config,stop_at,attempt):
    import torch
    from gb10_runtime import signature
    run=Path(run);checkpoint=run/'checkpoints'/f'global_step_{stop_at}'
    record=validate(run/'checkpoints',checkpoint,signature(config))
    extra=torch.load(checkpoint/'extra_state/extra_state_rank_0.pt',map_location='cpu',weights_only=False)
    if extra['global_step']!=stop_at or extra['lr_scheduler']['last_epoch']!=stop_at:
        raise RuntimeError('Final segment checkpoint step/scheduler mismatch')
    rows=[json.loads(s) for s in (run/'logs/updates.jsonl').read_text().splitlines() if s.strip()]
    boundary=json.loads((run/'segment-resume-verified.json').read_text())
    expected=list(range(boundary['restored_global_step']+1,stop_at+1))
    actual=[r['optimizer_step'] for r in rows if r.get('launch_attempt')==attempt]
    if actual!=expected or rows[-1]['optimizer_step']!=stop_at:
        raise RuntimeError('Segment update ledger mismatch')
    return {'state':'segment_complete','stopped_at_optimizer_step':stop_at,
            'original_max_steps':config['train']['max_steps'],'original_lr_schedule_preserved':True,
            'checkpoint':str(checkpoint),'checkpoint_bytes':sum(record['files'].values()),
            'new_optimizer_updates':len(actual),'automatic_export':False,'evaluation_started':False,'automatic_restart':False}
