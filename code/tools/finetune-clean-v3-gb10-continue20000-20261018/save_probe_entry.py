"""Isolated real-model save/continue probe. Never used by formal training."""
from pathlib import Path
import ast
import hashlib
import inspect
import json
import os
import sys


def main():
    import torch
    import torch.distributed as dist
    import yaml
    from storage_policy import atomic_json, checked_checkpoint, validate
    import gb10_runtime
    import resume_guard
    probe = Path(os.environ['LINGBOT_SAVE_PROBE_RUN']).resolve()
    project = Path('/home/zzj/competitions/lingbot')
    if probe != project/'runs/finetune-clean-v3-gb10/save-continuation-probe-20261016':
        raise RuntimeError('Unreviewed probe output')
    mode = os.environ['LINGBOT_SAVE_PROBE_MODE']
    if mode not in ('continue', 'reload'):
        raise ValueError(mode)
    atomic_json(probe/'child-pid.json', {'pid':os.getpid(),'mode':mode})
    original_run = project/'runs/finetune-clean-v3-gb10/pilot'
    config_path = original_run/'frozen-config.yaml'
    config = yaml.safe_load(config_path.read_text())
    control = json.loads((probe/'tools/pilot-control.json').read_text())
    sys.path.insert(0,str(project/'repos/lingbot-vla-v2'))
    import tasks.vla.train_lingbotvla as official
    if hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest() != control['upstream_trainer_sha256']:
        raise RuntimeError('Official trainer changed')
    gb10_runtime.install(official, config, control)
    parse = official.parse_args
    source = original_run/'checkpoints/global_step_20' if mode=='continue' else probe/'checkpoints/global_step_24'
    def parse_for_probe(rootclass):
        args = parse(rootclass)
        # The original model, optimizer, batch settings and LR scheduler are
        # constructed with the original 20-step pilot configuration.
        args.train.output_dir = str(probe)
        args.train.save_checkpoint_path = str(probe/'checkpoints')
        args.train.model_assets_dir = str(probe/'model_assets')
        args.train.load_checkpoint_path = str(source)
        return args
    official.parse_args = parse_for_probe
    expected = gb10_runtime.signature(config)
    records = []

    def tensor_hash(value):
        value = value.detach()
        if hasattr(value, 'to_local'):
            value = value.to_local()
        return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()

    def model_evidence(model,optimizer):
        parameter = next(p for n,p in model.named_parameters() if n=='model.action_out_proj.weight')
        state = optimizer.state[parameter]
        return {'projection_sha256':tensor_hash(parameter),
                'adam_exp_avg_sha256':tensor_hash(state['exp_avg']),
                'adam_exp_avg_sq_sha256':tensor_hash(state['exp_avg_sq']),
                'adam_step':int(state['step'].item())}

    factory = official.build_checkpointer
    def probe_factory(*args,**kwargs):
        cls = factory(*args,**kwargs)
        save = cls.save
        @classmethod
        def probe_save(inner,path,training_state,global_steps=None,save_async=False):
            if mode!='continue' or global_steps not in (22,24) or Path(path).resolve()!=probe/'checkpoints':
                raise RuntimeError('Unexpected probe save')
            # This destructive retention is test-only: original step20 remains
            # untouched and serves as recovery. Never used for formal retention.
            if global_steps==24:
                import shutil
                prior=checked_checkpoint(probe/'checkpoints',probe/'checkpoints/global_step_22')
                manifest=validate(probe/'checkpoints',prior,expected)
                updates=[json.loads(x)['optimizer_step'] for x in (probe/'logs/updates.jsonl').read_text().splitlines()]
                if not all(s in updates for s in (23,24)):
                    raise RuntimeError('No completed training after first save')
                atomic_json(probe/'first-checkpoint-before-removal.json',manifest)
                shutil.rmtree(prior)
            evidence=model_evidence(training_state['model'],training_state['optimizer'])
            save(path,training_state,global_steps=global_steps,save_async=save_async)
            manifest=validate(probe/'checkpoints',probe/f'checkpoints/global_step_{global_steps}',expected)
            record={'step':global_steps,'bytes':sum(manifest['files'].values()),'model_optimizer_evidence':evidence,
                    'scheduler':training_state['extra_state']['lr_scheduler']}
            records.append(record)
            atomic_json(probe/'save-records.json',records)
        cls.save=probe_save
        return cls
    official.build_checkpointer=probe_factory

    def boundary(scope):
        import random,numpy as np
        target=20 if mode=='continue' else 24
        if not scope.get('loaded') or int(scope['global_step'])!=target:
            raise RuntimeError('Expected full restoration of step '+str(target))
        extra=scope['state']['extra_state']
        if not resume_guard.trees_equal(scope['lr_scheduler'].state_dict(),extra['lr_scheduler']):
            raise RuntimeError('Scheduler restoration mismatch')
        steps=[int(v['step'].item()) for v in scope['optimizer'].state.values() if 'step' in v]
        if not steps or min(steps)!=target or max(steps)!=target:
            raise RuntimeError('Adam states not fully restored')
        if not torch.equal(torch.get_rng_state(),extra['torch_rng_state']):
            raise RuntimeError('CPU RNG not restored')
        if not resume_guard.trees_equal(torch.cuda.get_rng_state_all(),extra['cuda_random_states']):
            raise RuntimeError('CUDA RNG not restored')
        if not resume_guard.trees_equal(random.getstate(),extra['python_random_state']) or not resume_guard.trees_equal(np.random.get_state(),extra['numpy_random_state']):
            raise RuntimeError('Python/NumPy RNG not restored')
        if mode=='reload':
            expected_record=json.loads((probe/'save-records.json').read_text())[-1]
            actual=model_evidence(scope['model'],scope['optimizer'])
            if actual!=expected_record['model_optimizer_evidence'] or not resume_guard.trees_equal(extra['lr_scheduler'],expected_record['scheduler']):
                raise RuntimeError('Saved model/Adam/scheduler differs after full reload')
            atomic_json(probe/'reload-result.json',{'passed':True,'restored_step':target,'optimizer_groups':len(steps),
                'projection_and_adam_moments_exact_hash_match':True,'scheduler_and_rng_restored':True,'updates_executed':0})
            scope['writer'].close();dist.barrier();dist.destroy_process_group()
            raise SystemExit(0)
        # The original scheduler has already been constructed and restored.
        # Extend only the isolated test loop, without rebuilding its LR schedule.
        args=scope['args']
        original={'max_steps':args.train.max_steps,'save_steps':args.train.save_steps,'train_steps':args.train.train_steps}
        args.train.max_steps=26;args.train.save_steps=2
        if args.train.train_steps<=25:
            raise RuntimeError('Probe requires native epoch length beyond step25')
        scope['train_dataloader'].load_state_dict(extra['train_dataloader'])
        torch.set_rng_state(extra['torch_rng_state']);torch.cuda.set_rng_state_all(extra['cuda_random_states'])
        random.setstate(extra['python_random_state']);np.random.set_state(extra['numpy_random_state'])
        atomic_json(probe/'resume-result.json',{'passed':True,'restored_step':target,'optimizer_groups':len(steps),
            'original_pilot_loop':original,'test_saves':[22,24],'test_exit_after_update':25,
            'scheduler_constructed_from_original_pilot':True,'config_signature':expected,
            'formal_config_or_checkpoint_modified':False})
        return target//args.train.train_steps,target%args.train.train_steps

    def after_update(scope):
        step=int(scope['global_step'])
        if step>25:
            raise RuntimeError('Probe update budget exceeded')
        if step!=25:
            return
        torch.cuda.synchronize()
        rows=[json.loads(x) for x in (probe/'logs/updates.jsonl').read_text().splitlines()]
        if [r['optimizer_step'] for r in rows]!=[21,22,23,24,25] or [r['step'] for r in records]!=[22,24]:
            raise RuntimeError('Save/continue sequence incomplete')
        if any(r['micro_batches']!=4 for r in rows):
            raise RuntimeError('Unexpected pilot batch accumulation')
        atomic_json(probe/'continuation-result.json',{'passed':True,'source_step':20,'completed_steps':[21,22,23,24,25],
            'successful_saves':[22,24],'completed_update_after_second_save':25,
            'final_scheduler_step':scope['lr_scheduler'].last_epoch,'formal_training_resumed':False})
        scope['writer'].close();dist.barrier();dist.destroy_process_group()
        raise SystemExit(0)

    tree=resume_guard.transform_source(inspect.getsource(official.main))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    boundaries=0;updates=0
    for i,node in enumerate(function.body):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='__gb10_resume_boundary':
            function.body[i]=ast.parse('start_epoch, start_step = __gb10_resume_boundary(locals())').body[0];boundaries+=1
    for node in ast.walk(function):
        if not isinstance(node,ast.For):
            continue
        for i,statement in enumerate(list(node.body)):
            if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Attribute) and isinstance(statement.value.func.value,ast.Name) and statement.value.func.value.id=='optimizer' and statement.value.func.attr=='zero_grad':
                node.body.insert(i+1,ast.parse('__save_probe_after_update(locals())').body[0]);updates+=1
    if (boundaries,updates)!=(1,1):
        raise RuntimeError('Unexpected trainer structure for probe hooks')
    ast.fix_missing_locations(tree)
    namespace=official.__dict__.copy();namespace.update(__gb10_resume_boundary=boundary,__save_probe_after_update=after_update)
    exec(compile(tree,official.__file__,'exec'),namespace)
    sys.argv=[official.__file__,str(config_path)]
    namespace['main']()


if __name__=='__main__':
    main()
