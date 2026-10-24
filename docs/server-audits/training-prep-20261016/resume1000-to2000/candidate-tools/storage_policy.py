"""Small, testable checkpoint policy; deletions only within this run and after a committed save."""
from pathlib import Path
import hashlib,json,os,re,shutil
import math,statistics

GIB=1024**3

def planned_saves(max_steps,save_steps):
    if type(max_steps) is not int or type(save_steps) is not int or min(max_steps,save_steps)<1:
        raise ValueError('Update/save intervals must be positive integers')
    points=list(range(save_steps,max_steps+1,save_steps))
    if not points or points[-1]!=max_steps:points.append(max_steps)
    return points

def resolve_schedule(config,control,pilot_rows=None,pilot_batch=None):
    mode=control.get('checkpoint_schedule_mode','pilot_calibrated' if control['purpose']=='joint_50_tasks' else 'fixed')
    train=config['train'];points=planned_saves(train['max_steps'],train['save_steps'])
    if mode=='fixed':
        return {'mode':'fixed','save_every_updates':train['save_steps'],'planned_saves':points}
    if mode!='pilot_calibrated':raise ValueError('Unknown checkpoint schedule mode')
    result=checkpoint_schedule(pilot_rows or [],pilot_batch,train['global_batch_size'],maximum=train['save_steps'])
    return {'mode':mode,**result,'planned_saves':planned_saves(train['max_steps'],result['save_every_updates'])}

def checkpoint_schedule(pilot_rows,pilot_batch,training_batch,maximum=250,target_seconds=1800):
    durations=[float(e['seconds_since_previous_update']) for e in pilot_rows if int(e['optimizer_step'])>1]
    if len(durations)<5 or any(not math.isfinite(v) or v<=0 for v in durations):
        raise ValueError('Insufficient valid pilot timing; do not guess a long-run checkpoint interval')
    median=statistics.median(durations)
    estimate=median*training_batch/pilot_batch
    interval=min(maximum,max(1,int(target_seconds/estimate)))
    if interval>=10:interval=interval//10*10
    return {'pilot_median_seconds':median,'estimated_training_update_seconds':estimate,
        'pilot_effective_batch':pilot_batch,'training_effective_batch':training_batch,
        'target_seconds':target_seconds,'maximum_interval':maximum,'save_every_updates':interval}

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def directory_bytes(path):
    seen=set();total=0
    for p in Path(path).rglob('*'):
        if p.is_symlink():continue
        if p.is_file():
            stat=p.stat();key=(stat.st_dev,stat.st_ino)
            if key not in seen:total+=stat.st_blocks*512;seen.add(key)
    return total

def checked_checkpoint(checkpoints,path):
    checkpoints=Path(checkpoints).resolve();path=Path(path)
    if path.is_symlink() or path.resolve().parent!=checkpoints or not re.fullmatch(r'global_step_[1-9][0-9]*',path.name):
        raise ValueError('Checkpoint path outside the reviewed output root')
    return path.resolve()

def commit(checkpoints,path,signature):
    path=checked_checkpoint(checkpoints,path)
    required=['model/.metadata','optimizer/.metadata','extra_state/extra_state_rank_0.pt']
    if not all((path/n).is_file() for n in required):raise RuntimeError('Incomplete DCP components')
    for group in ('model','optimizer'):
        if not any((path/group).glob('*.distcp')):raise RuntimeError('DCP tensor shards missing')
    files={str(p.relative_to(path)):p.stat().st_size for p in path.rglob('*') if p.is_file() and p.name!='complete.json'}
    digests={n:hashlib.sha256((path/n).read_bytes()).hexdigest() for n in required}
    atomic_json(path/'complete.json',{'signature':signature,'files':files,'metadata_sha256':digests})

def validate(checkpoints,path,signature):
    path=checked_checkpoint(checkpoints,path)
    record=json.loads((path/'complete.json').read_text())
    if record['signature']!=signature:raise ValueError('Checkpoint/configuration signature mismatch')
    for name,size in record['files'].items():
        p=path/name
        if not p.resolve().is_relative_to(path) or p.is_symlink() or not p.is_file() or p.stat().st_size!=size:
            raise RuntimeError('Checkpoint component changed: '+name)
    for name,digest in record['metadata_sha256'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Checkpoint metadata corrupted: '+name)
    return record

def prune(checkpoints,signature,keep):
    checkpoints=Path(checkpoints);complete=[]
    if keep<1:raise ValueError('At least one recovery point must be retained')
    for p in checkpoints.glob('global_step_*'):
        if (p/'complete.json').is_file():
            validate(checkpoints,p,signature);complete.append(p)
    complete.sort(key=lambda p:int(p.name.rsplit('_',1)[1]),reverse=True)
    removed=[]
    for path in complete[keep:]:
        target=checked_checkpoint(checkpoints,path)
        # Validate the absolute deletion target again immediately before removal.
        validate(checkpoints,target,signature)
        shutil.rmtree(target);removed.append(str(target))
    return removed

def check_space(control):
    training_root=Path(control['training_root']);training_root.mkdir(parents=True,exist_ok=True)
    reserve=int(control['checkpoint_reserve_gib']*GIB)
    used=directory_bytes(training_root)
    if used+reserve>control['training_storage_limit_gib']*GIB:
        raise RuntimeError('Training storage budget would be exceeded by another checkpoint')
    if shutil.disk_usage(training_root).free<reserve+control['disk_free_floor_gib']*GIB:
        raise RuntimeError('Insufficient disk reserve for a safe checkpoint')
