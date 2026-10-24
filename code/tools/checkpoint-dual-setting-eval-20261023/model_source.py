"""Pin the single step20000 inference export; no model imports."""
import hashlib,json
from pathlib import Path
from common import run_path
from validation import digest

def verify_source(config,full_hash=False):
    root=Path(config['project_root']).resolve();hf=(root/config['model_relative']).resolve();train=root/config['training_relative']
    expected=train/'exports/global_step_20000/hf_ckpt'
    source=root/config['model_source_relative'];record=json.loads(source.read_text())
    if hf!=expected or (root/config['model_wrapper_relative']).resolve()!=hf:raise ValueError('Unexpected step20000 model path')
    if record['training_step']!=20000 or record['hf_path']!=str(hf):raise ValueError('Wrong exported checkpoint')
    if record['configuration_signature']!=config['training_configuration_signature']:raise ValueError('Wrong training recipe')
    for entry in record['small_files']:
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Model source evidence changed: '+entry['path'])
    manifest=json.loads((hf/'export-manifest.json').read_text())
    if manifest['configuration_signature']!=config['training_configuration_signature'] or Path(manifest['checkpoint'])!=train/'checkpoints/global_step_20000':raise ValueError('Export came from another checkpoint')
    if set(record['shards'])!={p.name for p in hf.glob('*.safetensors')}:raise ValueError('Model shard set mismatch')
    for name,entry in record['shards'].items():
        path=hf/name;st=path.stat()
        if st.st_size!=entry['bytes'] or st.st_mtime_ns!=entry['mtime_ns'] or entry['sha256']!=manifest['weights'][name]['sha256']:
            raise ValueError('Model shard changed: '+name)
        if full_hash and digest(path)!=entry['sha256']:raise ValueError('Model shard checksum mismatch: '+name)
    reload=json.loads((train/'reload-verification.json').read_text())
    if not reload['passed'] or not reload['strict_deployment_load'] or not reload['action_projection_exact_match'] or reload['checkpoint']!=str(hf):raise ValueError('Strict deployment load evidence is absent')
    return digest(source)

def expected_episode(config,task,index,scene_seeds,instruction):
    plan=json.loads((run_path(config)/'episodes-plan.json').read_text())
    entry=plan['episodes'][config['tasks'].index(task)*config['test_num']+index]
    if entry['task']!=task or entry['episode_index']!=index or entry['setting']!=config['task_config']:raise ValueError('Episode plan mismatch')
    if any(entry[k]!=v for k,v in scene_seeds.items()):raise ValueError('Episode seed drift')
    if entry.get('baseline_instruction') is not None and entry['baseline_instruction']!=instruction:
        raise ValueError('Clean instruction differs from the matched baseline; preserve and review')

def compare(config):
    from common import read_events,atomic_json
    plan=json.loads((run_path(config)/'episodes-plan.json').read_text());current={}
    for task in config['tasks']:
        for e in read_events(run_path(config)/'tasks'/task/'episodes.jsonl'):
            if e['event']=='policy_completed':
                key=(task,e['episode_index'])
                if key in current:raise ValueError('Duplicate outcome')
                if e['setting']!=config['task_config']:raise ValueError('Mixed setting outcome')
                current[key]=e
    paired=[];extra=[];mismatch=[]
    for p in plan['episodes']:
        e=current.get((p['task'],p['episode_index']))
        if e is None:continue
        if p.get('baseline_instruction') is None:extra.append((p['task'],p['episode_index']));continue
        if any(e[k]!=p[k] for k in ('seed','policy_seed','instruction_seed')) or e['instruction']!=p['baseline_instruction']:
            mismatch.append((p['task'],p['episode_index']));continue
        paired.append((bool(p['baseline_success']),bool(e['success'])))
    result={'setting':config['task_config'],'matched_valid_episodes':len(paired),'baseline_successes_on_matched':sum(a for a,b in paired),
            'trained_successes_on_matched':sum(b for a,b in paired),'valid_slots_without_baseline':extra,'input_mismatches':mismatch,
            'missing_valid_policy_slots':len(plan['episodes'])-len(current),'baseline_valid_episodes':plan['baseline_valid_episodes'],
            'note':'Only clean has a paired step10000 baseline; randomized is a separate scene distribution, not paired clean.'}
    atomic_json(run_path(config)/'comparison.json',result);return result
