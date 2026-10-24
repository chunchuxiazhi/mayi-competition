"""Small provenance gates. Never imports torch or loads a model."""
import hashlib
import json
from pathlib import Path
from common import run_path

TRAIN_SIGNATURE='ffcebfebe757c20c1ac64015eadeb7142f0398b34e6276ab0d953081cd625f33'

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_source(config,full_hash=False):
    root=Path(config['project_root']).resolve();run=run_path(config).resolve()
    source_path=run/'model-source.json'
    record=json.loads(source_path.read_text())
    hf=(root/config['model_relative']).resolve()
    expected=root/'runs/finetune-clean-v3-gb10/clean50-10000-fixed1000/exports/global_step_10000/hf_ckpt'
    if hf!=expected or (root/config['model_wrapper_relative']).resolve()!=hf:raise ValueError('Unexpected step10000 model path')
    if record['training_step']!=10000 or record['hf_path']!=str(hf):raise ValueError('Wrong training checkpoint')
    train=hf.parent.parent.parent
    if digest(train/'frozen-config.yaml')!=TRAIN_SIGNATURE:raise ValueError('Training config changed')
    for entry in record['small_files']:
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Source evidence changed: '+entry['path'])
    manifest=json.loads((hf/'export-manifest.json').read_text())
    # The trainer hashes canonical JSON, while TRAIN_SIGNATURE pins the saved file bytes.
    # These are intentionally separate checks and must not be compared to one another.
    frozen=json.loads((train/'frozen-config.yaml').read_text())
    canonical_signature=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    if manifest['configuration_signature']!=canonical_signature:raise ValueError('Export config signature mismatch')
    if record['training_configuration_signature']!=canonical_signature:raise ValueError('Source config signature mismatch')
    if Path(manifest['checkpoint'])!=train/'checkpoints/global_step_10000':raise ValueError('Export source mismatch')
    for name,entry in record['shards'].items():
        p=hf/name;st=p.stat()
        if st.st_size!=entry['bytes'] or st.st_mtime_ns!=entry['mtime_ns']:raise ValueError('Weight shard changed: '+name)
        if entry['sha256']!=manifest['weights'][name]['sha256']:raise ValueError('Shard evidence mismatch')
        if full_hash and digest(p)!=entry['sha256']:raise ValueError('Weight checksum mismatch: '+name)
    if set(record['shards'])!={p.name for p in hf.glob('*.safetensors')}:raise ValueError('Shard set changed')
    reload=json.loads((train/'reload-verification.json').read_text())
    if not reload['passed'] or not reload['strict_deployment_load'] or reload['checkpoint']!=str(hf):raise ValueError('Strict reload verification absent')
    return digest(source_path)

def expected_episode(config,task,index,scene_seeds,instruction):
    plan=json.loads((run_path(config)/'episodes-plan.json').read_text())
    entry=plan['episodes'][config['tasks'].index(task)*config['test_num']+index]
    if entry['task']!=task or entry['episode_index']!=index:raise ValueError('Episode plan mismatch')
    if any(entry[k]!=v for k,v in scene_seeds.items()):raise ValueError('Episode seed drift')
    prior=entry.get('baseline_instruction')
    if prior is not None and prior!=instruction:raise ValueError('Instruction differs from paired baseline; review before running')

def compare(config):
    from common import read_events,atomic_json
    plan=json.loads((run_path(config)/'episodes-plan.json').read_text())
    current={}
    for task in config['tasks']:
        for e in read_events(run_path(config)/'tasks'/task/'episodes.jsonl'):
            if e['event']=='policy_completed':
                key=(task,e['episode_index'])
                if key in current:raise ValueError('Duplicate comparison outcome')
                current[key]=e
    paired=[];extra=[];mismatch=[]
    for p in plan['episodes']:
        e=current.get((p['task'],p['episode_index']))
        if e is None:continue
        if p.get('baseline_instruction') is None:extra.append((p['task'],p['episode_index']));continue
        if any(e[k]!=p[k] for k in ('seed','policy_seed','instruction_seed')) or e['instruction']!=p['baseline_instruction']:
            mismatch.append((p['task'],p['episode_index']));continue
        paired.append((bool(p['baseline_success']),bool(e['success'])))
    out={'matched_valid_episodes':len(paired),'baseline_successes_on_matched':sum(a for a,b in paired),
         'trained_successes_on_matched':sum(b for a,b in paired),'newly_valid_slots_without_baseline':extra,
         'input_mismatches':mismatch,'baseline_valid_episodes':plan['baseline_valid_episodes'],
         'missing_valid_trained_slots':len(plan['episodes'])-len(current),
         'note':'Compare only identical scene/policy/instruction seeds and actual instructions; expert exceptions are separate.'}
    atomic_json(run_path(config)/'comparison.json',out)
    return out
