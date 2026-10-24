"""Verify submitted HF files directly; no dependency on training DCP or mtime."""
import hashlib,json
from pathlib import Path
from common import run_path,seeds,atomic_json

def verify_source(config,full_hash=False):
    model=Path(config['model_relative']).resolve()
    manifest_path=Path(config['model_manifest'])
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['step']!=20000 or config['training_step']!=20000:raise ValueError('Wrong checkpoint step')
    index=json.loads((model/'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values())!={r['path'] for r in manifest['files']}:raise ValueError('Wrong shard index')
    for row in manifest['files']:
        path=model/row['path']
        if path.stat().st_size!=row['bytes']:raise ValueError('Model shard size mismatch: '+row['path'])
        if full_hash:
            with path.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=row['sha256']:raise ValueError('Model shard checksum mismatch')
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()

def expected_episode(config,task,index,scene_seeds,instruction):
    plan=json.loads((run_path(config)/'episodes-plan.json').read_text())
    entry=plan['episodes'][config['tasks'].index(task)*config['test_num']+index]
    if entry['task']!=task or entry['episode_index']!=index or entry['instruction_type']!=config['instruction_type']:raise ValueError('Episode identity mismatch')
    if any(entry[k]!=v for k,v in scene_seeds.items()):raise ValueError('Seed drift')
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Missing actual instruction')

def compare(config):
    # Fresh reproduction has no dependency on any historical baseline run.
    atomic_json(run_path(config)/'reproduction-source.json',{'source_mode':'submitted_hf_sha256',
        'training_step':20000,'manifest_sha256':verify_source(config),'paired_baseline_required':False})
