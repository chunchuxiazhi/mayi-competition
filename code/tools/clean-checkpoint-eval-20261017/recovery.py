"""Reviewed recovery decisions; no simulator or model imports."""
import re
from pathlib import Path
from common import fingerprint

class TaskIncomplete(RuntimeError):
    """Known scene/metadata failure: retain slot and continue other tasks."""

class TaskSliceCompleted(Exception):
    """Intentional smoke slice; prevent the legacy full-task score export."""

def global_resource_fault(exc):
    text=str(exc).lower()
    return isinstance(exc,(MemoryError,ImportError,SystemExit)) or any(s in text for s in
        ('out of memory','cuda error','cuda runtime','device lost','device-side assert','no space left on device'))

def validate_revision(original, current):
    if current == original:
        return
    allowed={'expert_failure_policy','execution_approval','control_revision'}
    if {k:v for k,v in original.items() if k not in allowed} != {k:v for k,v in current.items() if k not in allowed}:
        raise RuntimeError('Revision changed protected experimental configuration')
    revision=current.get('control_revision',{})
    if (revision.get('parent_config_sha256')!=fingerprint(original)
        or revision.get('version')!=2
        or current.get('expert_failure_policy')!='validate_metadata_then_policy_same_seed'):
        raise RuntimeError('Unrecognized control revision')

def recorded_expert(events, base):
    rows=[e for e in events if e['event'] in ('expert_passed','expert_failed')]
    if not rows:return None
    if len(rows)!=1:raise RuntimeError('Ambiguous completed expert prechecks; review ledger')
    row=rows[0]
    for k in ('task','episode_index','seed','setting','instruction_type'):
        if row.get(k)!=base[k]:raise RuntimeError('Expert provenance mismatch: '+k)
    if type(row.get('expert_success')) is not bool:raise RuntimeError('Expert outcome missing')
    return row

def validate_metadata(info, repo):
    if not isinstance(info,dict):raise TaskIncomplete('Expert returned no parameter dictionary')
    objects=(Path(repo)/'description/objects_description').resolve()
    for key,value in info.items():
        if not isinstance(key,str) or not re.fullmatch(r'\{[^{}]+\}',key):
            raise TaskIncomplete('Invalid parameter key: '+repr(key))
        if not isinstance(value,str) or not value.strip():raise TaskIncomplete('Missing parameter: '+key)
        name=key[1:-1]
        if len(name)==1 and name.islower() and value not in ('left','right','both','dual'):
            raise TaskIncomplete('Invalid arm parameter: '+key)
        if '/' in value or '\\' in value:
            target=(objects/(value+'.json')).resolve()
            if not target.is_relative_to(objects) or not target.is_file():
                raise TaskIncomplete('Missing object description: '+value)
            import json
            descriptions=json.loads(target.read_text())
            if not descriptions.get('seen') or not all(isinstance(s,str) and s.strip() for s in descriptions['seen']):
                raise TaskIncomplete('Invalid seen object descriptions: '+value)

def generate_seen(task,info,seed,candidates,repo):
    # The official generator and ordering are preserved. No fallback text is invented.
    import random
    import numpy as np
    from generate_episode_instructions import generate_episode_descriptions
    validate_metadata(info,repo)
    py_state=random.getstate();np_state=np.random.get_state()
    try:
        random.seed(seed);np.random.seed(seed)
        try:descriptions=generate_episode_descriptions(task,[info],candidates)
        except (Exception,SystemExit) as exc:raise TaskIncomplete('Official instruction generation failed: '+str(exc)) from exc
        if len(descriptions)!=1 or not descriptions[0].get('seen'):
            raise TaskIncomplete('No official seen instructions for returned parameters')
        choices=descriptions[0]['seen']
        if any(not isinstance(s,str) or not s.strip() or re.search(r'\{[^{}]+\}',s) for s in choices):
            raise TaskIncomplete('Official seen instructions contain missing parameters')
        return str(np.random.choice(choices))
    finally:
        random.setstate(py_state);np.random.set_state(np_state)
