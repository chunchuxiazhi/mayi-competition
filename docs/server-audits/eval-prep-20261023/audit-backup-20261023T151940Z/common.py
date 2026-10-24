"""Metadata and append-only accounting. No model or simulator imports."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent
LEDGER_LIMIT_BYTES = 2*1024**2

def load_config(path=None):
    p = Path(path) if path else HERE/'config-clean.json'
    return json.loads(p.read_text(encoding='utf-8'))

def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def run_path(config):
    return Path(config['project_root'])/config['run_relative']

def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def read_events(path):
    if not Path(path).exists():return []
    events=[]
    with Path(path).open(encoding='utf-8') as f:
        for line_number,line in enumerate(f,1):
            if not line.strip():continue
            try: events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Ledger damaged at {path}:{line_number}; preserve and review, do not skip') from exc
    return events

def append_event(path,event):
    # Exactly one simulator process holds the task lock and writes this ledger.
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    event={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**event}
    line=json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n'
    if (path.stat().st_size if path.exists() else 0)+len(line.encode('utf-8'))>LEDGER_LIMIT_BYTES:
        raise RuntimeError('Structured ledger storage cap reached; preserve existing records')
    with path.open('a',encoding='utf-8') as f:
        f.write(line);f.flush();os.fsync(f.fileno())
    return event

def seeds(config,task_index,episode_index):
    offset=task_index*1000+episode_index
    return {'seed':config['seed_base']+episode_index,
            'policy_seed':config['policy_seed_base']+offset,
            'instruction_seed':config['instruction_seed_base']+offset}

def summarize(config,task):
    directory=run_path(config)/'tasks'/task
    events=read_events(directory/'episodes.jsonl')
    terminals={};started={};exceptions=[];interruptions=[]
    for e in events:
        if e.get('task')!=task or e.get('setting')!=config['task_config'] or type(e.get('episode_index')) is not int or e['episode_index'] not in range(config['test_num']):
            raise ValueError('Cross-setting, task or episode-index contamination in ledger')
    for e in events:
        kind=e['event'];index=e['episode_index']
        if kind=='policy_started':started[e['attempt_id']]=e
        if kind in ('policy_completed','policy_exception','policy_interrupted'):
            if kind=='policy_interrupted':interruptions.append(e)
            else:
                if index in terminals:raise RuntimeError(f'Duplicate terminal episode: {task}/{index}')
                terminals[index]=e
            started.pop(e['attempt_id'],None)
        if kind=='policy_exception':exceptions.append(e)
    valid=[e for e in terminals.values() if e['event']=='policy_completed']
    if any(type(e['success']) is not bool for e in valid):raise ValueError('Invalid success field')
    experts=read_events(directory/'expert.jsonl')
    successes=sum(e['success'] for e in valid)
    result={'label':config['label'],'task':task,'setting':config['task_config'],'instruction_type':config['instruction_type'],
            'planned_policy_episodes':config['test_num'],'completed_valid_policy_episodes':len(valid),
            'successes':successes,'success_rate':successes/len(valid) if valid else None,
            'policy_exceptions':len(exceptions),'interrupted_attempts':len(interruptions),
            'cleanup_exceptions':sum(e['event']=='cleanup_exception' for e in events),
            'historically_blocked_episode_slots':len({e['episode_index'] for e in events if e['event']=='episode_blocked'}),
            'expert_blocked_episode_slots':len({e['episode_index'] for e in events if e['event']=='episode_blocked'}-set(terminals)),
            'policy_attempts_initiated':sum(e['event']=='policy_started' for e in events),
            'not_started_policy_slots':config['test_num']-len({e['episode_index'] for e in events if e['event']=='policy_started'}),
            'open_policy_attempts':len(started),'terminal_episode_slots':len(terminals),
            'not_validly_completed_slots':config['test_num']-len(valid),
            'expert_attempts_initiated':sum(e['event']=='expert_started' for e in experts),
            'expert_failed_or_exception':sum(e['event'] in ('expert_failed','expert_exception') for e in experts),
            'complete':len(valid)==config['test_num'],'standard_100_episode_score':False,
            'overall_competition_score':None}
    atomic_json(directory/'summary.json',result)
    return result

def overall(config):
    rows=[]
    for task in config['tasks']:
        p=run_path(config)/'tasks'/task/'summary.json'
        rows.append(json.loads(p.read_text()) if p.exists() else {
            'task':task,'planned_policy_episodes':config['test_num'],'completed_valid_policy_episodes':0,
            'successes':0,'success_rate':None,'complete':False,'policy_exceptions':0,
            'interrupted_attempts':0,'cleanup_exceptions':0,'expert_blocked_episode_slots':0,
            'open_policy_attempts':0,'expert_attempts_initiated':0})
    valid=sum(r['completed_valid_policy_episodes'] for r in rows)
    successes=sum(r['successes'] for r in rows)
    data={'label':config['label'],'setting':config['task_config'],'instruction_type':config['instruction_type'],'training_step':config['training_step'],'planned_tasks':len(rows),'planned_policy_episodes':len(rows)*config['test_num'],
          'completed_valid_policy_episodes':valid,'successes':successes,
          'success_rate':successes/valid if valid else None,'complete':all(r['complete'] for r in rows),
          'policy_exceptions':sum(r['policy_exceptions'] for r in rows),
          'cleanup_exceptions':sum(r['cleanup_exceptions'] for r in rows),
          'expert_blocked_episode_slots':sum(r['expert_blocked_episode_slots'] for r in rows),
          'interrupted_attempts':sum(r['interrupted_attempts'] for r in rows),
          'open_policy_attempts':sum(r['open_policy_attempts'] for r in rows),
          'expert_attempts_initiated':sum(r['expert_attempts_initiated'] for r in rows),
          'per_task':rows,'standard_100_episode_score':False,'overall_competition_score':None}
    atomic_json(run_path(config)/'summary.json',data)
    return data

def ready_errors(config,approval=False):
    errors=[]
    if not approval:errors.append('Explicit start approval flag is absent')
    from validation import validate_config,validate_inputs
    for check in (validate_config,validate_inputs):
        try:check(config)
        except Exception as exc:errors.append('Configuration/input: '+str(exc))
    from model_source import verify_source
    try:verify_source(config)
    except Exception as exc:errors.append('Model source: '+str(exc))
    return errors

def require_ready(config,approval):
    errors=ready_errors(config,approval)
    if errors:raise RuntimeError('Start blocked: '+'; '.join(errors))
