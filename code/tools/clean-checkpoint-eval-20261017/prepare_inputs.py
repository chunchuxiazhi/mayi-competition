"""Freeze deterministic slots and prior baseline inputs, without simulator/model imports."""
import ast,json,re,subprocess
from pathlib import Path
from common import HERE,load_config,run_path,seeds,read_events,atomic_json
from model_source import digest

def main():
    cfg=load_config();root=Path(cfg['project_root']);run=run_path(cfg);baseline=root/cfg['baseline_relative']
    tasks=cfg['tasks'];assert len(tasks)==len(set(tasks))==50
    assert tasks==json.loads((HERE/'tasks50.json').read_text())
    source=root/'repos/lingbot-vla-v2'/cfg['task_list_source']['path']
    assert digest(source)==cfg['task_list_source']['sha256']
    text=source.read_text()
    # The pinned official launcher explicitly lists all 50 task names.
    for task in tasks:assert re.search(r'\b'+re.escape(task)+r'\b',text),task
    slots=[];evidence=[];valid=0
    for ti,task in enumerate(tasks):
        p=baseline/'tasks'/task/'episodes.jsonl';ev=read_events(p)
        if p.exists():evidence.append({'path':str(p),'sha256':digest(p)})
        completed={}
        for e in ev:
            if e['event']=='policy_completed':
                assert e['episode_index'] not in completed
                completed[e['episode_index']]=e
        for i in range(cfg['test_num']):
            ss=seeds(cfg,ti,i);prior=completed.get(i)
            if prior:
                assert all(prior[k]==v for k,v in ss.items())
                assert prior['instruction_type']=='seen' and prior['setting']=='demo_clean'
                valid+=1
            slots.append({'task':task,'episode_index':i,**ss,
                          'baseline_instruction':prior['instruction'] if prior else None,
                          'baseline_success':prior['success'] if prior else None})
    plan={'episodes':slots,'baseline_valid_episodes':valid,'baseline_ledgers':evidence}
    target=run/'episodes-plan.json'
    if target.exists():assert json.loads(target.read_text())==plan,'Baseline changed; preserve old plan'
    else:atomic_json(target,plan)
    official_paths=[root/'repos/RoboTwin/task_config/demo_clean.yml',root/'repos/RoboTwin/script/eval_policy_client_lingbotvla.py']
    official_paths += [root/'repos/RoboTwin/envs'/f'{t}.py' for t in tasks]
    official_paths += [root/'repos/RoboTwin/envs/_base_task.py']
    existing={str(p):digest(p) for p in official_paths if p.is_file()}
    assert len(existing)>=51,'Official task source missing'
    report={'official_files_sha256':existing,'official_launcher_sha256':digest(source),
            'baseline_valid_episodes':valid,'planned_slots':len(slots),'duplicate_slots':False,
            'training_and_baseline_files_modified':False,'expert_quota_official_scope_confirmed':False}
    p=run/'preparation/input-provenance.json'
    if p.exists():assert json.loads(p.read_text())==report,'Official sources changed'
    else:atomic_json(p,report)
    print(json.dumps({'planned_slots':len(slots),'paired_baseline_valid':valid,'task_sources':len(existing)}))

if __name__=='__main__':main()
