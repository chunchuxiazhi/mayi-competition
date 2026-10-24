"""Create only small frozen plans/provenance for both settings; no simulation."""
import argparse,json,subprocess
from pathlib import Path
from common import HERE,load_config,run_path,seeds,read_events,atomic_json,fingerprint
from validation import digest,validate_config

def prepare(cfg):
    validate_config(cfg);root=Path(cfg['project_root']);run=run_path(cfg);baseline=root/cfg['baseline_relative'] if cfg['baseline_relative'] else None
    slots=[];evidence=[];valid=0
    for ti,task in enumerate(cfg['tasks']):
        completed={}
        if baseline:
            p=baseline/'tasks'/task/'episodes.jsonl'
            if p.is_file():evidence.append({'path':str(p),'sha256':digest(p)})
            for entry in read_events(p):
                if entry['event']=='policy_completed':
                    if entry['episode_index'] in completed:raise RuntimeError('Duplicate baseline outcome')
                    completed[entry['episode_index']]=entry
        for index in range(cfg['test_num']):
            ss=seeds(cfg,ti,index);prior=completed.get(index)
            if prior:
                if prior['setting']!=cfg['task_config'] or prior['instruction_type']!=cfg['instruction_type'] or any(prior[k]!=v for k,v in ss.items()):raise ValueError('Baseline input mismatch')
                valid+=1
            slots.append({'task':task,'setting':cfg['task_config'],'instruction_type':cfg['instruction_type'],
                          'episode_index':index,**ss,'baseline_instruction':prior['instruction'] if prior else None,
                          'baseline_success':prior['success'] if prior else None})
    plan={'configuration_sha256':fingerprint(cfg),'episodes':slots,'baseline_valid_episodes':valid,'baseline_ledgers':evidence}
    official=[root/'repos/RoboTwin/task_config'/f"{cfg['task_config']}.yml",root/'repos/RoboTwin/script/eval_policy_client_lingbotvla.py',root/'repos/RoboTwin/envs/_base_task.py']
    official += [root/'repos/RoboTwin/envs'/f'{task}.py' for task in cfg['tasks']]
    official += [root/'repos/lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py',
                 root/'repos/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',
                 root/'repos/lingbot-vla-v2/configs/robot_configs/robotwin.yaml',
                 root/'repos/RoboTwin/script/deploy/websocket_client_policy.py',
                 root/'repos/RoboTwin/script/deploy/msgpack_numpy.py']
    for path in official:
        if not path.is_file():raise FileNotFoundError(path)
    versions={}
    for name in ('lingbot-vla-v2','RoboTwin'):
        repo=root/'repos'/name
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True)
        changes=subprocess.run(['git','status','--porcelain'],cwd=repo,capture_output=True,text=True) if commit.returncode==0 else None
        versions[name]={'commit':commit.stdout.strip() if commit.returncode==0 else None,
                        'working_tree_changes':changes.stdout.splitlines() if changes is not None and changes.returncode==0 else None,
                        'git_metadata_available':commit.returncode==0,
                        'provenance_note':'Pinned source-file hashes remain authoritative when the checkout is a source snapshot without Git metadata.'}
    report={'official_files_sha256':{str(p):digest(p) for p in official},'code_versions':versions,
            'baseline_valid_episodes':valid,'planned_slots':len(slots),'setting':cfg['task_config'],
            'expert_quota_official_scope_confirmed':False,'normalization_method_officially_confirmed':False,
            'parameters_updated_during_evaluation':False,'data_model_or_old_run_files_modified':False}
    for path,data in ((run/'episodes-plan.json',plan),(run/'preparation/input-provenance.json',report)):
        if path.exists():
            previous=json.loads(path.read_text())
            if previous!=data:
                # Only add source-file pins before the first real invocation.
                # Never change slots, seeds or existing source hashes.
                a={k:v for k,v in previous.items() if k!='official_files_sha256'}
                b={k:v for k,v in data.items() if k!='official_files_sha256'}
                can_extend=(path.name=='input-provenance.json' and a==b and
                            all(data['official_files_sha256'].get(k)==v for k,v in previous['official_files_sha256'].items()) and
                            not any((root/cfg['suite_relative']/setting/'invocations.jsonl').exists() for setting in ('clean','randomized')))
                if not can_extend:raise ValueError('Existing frozen inputs changed; preserve and review')
                backup=path.with_name('input-provenance-before-source-extension.json')
                if not backup.exists():atomic_json(backup,previous)
                atomic_json(path,data)
        else:atomic_json(path,data)
    return {'prepared':True,'planned_slots':len(slots),'setting':cfg['task_config'],'baseline_valid_episodes':valid}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    print(json.dumps(prepare(load_config(args.config))))

if __name__=='__main__':main()
