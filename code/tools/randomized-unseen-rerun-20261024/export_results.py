"""Export verified raw counts and their evidence. Never labels 10 rounds as official 100."""
import argparse,csv,fcntl,hashlib,io,json,os,shutil,tempfile,zipfile
from pathlib import Path
from common import HERE,load_config,run_path,read_events,atomic_json,summarize,fingerprint,durable_mkdir,fsync_directory
from validation import digest,validate_inputs

def build_results(cfgs):
    groups=[]
    for cfg in cfgs:
        plan=json.loads((run_path(cfg)/'episodes-plan.json').read_text())
        planned={(e['task'],e['episode_index']):e for e in plan['episodes']}
        for task in cfg['tasks']:
            summary=summarize(cfg,task);events=read_events(run_path(cfg)/'tasks'/task/'episodes.jsonl')
            rounds=[]
            for index in range(cfg['test_num']):
                reference=planned[(task,index)];records=[e for e in events if e['episode_index']==index]
                terminal=[e for e in records if e['event'] in ('policy_completed','policy_exception')]
                if len(terminal)>1:raise ValueError('Duplicate outcome during export')
                chosen=terminal[0] if terminal else next((e for e in reversed(records) if e['event'] in ('policy_interrupted','episode_blocked')),None)
                closed={e['attempt_id'] for e in records if e['event'] in ('policy_completed','policy_exception','policy_interrupted')}
                open_attempt=next((e for e in reversed(records) if e['event']=='policy_started' and e['attempt_id'] not in closed),None)
                if open_attempt is not None and terminal:raise ValueError('Completed slot also has an open attempt')
                if open_attempt is not None:chosen=open_attempt
                status={'policy_completed':'valid','policy_exception':'policy_exception','policy_interrupted':'interrupted',
                        'episode_blocked':'expert_blocked','policy_started':'open'}.get(chosen['event'],'not_started') if chosen else 'not_started'
                if chosen:
                    for key in ('seed','policy_seed','instruction_seed'):
                        if chosen.get(key)!=reference[key]:raise ValueError('Seed/provenance drift during export: '+key)
                    if chosen.get('instruction_type')!=cfg['instruction_type']:raise ValueError('Instruction type drift')
                valid=status=='valid'
                if valid and (type(chosen['success']) is not bool or not chosen.get('instruction')):raise ValueError('Malformed valid outcome')
                if valid and reference.get('baseline_instruction') is not None and chosen['instruction']!=reference['baseline_instruction']:
                    raise ValueError('Paired clean instruction drift')
                rounds.append({'episode_index':index,'status':status,'success':chosen['success'] if valid else None,
                    'instruction':chosen.get('instruction') if chosen else None,'seed':reference['seed'],
                    'policy_seed':reference['policy_seed'],'instruction_seed':reference['instruction_seed'],
                    'execution_steps':chosen.get('execution_steps') if chosen else None,
                    'elapsed_seconds':chosen.get('elapsed_seconds') if chosen else None,
                    'exception':chosen.get('exception') if chosen else None})
            groups.append({'task':task,'setting':cfg['setting_name'],'task_config':cfg['task_config'],
                'instruction_type':cfg['instruction_type'],'test_num_planned':cfg['test_num'],
                'test_num_actual_valid':summary['completed_valid_policy_episodes'],'success_num':summary['successes'],
                'failure_num':summary['completed_valid_policy_episodes']-summary['successes'],
                'success_rate_on_valid':summary['success_rate'],'expert_blocked_slots':summary['expert_blocked_episode_slots'],
                'policy_exceptions':summary['policy_exceptions'],'interrupted_attempts':summary['interrupted_attempts'],
                'open_policy_attempts':summary['open_policy_attempts'],'rounds':rounds})
    valid=sum(g['test_num_actual_valid'] for g in groups);success=sum(g['success_num'] for g in groups)
    return {'format':'lingbot_quick_results_v1','label':'20000-step, 10 rounds per task and setting, quick evaluation',
            'training_step':20000,'standard_100_episode_score':False,'official_schema_verified':False,
            'planned_rounds':sum(g['test_num_planned'] for g in groups),'valid_rounds':valid,'successes':success,
            'success_rate_on_valid':success/valid if valid else None,'tasks_and_settings':groups}

def assert_no_runtime(cfgs):
    marker=str(HERE).encode()
    for p in Path('/proc').glob('[0-9]*'):
        try:command=(p/'cmdline').read_bytes()
        except OSError:continue
        if marker in command and any(x in command for x in (b'/driver.py',b'/serve.py',b'/evaluate_task.py')):
            raise RuntimeError('Export requires evaluation workers and service to have exited')

def verify_archive(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    with zipfile.ZipFile(directory/'quick_results_evidence.zip') as archive:
        if len(archive.namelist())!=len(set(archive.namelist())) or set(archive.namelist())!=set(manifest['files'])|{'manifest.json'}:
            raise ValueError('Archive members differ from the manifest')
        if archive.read('manifest.json')!=(directory/'manifest.json').read_bytes():raise ValueError('Archive manifest mismatch')
        for name,record in manifest['files'].items():
            if archive.getinfo(name).file_size!=record['bytes']:raise ValueError('Archive member size mismatch')
            hashed=hashlib.sha256()
            with archive.open(name) as f:
                while block:=f.read(1024*1024):hashed.update(block)
            if hashed.hexdigest()!=record['sha256']:raise ValueError('Archive member checksum mismatch: '+name)
            if '/' not in name and digest(directory/name)!=record['sha256']:raise ValueError('Result file checksum mismatch: '+name)
    return manifest

def point_to_export(suite,target):
    atomic_json(Path(suite)/'latest-result-export.json',{'directory':str(target),'manifest_sha256':digest(target/'manifest.json'),
                                                       'archive_sha256':digest(target/'quick_results_evidence.zip')})

def export(cfgs):
    assert_no_runtime(cfgs)
    from model_source import verify_source
    for cfg in cfgs:validate_inputs(cfg);verify_source(cfg)
    current_code={p.name:digest(p) for p in HERE.glob('*.py')}
    for cfg in cfgs:
        launch_path=run_path(cfg)/'launch-manifest.json'
        if launch_path.is_file():
            launch=json.loads(launch_path.read_text())
            if launch['config_sha256']!=fingerprint(cfg) or launch['code_sha256']!=current_code or launch['model_source_sha256']!=verify_source(cfg):
                raise RuntimeError('Export code/config/model no longer matches the actual evaluation launch')
    suite=Path(cfgs[0]['project_root'])/cfgs[0]['suite_relative'];exports=suite/'exports';durable_mkdir(exports)
    if shutil.disk_usage(suite).free<8*2**30:raise OSError('Keep 8GiB disk reserve for result export')
    locks=[]
    try:
        for cfg in cfgs:
            lock=(run_path(cfg)/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        result=build_results(cfgs);evidence=[]
        for cfg in cfgs:
            run=run_path(cfg)
            paths=[run/'episodes-plan.json',run/'preparation/input-provenance.json',run/'approved-config.json',
                   run/'launch-manifest.json',run/'execution-status.json']
            paths.extend(p for task in cfg['tasks'] for p in (run/'tasks'/task/'episodes.jsonl',run/'tasks'/task/'expert.jsonl') if p.is_file())
            evidence.extend((p,'evidence/'+str(p.relative_to(suite))) for p in paths if p.is_file())
        source=Path(cfgs[0]['project_root'])/cfgs[0]['model_source_relative'];evidence.append((source,'evidence/model-source.json'))
        for path in HERE.iterdir():
            if path.is_file() and path.suffix in ('.py','.json','.md'):evidence.append((path,'code/'+path.name))
        files={name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p,name in evidence}
        result_digest=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
        identity=hashlib.sha256(json.dumps({'result':result_digest,'files':files},sort_keys=True).encode()).hexdigest()
        target=exports/('quick-'+identity[:16])
        if target.exists():
            if json.loads((target/'manifest.json').read_text())['identity_sha256']!=identity:raise RuntimeError('Existing export identity mismatch')
            verify_archive(target);point_to_export(suite,target)
            return {'exported':str(target),'reused_identical_export':True,'official_submission_ready':False}
        temp=Path(tempfile.mkdtemp(prefix='.partial-',dir=exports))
        atomic_json(temp/'quick_results.json',result)
        output=io.StringIO();writer=csv.writer(output)
        columns=['setting','task','test_num_planned','test_num_actual_valid','success_num','failure_num','success_rate_on_valid','expert_blocked_slots','policy_exceptions','interrupted_attempts','open_policy_attempts']
        writer.writerow(columns)
        for group in result['tasks_and_settings']:writer.writerow([group[k] for k in columns])
        with (temp/'per_task.csv').open('w',encoding='utf-8-sig',newline='') as f:f.write(output.getvalue());f.flush();os.fsync(f.fileno())
        readiness={'official_submission_ready':False,'official_schema_verified':False,'required_rounds_per_task_setting':100,
                   'actual_planned_rounds_per_task_setting':10,'valid_rounds':result['valid_rounds'],
                   'planned_rounds':result['planned_rounds'],'model_weights_in_this_small_archive':False,
                   'blocking_items':['Only 10 rounds planned; latest QA permits actual counts with official re-evaluation final.',
                                     'Official attachment schema and upload file-size limit have not been verified.',
                                     'Final submission must include the matching model weights, code and reproduction report.']}
        atomic_json(temp/'submission_readiness.json',readiness)
        for name in ('quick_results.json','per_task.csv','submission_readiness.json'):
            files[name]={'bytes':(temp/name).stat().st_size,'sha256':digest(temp/name)}
        manifest={'identity_sha256':identity,'model_source_sha256':verify_source(cfgs[0]),'files':files,
                  'config_sha256':{c['setting_name']:fingerprint(c) for c in cfgs},'weights_or_datasets_copied':False}
        atomic_json(temp/'manifest.json',manifest)
        with zipfile.ZipFile(temp/'quick_results_evidence.zip','w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
            for name in ('quick_results.json','per_task.csv','submission_readiness.json','manifest.json'):archive.write(temp/name,name)
            for path,name in evidence:
                if digest(path)!=files[name]['sha256']:raise RuntimeError('Evidence changed during export')
                archive.write(path,name)
        verify_archive(temp)
        with (temp/'quick_results_evidence.zip').open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,target);fsync_directory(exports)
        point_to_export(suite,target)
        return {'exported':str(target),'valid_rounds':result['valid_rounds'],'official_submission_ready':False}
    finally:
        for lock in reversed(locks):fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def main():
    parser=argparse.ArgumentParser();parser.parse_args()
    cfgs=[load_config(HERE/f'config-{name}.json') for name in ('clean','randomized')]
    suite=Path(cfgs[0]['project_root'])/cfgs[0]['suite_relative']
    with (suite/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(export(cfgs)))

if __name__=='__main__':main()
