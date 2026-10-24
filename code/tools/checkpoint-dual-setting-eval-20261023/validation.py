"""Read-only checks: no model, Torch, NumPy, YAML or simulator imports."""
import ast,hashlib,json,re,shlex
from pathlib import Path
from common import HERE,fingerprint,load_config,run_path,seeds

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def validate_checkpoint(cfg):
    root=Path(cfg['project_root']).resolve();train=root/cfg['training_relative']
    source=train/'checkpoints/global_step_20000';commit=source/'complete.json'
    if digest(train/'frozen-config.yaml')!=cfg['training_config_sha256']:raise ValueError('Frozen training configuration changed')
    frozen=json.loads((train/'frozen-config.yaml').read_text())
    signature=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    if signature!=cfg['training_configuration_signature'] or frozen['train']['max_steps']!=20000:
        raise ValueError('Wrong training signature or step')
    if digest(commit)!=cfg['checkpoint_commit_sha256']:raise ValueError('Source checkpoint commit changed')
    record=json.loads(commit.read_text())
    if record['signature']!=signature:raise ValueError('Checkpoint signature mismatch')
    for name,size in record['files'].items():
        path=source/name
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or path.stat().st_size!=size:
            raise ValueError('Checkpoint component changed: '+name)
    for name,sha in record['metadata_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('Checkpoint metadata changed: '+name)
    return {'source_checkpoint':str(source),'configuration_signature':signature,'component_bytes':sum(record['files'].values())}

def validate_config(cfg):
    root=Path(cfg['project_root']).resolve();run=run_path(cfg).resolve();suite=(root/cfg['suite_relative']).resolve()
    if not suite.is_relative_to(root/'runs') or run!=suite/cfg['setting_name']:raise ValueError('Output escaped the suite')
    expected_setting={'clean':'demo_clean','randomized':'demo_randomized'}
    if cfg['task_config']!=expected_setting.get(cfg['setting_name']) or cfg['instruction_type']!='seen':raise ValueError('Wrong setting/instruction configuration')
    if cfg['training_step']!=20000 or cfg['test_num']!=10:raise ValueError('This quick suite is pinned to step20000 and 10 rounds')
    tasks=json.loads((HERE/'tasks50.json').read_text())
    if cfg['tasks']!=tasks or len(set(tasks))!=50 or len(tasks)!=50:raise ValueError('Task list mismatch/duplicates')
    launcher=root/'repos/lingbot-vla-v2'/cfg['task_list_source']['path']
    if digest(launcher)!=cfg['task_list_source']['sha256']:raise ValueError('Official launcher changed')
    match=re.search(r'task_list_all=\((.*?)\)',launcher.read_text(),re.S)
    if not match or shlex.split(match.group(1))!=tasks:raise ValueError('Not exactly the official 50-task list')
    if cfg['workers']!=2 or cfg['allowed_workers']!=[1,2]:raise ValueError('Only reviewed two lanes or single-lane replay')
    if not cfg['use_fp32'] or cfg['use_bf16'] or cfg['use_compile'] or not cfg['chunk_ret'] or cfg['use_length']!=50:raise ValueError('Inference recipe changed')
    if any(cfg[k] for k in ('save_data','save_frames','eval_video_log','save_observation_trajectory')):raise ValueError('Observation/video storage is disabled')
    train=root/cfg['training_relative'];expected=train/'exports/global_step_20000/hf_ckpt'
    if root/cfg['model_relative']!=expected or root/cfg['model_wrapper_relative']!=expected:raise ValueError('Wrong shared model source')
    if run.is_relative_to(expected.resolve()):raise ValueError('Results must not be inside model weights')
    norm=cfg['normalization']
    if norm['status']!='verified_clean_only' or digest(norm['path'])!=norm['sha256'] or not Path(norm['evidence']).is_file():raise ValueError('Clean statistics provenance changed')
    if not Path(cfg['expert_rule_review']['evidence']).is_file() or cfg['expert_attempt_limit_per_scene']!=5:raise ValueError('Expert review/guard changed')
    if cfg['expert_failure_policy']!='validate_metadata_then_policy_same_seed':raise ValueError('Expert recovery policy changed')
    for name,sha in cfg['official_runtime_sha256'].items():
        if digest(root/name)!=sha:raise ValueError('Official runtime source changed: '+name)
    client=(root/'repos/RoboTwin/script/eval_policy_client_lingbotvla.py').read_text()
    tree=ast.parse(client)
    # Guard the earlier reviewed upstream change: all planned counts come from usr_args.
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignments=[n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='test_num' for x in n.targets)]
    if len(assignments)!=1 or ast.unparse(assignments[0].value)!="int(usr_args.get('test_num', 100))":raise ValueError('Native client no longer supports configured test_num')
    if cfg['setting_name']=='randomized':
        directory=root/'repos/RoboTwin/assets/background_texture/unseen'
        files=[p for p in directory.iterdir() if p.is_file()]
        if not files or {p.name for p in files}!={f'{i}.png' for i in range(len(files))}:raise ValueError('Randomized evaluation texture IDs missing')
    checkpoint=validate_checkpoint(cfg)
    return {'passed':True,'setting':cfg['task_config'],'tasks':50,'planned_slots':500,'configuration_sha256':fingerprint(cfg),**checkpoint}

def validate_inputs(cfg):
    run=run_path(cfg);plan=json.loads((run/'episodes-plan.json').read_text());provenance=json.loads((run/'preparation/input-provenance.json').read_text())
    if plan['configuration_sha256']!=fingerprint(cfg) or len(plan['episodes'])!=500:raise ValueError('Wrong frozen episode plan')
    keys=set()
    for entry in plan['episodes']:
        key=(entry['task'],entry['episode_index'])
        if key in keys or entry['setting']!=cfg['task_config'] or entry['instruction_type']!=cfg['instruction_type']:raise ValueError('Duplicate/cross-setting episode slot')
        keys.add(key)
        if entry['task'] not in cfg['tasks'] or entry['episode_index'] not in range(cfg['test_num']):raise ValueError('Invalid task/episode index')
        expected=seeds(cfg,cfg['tasks'].index(entry['task']),entry['episode_index'])
        if any(entry[k]!=v for k,v in expected.items()):raise ValueError('Seed drift')
    for path,sha in provenance['official_files_sha256'].items():
        if digest(path)!=sha:raise ValueError('Official input changed: '+path)
    for entry in plan['baseline_ledgers']:
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Paired baseline changed')
    return {'passed':True,'setting':cfg['task_config'],'unique_planned_slots':len(keys)}
