"""Static checks for a fresh replay using submitted HF weights and explicit paths."""
import ast,hashlib,json,re,shlex
from pathlib import Path
from common import fingerprint,run_path,seeds

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def validate_config(cfg):
    root=Path(cfg['project_root']).resolve();run=run_path(cfg).resolve()
    if not run.is_relative_to(root/'runs') or cfg['source_mode']!='submitted_hf':raise ValueError('Invalid run root/source mode')
    setting=cfg['setting_name']
    if cfg['task_config']!={'clean':'demo_clean','randomized':'demo_randomized'}[setting] or cfg['instruction_type']!={'clean':'seen','randomized':'unseen'}[setting]:raise ValueError('Wrong setting/instruction pair')
    tasks=cfg['tasks']
    launcher=root/'repos/lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh'
    match=re.search(r'task_list_all=\((.*?)\)',launcher.read_text(),re.S)
    if len(tasks)!=50 or len(set(tasks))!=50 or not match or shlex.split(match.group(1))!=tasks:raise ValueError('Not the official 50 tasks')
    if type(cfg['test_num']) is not int or not 1<=cfg['test_num']<=1000:raise ValueError('Invalid actual test count')
    if cfg['workers'] not in (1,2) or cfg['expert_attempt_limit_per_scene']!=5:raise ValueError('Invalid worker/quota setting')
    if not cfg['use_fp32'] or cfg['use_bf16'] or cfg['use_compile'] or cfg['use_length']!=50:raise ValueError('Inference recipe changed')
    if any(cfg[k] for k in ('save_data','save_frames','eval_video_log','save_observation_trajectory')):raise ValueError('Observation saving disabled')
    norm=cfg['normalization']
    if digest(norm['path'])!=norm['sha256']:raise ValueError('Clean normalization changed')
    for path,sha in cfg['runtime_source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Runtime source changed: '+path)
    client=ast.parse((root/'repos/RoboTwin/script/eval_policy_client_lingbotvla.py').read_text())
    main=next(n for n in client.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignments=[n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='test_num' for t in n.targets)]
    if len(assignments)!=1 or ast.unparse(assignments[0].value)!="int(usr_args.get('test_num', 100))":raise ValueError('Client test_num adapter missing')
    if setting=='randomized':
        textures=root/'repos/RoboTwin/assets/background_texture/unseen'
        files={p.name for p in textures.iterdir() if p.is_file()}
        if not files or files!={f'{i}.png' for i in range(len(files))}:raise ValueError('Randomized texture assets incomplete')
    return {'passed':True,'source_mode':'submitted_hf','setting':setting}

def validate_inputs(cfg):
    plan=json.loads((run_path(cfg)/'episodes-plan.json').read_text())
    if plan['configuration_sha256']!=fingerprint(cfg):raise ValueError('Plan/config mismatch')
    expected=[{'task':task,'episode_index':i,'setting':cfg['task_config'],'instruction_type':cfg['instruction_type'],
        **seeds(cfg,t,i)} for t,task in enumerate(cfg['tasks']) for i in range(cfg['test_num'])]
    if plan['episodes']!=expected:raise ValueError('Duplicate/missing episode or seed drift')
    return {'passed':True,'planned_slots':len(expected)}
