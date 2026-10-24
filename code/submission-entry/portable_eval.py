"""Fresh clean/seen and randomized/unseen replay. Default: read-only plan/check."""
import argparse,hashlib,json,os,signal,subprocess,sys
from pathlib import Path

BUNDLE=Path(__file__).resolve().parents[2]
RUNTIME=Path(__file__).resolve().parent/'portable_runtime'
PACKAGED=(BUNDLE/'02_代码材料').is_dir()
CONFIGS=BUNDLE/('02_代码材料/configs' if PACKAGED else 'code/submission-configs')
ORIGINAL_CONFIGS=BUNDLE/('02_代码材料/configs/original' if PACKAGED else 'code/configs')
SOURCES=BUNDLE/('02_代码材料/src/runtime-source' if PACKAGED else 'code/runtime-source')
TOOLS=BUNDLE/('02_代码材料/src/tools' if PACKAGED else 'code/tools')
MODEL_MANIFEST=BUNDLE/('04_复现与调优说明/model_manifest.json' if PACKAGED else 'models/step_20000/manifest.json')
sys.path.insert(0,str(RUNTIME))
from common import atomic_json,fingerprint,seeds

def absolute(value):return Path(os.path.abspath(os.path.expanduser(str(value))))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setting',choices=('clean','randomized','both'),default='both')
    p.add_argument('--workdir',type=absolute,required=True,help='Fresh external output/cache/layout root; never a historical evaluation run')
    p.add_argument('--model',type=absolute,default=BUNDLE/('03_模型权重/checkpoint' if PACKAGED else 'models/step_20000/hf_ckpt'))
    p.add_argument('--lingbot-root',type=absolute,default=SOURCES/'lingbot-vla-v2')
    p.add_argument('--robotwin-root',type=absolute,default=SOURCES/'RoboTwin')
    p.add_argument('--qwen-root',type=absolute,default=os.environ.get('QWEN3VL_PATH'))
    p.add_argument('--vla-python',type=absolute,default=os.environ.get('VLA_PYTHON'))
    p.add_argument('--sim-python',type=absolute,default=os.environ.get('SIM_PYTHON'))
    p.add_argument('--mplib-lib',type=absolute,help='Optional native MPLib build/lib directory, prepended to LD_LIBRARY_PATH')
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--test-num',type=int,default=10)
    p.add_argument('--port',type=int,default=19380)
    p.add_argument('--hash-model',action='store_true',help='Check full tensor SHA256 without loading weights')
    p.add_argument('--check',action='store_true',help='Check files/paths only; no model imports or subprocesses')
    p.add_argument('--execute',action='store_true',help='Explicitly launch future reviewer evaluation; never used when assembling this submission')
    return p

def build_configs(a):
    if not 1<=a.test_num<=1000 or not 1024<=a.port<=65535:raise ValueError('Invalid round count or port')
    root=a.workdir.resolve()
    if root==BUNDLE or root.is_relative_to(BUNDLE):raise ValueError('Choose an output directory outside the submission bundle')
    settings=('clean','randomized') if a.setting=='both' else (a.setting,)
    configs=[]
    for setting in settings:
        cfg=json.loads((CONFIGS/f'eval-{ "clean-seen" if setting=="clean" else "randomized-unseen"}.json').read_text(encoding='utf-8'))
        # The original frozen JSON remains evidence; create a separate portable copy.
        for key in ('training_relative','training_tools_relative','training_configuration_signature','training_config_sha256',
                    'checkpoint_commit_sha256','model_source_relative','baseline_relative','control_revision','official_runtime_sha256'):
            cfg.pop(key,None)
        cfg.update(source_mode='submitted_hf',project_root=str(root),suite_relative='runs/replay',run_relative='runs/replay/'+setting,
            model_relative=str(a.model.resolve()),model_wrapper_relative=str(a.model.resolve()),workers=a.workers,
            test_num=a.test_num,port=a.port,model_manifest=str(MODEL_MANIFEST),
            label=f'20,000-step submitted model: {setting}, {a.test_num} planned rounds per task; actual counts only',
            execution_approval={'explicit_cli_required':True,'historical_smoke_not_reused':True},
            mplib_lib=str(a.mplib_lib) if a.mplib_lib else None)
        cfg['normalization']={'path':str(ORIGINAL_CONFIGS/'normalization/robotwin_clean_v3.json'),
            'sha256':'0b5ed854065e9c64e1b8934c274c38d0efa062d743bfad071f1a089983125611',
            'source':'allowed clean V3; see reproduction report'}
        cfg['runtime_source_sha256']={}
        for repo,rel in [(a.lingbot_root,'deploy/lingbot_vla_v2_policy.py'),(a.robotwin_root,'script/eval_policy_client_lingbotvla.py'),
                         (a.robotwin_root,'script/deploy/websocket_client_policy.py'),(a.robotwin_root,'task_config/'+cfg['task_config']+'.yml')]:
            path=repo/rel
            if path.is_file():cfg['runtime_source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        configs.append(cfg)
    return configs

def preflight(a,configs):
    required=[a.model/'config.json',a.model/'model.safetensors.index.json',
        a.lingbot_root/'lingbotvla/models/__init__.py',a.lingbot_root/'lingbotvla/data/__init__.py',
        a.robotwin_root/'script/eval_policy_client_lingbotvla.py',a.robotwin_root/'script/deploy/websocket_client_policy.py']
    if not a.qwen_root or not a.vla_python or not a.sim_python:raise ValueError('Provide --qwen-root, --vla-python and --sim-python (or documented environment variables)')
    required += [a.qwen_root/'config.json',a.vla_python,a.sim_python]
    for p in required:
        if not p.is_file():raise FileNotFoundError(p)
    # A Python executable must remain under its virtualenv prefix; resolving its
    # binary symlink here would accidentally discard pyvenv.cfg.
    for p in (a.vla_python,a.sim_python):
        if p.parent.name!='bin':raise ValueError('Use the Linux virtualenv bin/python entry: '+str(p))
    from model_source import verify_source
    for cfg in configs:
        verify_source(cfg,full_hash=a.hash_model)
        for rel in ['task_config/'+cfg['task_config']+'.yml','task_config/_embodiment_config.yml','task_config/_camera_config.yml']:
            if not (a.robotwin_root/rel).is_file():raise FileNotFoundError(a.robotwin_root/rel)
        # Reuse the installed assets, but demand the reviewed client/tasks and
        # instruction generator rather than silently replacing their behavior.
        for rel in ['script/eval_policy_client_lingbotvla.py','script/deploy/websocket_client_policy.py',
                    'description/utils/generate_episode_instructions.py',*['envs/'+t+'.py' for t in cfg['tasks']]]:
            expected=SOURCES/'RoboTwin'/rel
            actual=a.robotwin_root/rel
            if not actual.is_file() or not expected.is_file() or hashlib.sha256(actual.read_bytes()).digest()!=hashlib.sha256(expected.read_bytes()).digest():
                raise ValueError('Install the bundled reviewed RoboTwin source with official assets: '+rel)
        if not (a.robotwin_root/'assets').is_dir():raise FileNotFoundError('RoboTwin assets are an external prerequisite')
    return {'passed':True,'tensor_hash_requested':a.hash_model,'model_loaded':False,'runtime_started':False}

def link_directory(target,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink():
        if destination.resolve()!=target.resolve():raise ValueError('Existing layout link points elsewhere: '+str(destination))
    elif destination.exists():raise ValueError('Refusing to replace an existing directory: '+str(destination))
    else:destination.symlink_to(target,target_is_directory=True)

def prepare_layout(a,configs):
    root=a.workdir.resolve();root.mkdir(parents=True,exist_ok=True)
    link_directory(a.lingbot_root,root/'repos/lingbot-vla-v2')
    link_directory(a.robotwin_root,root/'repos/RoboTwin')
    link_directory(a.qwen_root,root/'models/Qwen3-VL-4B-Instruct')
    link_directory(a.vla_python.parent.parent,root/'envs/vla')
    link_directory(a.sim_python.parent.parent,root/'envs/sim')
    for cfg in configs:
        directory=root/cfg['run_relative'];directory.mkdir(parents=True,exist_ok=True)
        target=directory/'portable-config.json'
        if target.exists() and json.loads(target.read_text())!=cfg:raise ValueError('Existing replay uses different model/config; choose a fresh workdir')
        if not target.exists():atomic_json(target,cfg)
        plan={'configuration_sha256':fingerprint(cfg),'episodes':[
            {'task':task,'episode_index':i,'setting':cfg['task_config'],'instruction_type':cfg['instruction_type'],**seeds(cfg,t,i)}
            for t,task in enumerate(cfg['tasks']) for i in range(cfg['test_num'])]}
        target=directory/'episodes-plan.json'
        if target.exists() and json.loads(target.read_text())!=plan:raise ValueError('Replay plan changed')
        if not target.exists():atomic_json(target,plan)
    return [root/cfg['run_relative']/'portable-config.json' for cfg in configs]

def export_replay_results(a,configs):
    from common import overall
    data={'schema_version':1,'team_id':'1385264','results':{}}
    details=[]
    for cfg in configs:
        summary=overall(cfg,rebuild=True);details.append(summary)
        data['results'][cfg['setting_name']]={r['task']:{'attempts':r['completed_valid_policy_episodes'],'successes':r['successes']} for r in summary['per_task']}
    atomic_json(a.workdir/'runs/replay/results.json',data)
    atomic_json(a.workdir/'runs/replay/accounting.json',{'settings':details,
        'historical_submission_results_modified':False,'expert_metadata_blocking_is_retained_and_disclosed':True})

def main(argv=None):
    a=parser().parse_args(argv)
    if a.check and a.execute:raise ValueError('--check cannot start evaluation')
    configs=build_configs(a)
    commands=[[str(a.vla_python or 'VLA_PYTHON'),'-B',str(RUNTIME/'driver.py'),'--config',str(a.workdir/c['run_relative']/'portable-config.json'),
               '--workers',str(a.workers),'--full-suite','--start-approved'] for c in configs]
    plan={'settings':[{'setting':c['task_config'],'instructions':c['instruction_type'],'tasks':len(c['tasks']),
                      'rounds_per_task':c['test_num'],'model_path':c['model_relative'],'output':str(a.workdir/c['run_relative'])} for c in configs],
          'commands_for_explicit_execution':commands,'automatic_batching':False,'new_entry_action_inference_verified':False,
          'runtime_started':False,'historical_results_unchanged':True}
    if a.check or a.execute:plan['file_check']=preflight(a,configs)
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if not a.execute:return 0
    if sys.platform!='linux':raise RuntimeError('Simulation runtime requires Linux + NVIDIA')
    paths=prepare_layout(a,configs)
    child=None
    def stop(signum,frame):
        # The driver owns service/workers and handles their cleanup itself.
        if child is not None and child.poll() is None:child.send_signal(signum)
        raise KeyboardInterrupt('Replay interrupted; no automatic retry')
    previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for cfg,path,command in zip(configs,paths,commands):
            from port_release import wait_for_port_release
            wait_for_port_release(cfg['port'])
            child=subprocess.Popen(command,cwd=RUNTIME)
            code=child.wait()
            if code:raise RuntimeError(f"{cfg['setting_name']} driver exited {code}; preserve output; no automatic retry")
        export_replay_results(a,configs)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for s,handler in previous.items():signal.signal(s,handler)
    return 0

if __name__=='__main__':raise SystemExit(main())
