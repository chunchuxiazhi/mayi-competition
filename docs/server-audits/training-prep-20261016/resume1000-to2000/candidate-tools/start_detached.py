"""Explicit, duplicate-protected background start for the reviewed 10000-step run."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
from storage_policy import atomic_json

HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');cli=parser.parse_args()
    if not cli.start:raise SystemExit('Prepared only. Add --start only after explicit approval of formal training.')
    import yaml
    from launch import validate_config
    ctl=json.loads((HERE/'formal10000-control.json').read_text());cfg=yaml.safe_load(Path(ctl['native_config_path']).read_text())
    validate_config(cfg,ctl);root=Path(ctl['project_root']);run=Path(cfg['train']['output_dir'])
    readiness=json.loads((root/'docs/training-prep-20261016/formal10000/readiness.json').read_text())
    if not readiness.get('engineering_checks_passed'):raise RuntimeError('Formal preflight is not complete')
    if readiness['configuration_sha256']!=hashlib.sha256(Path(ctl['native_config_path']).read_bytes()).hexdigest():raise RuntimeError('Configuration changed after preparation')
    if readiness['control_sha256']!=hashlib.sha256((HERE/'formal10000-control.json').read_bytes()).hexdigest():raise RuntimeError('Resource/retention control changed after preparation')
    for name,digest in readiness['code_sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Prepared source changed: '+name)
    active=subprocess.run(['pgrep','-af',str(HERE)+'/(train_entry|launch|export_cpu|verify_reload|verify_training_resume).py'],capture_output=True,text=True).stdout
    if active:raise RuntimeError('Another project model process is active')
    run.mkdir(parents=True,exist_ok=True);logs=run/'logs';logs.mkdir(exist_ok=True)
    # Exclusive creation prevents two requests racing to launch the same run.
    with (run/'launch-request.json').open('x') as f:
        json.dump({'authorized_optimizer_updates':10000,'save_steps':1000,'keep_checkpoints':2,
            'source':'explicit --start invocation','automatic_evaluation':False},f,indent=2)
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        HF_HOME=str(root/'cache/huggingface'),XDG_CACHE_HOME=str(root/'cache/xdg'),MPLCONFIGDIR=str(root/'cache/matplotlib'),TMPDIR=str(root/'cache/tmp'))
    argv=[str(root/'envs/vla/bin/python'),'-B',str(HERE/'launch.py'),'--phase','formal10000','--start']
    with (logs/'launcher.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(run/'launcher-pid.json',{'pid':child.pid,'argv':argv})
    print(json.dumps({'launch_requested':True,'pid':child.pid,'run_directory':str(run)}))
if __name__=='__main__':main()
