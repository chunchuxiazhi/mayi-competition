"""One-setting correction: matching two-session smoke then 50x10, detached only with explicit approval."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
from common import HERE,load_config,run_path,atomic_json,overall,append_event
from static_check import check
from runtime_resources import startup_check,ResourceGuard,suite_cache_baseline
from port_release import wait_for_port_release

def main():
    p=argparse.ArgumentParser();p.add_argument('--start-approved',action='store_true');p.add_argument('--background',action='store_true');args=p.parse_args()
    cfg=load_config(HERE/'config-randomized.json');report=check(cfg,require_model=True)
    if cfg['instruction_type']!='unseen' or cfg['setting_name']!='randomized' or cfg['baseline_relative'] is not None:raise ValueError('This entry is randomized+unseen only')
    if not args.start_approved:
        if args.background:raise ValueError('Background needs explicit start approval')
        print(json.dumps({'static_checks_passed':True,'runtime_started':False}));return
    root=Path(cfg['project_root']);suite=root/cfg['suite_relative'];run=run_path(cfg)
    from prepare_model import idle
    idle();startup_check(root,85,required_cpus=16)
    if args.background:
        marker=suite/'supervisor.json'
        if marker.exists():raise RuntimeError('Already dispatched; inspect existing run, do not duplicate')
        with (suite/'supervisor.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/'rerun.py'),'--start-approved'],cwd=HERE,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'start_unix':time.time(),'same_20000_checkpoint':True,'instruction_type':'unseen'})
        print(json.dumps({'dispatched':True,'pid':child.pid,'run':str(run)}));return
    lock=(suite/'suite.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    phase='preparing';error=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Supervisor signal '+str(signum))
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    guard=ResourceGuard(root,baseline_cache_bytes=suite_cache_baseline(root,suite))
    def call(mode):
        child=subprocess.Popen([sys.executable,'-B',str(HERE/'driver.py'),'--config',str(HERE/'config-randomized.json'),
                               '--workers','2','--start-approved',mode],cwd=HERE,start_new_session=True)
        try:rc=child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            raise
        if rc:raise RuntimeError('Driver exited '+str(rc)+' during '+mode)
        status=json.loads((run/'execution-status.json').read_text())
        if status.get('stop_reason') or not status.get('owned_children_exited'):raise RuntimeError('Driver failed/children not exited: '+str(status))
    try:
        for phase,mode in [('smoke_running','--smoke-only'),('full_running','--full-suite')]:
            atomic_json(suite/'execution-status.json',{'state':phase,'instruction_type':'unseen','planned_slots':500,'automatic_restart':False})
            wait_for_port_release(cfg['port'],check_resources=guard.check,on_event=lambda e:append_event(suite/'orchestrator-events.jsonl',e))
            idle();startup_check(root,85,required_cpus=16)
            call(mode)
            if phase=='smoke_running':
                evidence=json.loads((run/'smoke-result.json').read_text())
                if not evidence.get('passed'):raise RuntimeError('Smoke failed; full suite will not start')
                for row in evidence['cases']:
                    if row['instruction_type']!='unseen' or row['instruction_evidence']['pool']!='unseen':raise RuntimeError('Smoke used wrong instruction pool')
        phase='finished'
    except BaseException as exc:
        phase='stopped';error=type(exc).__name__+': '+str(exc);raise
    finally:
        summary=overall(cfg,rebuild=True);export=None
        try:
            from export_results import export as do_export
            if (run/'launch-manifest.json').exists():export=do_export([cfg])
        except Exception as exc:
            if error is None:error='Export: '+type(exc).__name__+': '+str(exc)
            phase='export_failed'
        atomic_json(suite/'execution-status.json',{'state':phase,'error':error,'summary':summary,'export':export,
             'instruction_type':'unseen','planned_slots':500,'automatic_restart':False,'training_started':False})
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
