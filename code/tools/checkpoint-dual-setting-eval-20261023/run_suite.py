"""Default static mode. Runtime stages require explicit flags and never chain automatically."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from common import HERE,load_config,run_path,atomic_json,fingerprint,overall
from validation import validate_config,validate_inputs,digest

def configs():return [load_config(HERE/f'config-{name}.json') for name in ('clean','randomized')]

def verify_smoke(cfg):
    from model_source import verify_source
    result=json.loads((run_path(cfg)/'smoke-result.json').read_text())
    code={p.name:digest(p) for p in HERE.glob('*.py')}
    if not result.get('passed') or result['config_sha256']!=fingerprint(cfg) or result['model_source_sha256']!=verify_source(cfg) or result['code_sha256']!=code:
        raise RuntimeError('A matching successful smoke for each setting is required')
    return result

def collect(cfgs):
    from export_results import assert_no_runtime
    assert_no_runtime(cfgs)
    root=Path(cfgs[0]['project_root'])/cfgs[0]['suite_relative'];rows=[]
    for cfg in cfgs:
        result=overall(cfg,rebuild=True);status_path=run_path(cfg)/'execution-status.json'
        status=json.loads(status_path.read_text()) if status_path.is_file() else {'state':'not_started'}
        rows.append({'setting':cfg['task_config'],'summary':result,'execution_status':status})
    valid=sum(r['summary']['completed_valid_policy_episodes'] for r in rows);success=sum(r['summary']['successes'] for r in rows)
    data={'label':'每任务10回合的20000步模型 clean+randomized 快速评测','training_step':20000,
          'planned_slots':1000,'valid_policy_episodes':valid,'successes':success,
          'success_rate_on_valid':success/valid if valid else None,
          'complete':all(r['summary']['complete'] for r in rows),
          'policy_exceptions':sum(r['summary']['policy_exceptions'] for r in rows),
          'expert_blocked_slots':sum(r['summary']['expert_blocked_episode_slots'] for r in rows),
          'interrupted_attempts':sum(r['summary']['interrupted_attempts'] for r in rows),
          'per_setting':rows,'standard_100_episode_score':False,'overall_competition_score':None}
    atomic_json(root/'summary.json',data)
    return data

def call(argv):
    from bounded_log import start_drain
    cfg=configs()[0];suite=Path(cfg['project_root'])/cfg['suite_relative'];errors=[]
    child=subprocess.Popen(argv,cwd=HERE,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    def log_failed(error):
        errors.append(error)
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    thread=start_drain(child.stdout,suite/'orchestrator-children.log',2*1024**2,log_failed)
    try:
        result=child.wait();thread.join(timeout=5)
        if errors:raise OSError('Orchestrator child log failed: '+errors[-1])
        return result
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        raise
    finally:thread.join(timeout=5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['static','prepare-model','smoke','full','export-results'],default='static')
    p.add_argument('--prepare-approved',action='store_true');p.add_argument('--start-approved',action='store_true')
    p.add_argument('--background',action='store_true');p.add_argument('--workers',type=int,choices=[1,2],default=2)
    p.add_argument('--retry-interrupted',action='store_true');args=p.parse_args()
    if args.mode in ('smoke','full') and not args.start_approved:raise RuntimeError('Explicit future evaluation start approval is absent')
    if args.mode=='prepare-model' and not args.prepare_approved:raise RuntimeError('Explicit future model preparation approval is absent')
    if args.background and args.mode!='full':raise ValueError('Only the explicitly approved full suite can detach')
    if args.mode=='smoke' and args.workers!=2:raise ValueError('Two-session smoke is required before full evaluation')
    if args.retry_interrupted and args.mode!='full':raise ValueError('Explicit interrupted-slot retry is full-mode only')
    cfgs=configs()
    if args.mode=='static':
        from static_check import check
        for cfg in cfgs:print(json.dumps(check(cfg)))
        return
    for cfg in cfgs:validate_config(cfg);validate_inputs(cfg)
    if args.mode=='full':
        from common import require_ready
        for cfg in cfgs:require_ready(cfg,args.start_approved);verify_smoke(cfg)
    root=Path(cfgs[0]['project_root'])/cfgs[0]['suite_relative']
    if args.background:
        from prepare_model import idle,available
        idle()
        from runtime_resources import startup_check
        startup_check(cfgs[0]['project_root'],85,required_cpus=8+4*args.workers)
        argv=[sys.executable,'-B',str(HERE/'run_suite.py'),'--mode','full','--start-approved','--workers',str(args.workers)]
        if args.retry_interrupted:argv.append('--retry-interrupted')
        with (root/'suite-launcher.log').open('ab') as log:
            child=subprocess.Popen(argv,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(2)
        if child.poll() is not None:raise RuntimeError('Suite supervisor exited early; inspect suite-launcher.log')
        atomic_json(root/'supervisor.json',{'pid':child.pid,'argv':argv,'automatic_restart':False})
        print(json.dumps({'dispatched':True,'pid':child.pid,'status':str(root/'execution-status.json')}));return
    lock=(root/'suite.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise KeyboardInterrupt('Suite interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    state='preparing_model' if args.mode=='prepare-model' else args.mode;reason=None
    atomic_json(root/'execution-status.json',{'state':state,'mode':args.mode,'automatic_restart':False})
    try:
        if args.mode=='prepare-model':
            rc=call([sys.executable,'-B',str(HERE/'prepare_model.py'),'--prepare-approved'])
            if rc:raise RuntimeError('Model preparation failed; no simulation started')
            state='model_prepared_paused';return
        if args.mode=='export-results':
            from export_results import export
            result=export(cfgs);state='results_exported';print(json.dumps(result));return
        for cfg in cfgs:
            path=HERE/f"config-{cfg['setting_name']}.json"
            if args.mode=='smoke' and (run_path(cfg)/'smoke-result.json').exists():
                verify_smoke(cfg);continue
            argv=[sys.executable,'-B',str(HERE/'driver.py'),'--config',str(path),'--workers',str(args.workers),
                  '--start-approved','--smoke-only' if args.mode=='smoke' else '--full-suite']
            if args.retry_interrupted:argv.append('--retry-interrupted')
            rc=call(argv)
            if rc:raise RuntimeError(cfg['setting_name']+' driver failed; no automatic retry')
            if args.mode=='smoke':verify_smoke(cfg)
            current=json.loads((run_path(cfg)/'execution-status.json').read_text())
            if current.get('stop_reason'):raise RuntimeError(cfg['setting_name']+': '+current['stop_reason'])
        summary=collect(cfgs)
        state='smoke_passed_paused' if args.mode=='smoke' else ('completed' if summary['complete'] else 'stopped_incomplete')
    except BaseException as exc:
        state='stopped';reason=type(exc).__name__+': '+str(exc);raise
    finally:
        try:
            summary=collect(cfgs);export_error=None;export_record=None
            if args.mode=='full' and any((run_path(c)/'tasks'/task/'episodes.jsonl').exists() for c in cfgs for task in c['tasks']):
                from export_results import export
                try:export_record=export(cfgs)
                except Exception as exc:
                    export_error=type(exc).__name__+': '+str(exc)
                    if state!='stopped':state='export_failed'
            atomic_json(root/'execution-status.json',{'state':state,'reason':reason,'mode':args.mode,
                         'valid_policy_episodes':summary['valid_policy_episodes'],'planned_slots':1000,
                         'result_export':export_record,'export_error':export_error,
                         'automatic_restart':False,'training_started':False})
            if export_error:raise RuntimeError('Raw episode records preserved, but result export failed: '+export_error)
        finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
