"""Sequential full-suite launcher. Default checks only; no model/simulation imports."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/zzj/competitions/lingbot')
TOOLS=ROOT/'tools/checkpoint-dual-setting-eval-20261023'
sys.path.insert(0,str(TOOLS))
from port_release import wait_for_port_release
from common import atomic_json,append_event,run_path
from run_suite import configs,verify_smoke,collect,call
from export_results import export,assert_no_runtime
from runtime_resources import ResourceGuard,startup_check,suite_cache_baseline
from validation import validate_config,validate_inputs
from prepare_model import idle

def verify_installation():
    record=json.loads((HERE/'install-manifest.json').read_text())
    for base,key in ((HERE,'dispatcher_sha256'),(TOOLS,'validated_runtime_sha256')):
        current={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.glob('*.py')}
        if current!=record[key]:raise RuntimeError('Installed code no longer matches reviewed '+key)
    return record

def check():
    installation=verify_installation();cfgs=configs()
    assert_no_runtime(cfgs)
    for cfg in cfgs:validate_config(cfg);validate_inputs(cfg);verify_smoke(cfg)
    return cfgs,installation

def run(args,cfgs,installation):
    suite=ROOT/cfgs[0]['suite_relative'];stop=threading.Event();reason=None;state='starting';result_export=None
    with (suite/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle();startup_check(ROOT,85,required_cpus=8+4*args.workers)
        guard=ResourceGuard(ROOT,baseline_cache_bytes=suite_cache_baseline(ROOT,suite))
        invocation=str(time.time_ns())
        evidence={'invocation_id':invocation,'installation':installation,'files':{p.name:p.read_text() for p in HERE.glob('*.py')},
                  'argv':sys.argv,'workers':args.workers,'same_policy_and_runtime_as_verified_smoke':True}
        atomic_json(suite/'preparation'/f'dispatch-source-{invocation}.json',evidence)
        def event(row):append_event(suite/'dispatch-events.jsonl',{'unix':time.time(),'invocation_id':invocation,**row})
        def interrupted(signum,frame):
            stop.set();raise KeyboardInterrupt('Dispatcher received signal '+str(signum))
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            for cfg in cfgs:
                state='waiting_for_port_release'
                atomic_json(suite/'execution-status.json',{'state':state,'setting':cfg['task_config'],'mode':'full','automatic_restart':False})
                idle();startup_check(ROOT,85,required_cpus=8+4*args.workers)
                wait_for_port_release(cfg['port'],stop=stop,on_event=event,check_resources=guard.check)
                state='running'
                event({'event':'setting_start','setting':cfg['task_config']})
                atomic_json(suite/'execution-status.json',{'state':state,'setting':cfg['task_config'],'mode':'full','automatic_restart':False})
                argv=[sys.executable,'-B',str(TOOLS/'driver.py'),'--config',str(TOOLS/f"config-{cfg['setting_name']}.json"),
                      '--workers',str(args.workers),'--start-approved','--full-suite']
                if args.retry_interrupted:argv.append('--retry-interrupted')
                rc=call(argv)
                current=json.loads((run_path(cfg)/'execution-status.json').read_text())
                if rc or current.get('stop_reason'):raise RuntimeError(cfg['setting_name']+' stopped: '+str(current.get('stop_reason') or rc))
                if not current.get('owned_children_exited'):raise RuntimeError('Previous setting children did not exit')
                event({'event':'setting_end','setting':cfg['task_config'],'all_owned_children_exited':True})
            summary=collect(cfgs);state='completed' if summary['complete'] else 'stopped_incomplete'
        except BaseException as exc:
            state='stopped';reason=type(exc).__name__+': '+str(exc);raise
        finally:
            summary=collect(cfgs)
            if summary['valid_policy_episodes']:
                try:result_export=export(cfgs)
                except Exception as exc:
                    if reason is None:reason='Result export failed: '+type(exc).__name__+': '+str(exc)
                    state='export_failed'
            atomic_json(suite/'execution-status.json',{'state':state,'reason':reason,'mode':'full','dispatcher_invocation_id':invocation,
                        'valid_policy_episodes':summary['valid_policy_episodes'],'planned_slots':1000,'result_export':result_export,
                        'automatic_restart':False,'training_started':False})
            if state=='export_failed':raise RuntimeError(reason)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('check','full'),default='check')
    parser.add_argument('--start-approved',action='store_true');parser.add_argument('--background',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2);parser.add_argument('--retry-interrupted',action='store_true')
    args=parser.parse_args()
    if args.mode!='full' and (args.background or args.start_approved or args.retry_interrupted):raise ValueError('Runtime flags require --mode full')
    if args.mode=='full' and not args.start_approved:raise RuntimeError('Explicit start approval is required')
    cfgs,installation=check()
    if args.mode=='check':
        print(json.dumps({'checks_passed':True,'settings':[c['task_config'] for c in cfgs],'planned_slots':1000,
                          'smokes_still_match':True,'runtime_started':False,'original_runtime_code_unchanged':True}));return
    if args.background:
        suite=ROOT/cfgs[0]['suite_relative'];idle();startup_check(ROOT,85,required_cpus=8+4*args.workers)
        argv=[sys.executable,'-B',str(HERE/'dispatch.py'),'--mode','full','--start-approved','--workers',str(args.workers)]
        if args.retry_interrupted:argv.append('--retry-interrupted')
        with (suite/'dispatcher-launch.log').open('ab') as output:
            child=subprocess.Popen(argv,cwd=HERE,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(2)
        if child.poll() is not None:raise RuntimeError('Dispatcher exited early; inspect dispatcher-launch.log')
        atomic_json(suite/'supervisor.json',{'pid':child.pid,'argv':argv,'automatic_restart':False})
        print(json.dumps({'dispatched':True,'pid':child.pid}));return
    run(args,cfgs,installation)

if __name__=='__main__':main()
