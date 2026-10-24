"""One inference service + at most two simulator subprocesses, no automatic retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from common import HERE,load_config,require_ready,run_path,atomic_json,fingerprint,overall
from recovery import validate_revision
from bounded_log import start_drain
from runtime_resources import ResourceGuard,startup_check,thread_env,suite_cache_baseline

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(HERE/'config-clean.json'))
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--start-approved',action='store_true');parser.add_argument('--retry-interrupted',action='store_true')
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--smoke-only',action='store_true');modes.add_argument('--full-suite',action='store_true')
    cli=parser.parse_args();config=load_config(cli.config)
    require_ready(config,cli.start_approved)
    # The authoritative config records the chosen concurrency, which may be reduced to one.
    if cli.workers>config['workers']:raise RuntimeError('Requested more workers than reviewed configuration')
    root=Path(config['project_root']);run=run_path(config);run.mkdir(parents=True,exist_ok=True)
    lock=(run/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    config_hash=fingerprint(config);original=run/'approved-config.json'
    if original.exists():validate_revision(load_config(original),config)
    else:atomic_json(original,config)
    frozen=run/'approved-config-r2.json' if config.get('control_revision') else original
    if frozen.exists() and fingerprint(load_config(frozen))!=config_hash:raise RuntimeError('Frozen revision differs')
    if not frozen.exists():atomic_json(frozen,config)
    if cli.smoke_only and cli.workers!=2:raise ValueError('Reviewed smoke requires exactly two workers')
    from model_source import verify_source,compare
    source_digest=verify_source(config)
    code_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    if cli.full_suite:
        smoke=load_config(run/'smoke-result.json')
        if not smoke['passed'] or smoke['config_sha256']!=config_hash or smoke['model_source_sha256']!=source_digest or smoke['code_sha256']!=code_hash:
            raise RuntimeError('Successful smoke for identical config, source and code required')
    project_lock=(root/'runs/finetune-clean-v3-gb10/training.lock').open('a')
    fcntl.flock(project_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    from prepare_model import idle,available
    idle()
    startup_check(root,85,required_cpus=8+4*cli.workers)
    guard_state=ResourceGuard(root,baseline_cache_bytes=suite_cache_baseline(root,root/config['suite_relative']))
    smoke_passed=False
    minimum_memory={'bytes':available()}
    manifest=run/'launch-manifest.json'
    if not manifest.exists():
        atomic_json(manifest,{'label':config['label'],'config_sha256':config_hash,'actual_workers':cli.workers,
            'model_repo':config['model_repo'],'model_path':str((root/config['model_relative']).resolve()),
            'training_step':config['training_step'],'model_source_sha256':source_digest,'parameters_updated_during_evaluation':False,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
            'preparation_manifest':str(run/'preparation/input-provenance.json'),'automatic_batching':False})
    previous_invocations=run/'invocations.jsonl'
    from common import append_event
    invocation_id=str(time.time_ns())
    append_event(previous_invocations,{'event':'approved_launch','invocation_id':invocation_id,
        'actual_workers':cli.workers,'retry_interrupted':cli.retry_interrupted,'config_sha256':config_hash,
        'smoke_only':cli.smoke_only,'full_suite':cli.full_suite,
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}})
    stop=threading.Event();children={};children_lock=threading.RLock();drainers=[];summary_lock=threading.Lock();stop_reason={'reason':None}

    def spawn(argv,name,cwd,env,limit):
        p=subprocess.Popen(argv,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        with children_lock:children[name]=p
        def log_error(error):
            stop_reason['reason']=name+' log write/drain failed: '+error
            request_stop()
        drainers.append(start_drain(p.stdout,run/'logs'/f'{name}.log',limit,log_error))
        return p

    def terminate(p):
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass

    def request_stop(signum=None,frame=None):
        if stop.is_set():return
        if signum is not None:stop_reason['reason']=f'Received signal {signum}'
        stop.set()
        with children_lock:
            for p in list(children.values()):terminate(p)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    (run/'logs').mkdir(exist_ok=True)
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED='0',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
        SETUPTOOLS_SCM_PRETEND_VERSION='0.7.8',TORCH_CUDA_ARCH_LIST='12.0',OMP_NUM_THREADS='4',MAX_JOBS='2',
        CUDA_HOME='/usr/local/cuda')
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key,relative in {'HF_HOME':'cache/huggingface','TORCH_HOME':'cache/torch',
        'TORCH_EXTENSIONS_DIR':'cache/torch_extensions','XDG_CACHE_HOME':'cache/xdg',
        'CUDA_CACHE_PATH':'cache/cuda','MPLCONFIGDIR':'cache/matplotlib','TMPDIR':'cache/tmp'}.items():
        path=root/relative;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    env['CUDA_VISIBLE_DEVICES']='0'
    env.update(thread_env(root,'sim'))
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def memory_guard():
        while not stop.wait(2):
            try:
                snapshot=guard_state.check();minimum_memory['bytes']=min(minimum_memory['bytes'],snapshot['unified_mem_available_bytes'])
                atomic_json(run/'resource-status.json',snapshot)
            except BaseException as exc:
                stop_reason['reason']='Resource protection: '+type(exc).__name__+': '+str(exc)
                request_stop();return
    guard=threading.Thread(target=memory_guard,daemon=True);guard.start()
    try:
        atomic_json(run/'execution-status.json',{'state':'starting','invocation_id':invocation_id,
            'summary':str(run/'summary.json'),'automatic_restart':False})
        # Do not contact or reuse a potentially unrelated resident service.
        with socket.socket() as s:s.bind(('127.0.0.1',config['port']))
        service_env=env.copy();service_env.update(thread_env(root,'inference'))
        service=spawn([str(root/'envs/vla/bin/python'),'-B','-u',str(HERE/'serve.py'),'--config',str(frozen),'--start-approved'],
            'inference',root/'repos/lingbot-vla-v2',service_env,16*1024*1024)
        deadline=time.monotonic()+1200
        while not stop.is_set():
            if service.poll() is not None:raise RuntimeError('Inference service exited before readiness')
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{config['port']}/healthz",timeout=2) as response:
                    if response.status==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('Inference service startup timed out')
            stop.wait(1)
        if stop.is_set():raise RuntimeError(stop_reason['reason'] or 'Stopped during service initialization')
        simenv=env.copy();simenv['LD_LIBRARY_PATH']=str(root/'envs/mplib-build/lib')
        if cli.smoke_only:simenv['LINGBOT_SMOKE_BARRIER_DIR']=str(run/'preparation/smoke'/invocation_id)
        def run_task(worker_id,task,indices=None):
            if stop.is_set():return
            argv=[str(root/'envs/sim/bin/python'),'-B','-u',str(HERE/'evaluate_task.py'),
                  '--config',str(frozen),'--task',task,'--worker-id',str(worker_id),'--start-approved']
            if cli.retry_interrupted:argv.append('--retry-interrupted')
            if indices is not None:argv+=['--episode-indices',*[str(i) for i in indices]]
            p=spawn(argv,f'eval-{task}',root/'repos/RoboTwin',simenv,2*1024*1024)
            task_deadline=time.monotonic()+(1800 if indices is not None else 7200)
            while p.poll() is None:
                if task_deadline is not None and time.monotonic()>task_deadline:
                    stop_reason['reason']='Technical task watchdog expired; stop and preserve interrupted slots, no score or automatic retry'
                    request_stop();raise TimeoutError(stop_reason['reason'])
                if stop.wait(1):break
                if service.poll() is not None:
                    stop_reason['reason']='Inference service exited; no restart'
                    request_stop();raise RuntimeError(stop_reason['reason'])
            if stop.is_set():return
            expected=(11,) if indices is not None else (11,10)
            if p.returncode not in expected:
                stop_reason['reason']=f'{task} exited {p.returncode}; global fault or smoke failure'
                request_stop();raise RuntimeError(stop_reason['reason'])
            with summary_lock:overall(config)
            if p.returncode==10:
                append_event(previous_invocations,{'event':'task_incomplete_continue_others','task':task,
                    'invocation_id':invocation_id,'exception':'See per-task structured ledgers; no replacement seed'})

        if cli.smoke_only:
            from common import read_events
            cases=[(c['task'],c['episode_index']) for c in config['smoke_cases']]
            before={t:{e['attempt_id'] for e in read_events(run/'tasks'/t/'episodes.jsonl')
                       if e['event']=='policy_started'} for t,i in cases}
            atomic_json(run/'execution-status.json',{'state':'smoke_running','invocation_id':invocation_id,'automatic_restart':False})
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run_task,w,t,[i]) for w,(t,i) in enumerate(cases)]
                for future in futures:future.result()
            evidence=[]
            for t,i in cases:
                events=read_events(run/'tasks'/t/'episodes.jsonl')
                completed=[e for e in events if e['event']=='policy_completed' and e['episode_index']==i and e['attempt_id'] not in before[t]]
                if len(completed)!=1:raise RuntimeError('Smoke did not complete exact planned episode: '+t)
                outcome=completed[0]
                first=[e for e in events if e['event']=='policy_first_action_verified' and e['attempt_id']==outcome['attempt_id']]
                if len(first)!=1 or outcome['verified_action_chunks']<1:raise RuntimeError('Missing smoke action/session evidence')
                evidence.append(outcome)
            barrier=run/'preparation/smoke'/invocation_id
            resets=[load_config(p) for p in barrier.glob('worker-*.json')]
            if len(resets)!=2 or len({e['session_id'] for e in evidence})!=2:raise RuntimeError('Smoke sessions not distinct')
            if {e['session_id'] for e in resets}!={e['session_id'] for e in evidence}:raise RuntimeError('Reset barrier mismatch')
            atomic_json(run/'smoke-result.json',{'passed':True,'invocation_id':invocation_id,'cases':evidence,
                'config_sha256':config_hash,'model_source_sha256':source_digest,'code_sha256':code_hash,
                'two_sessions_reset_before_first_inference':True,'automatic_batching':False,
                'counts_toward_planned_500':True,'setting':config['task_config'],'full_suite_started':False,
                'interpretation':'Technical completion, not a minimum task success score; failures remain recorded.'})
            smoke_passed=True
            return
        atomic_json(run/'execution-status.json',{'state':'running','stage':'full_50_tasks',
            'invocation_id':invocation_id,'summary':str(run/'summary.json'),'automatic_restart':False})
        queues=[config['tasks'][i::cli.workers] for i in range(cli.workers)]
        def worker(worker_id):
            try:
                for task in queues[worker_id]:
                    if stop.is_set():return
                    run_task(worker_id,task)
            except BaseException as exc:
                if not stop_reason['reason']:stop_reason['reason']=type(exc).__name__+': '+str(exc)
                request_stop()
                raise
        with ThreadPoolExecutor(max_workers=cli.workers) as pool:
            futures=[pool.submit(worker,i) for i in range(cli.workers)]
            for future in futures:future.result()
    except BaseException as exc:
        if not stop_reason['reason']:stop_reason['reason']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        request_stop()
        with children_lock:owned=list(children.values())
        deadline=time.monotonic()+15
        for p in owned:
            try:p.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait(timeout=5)
        for t in drainers:t.join(timeout=5)
        # Recover terminal markers after graceful or hard interruption, from files only.
        from common import read_events,summarize
        for task in config['tasks']:
            ledger=run/'tasks'/task/'episodes.jsonl'
            events=read_events(ledger);closed={e['attempt_id'] for e in events if e['event'] in ('policy_completed','policy_exception','policy_interrupted')}
            for e in events:
                if e['event']=='policy_started' and e['attempt_id'] not in closed:
                    ready=[q for q in events if q.get('attempt_id')==e['attempt_id'] and q['event']=='policy_instruction_ready']
                    append_event(ledger,{**(ready[-1] if ready else e),'event':'policy_interrupted','success':None,
                        'execution_steps':None,'elapsed_seconds':None,'exception':'Process ended before terminal record; no automatic retry'})
            summarize(config,task)
        result=overall(config);compare(config)
        atomic_json(run/'execution-status.json',{'state':'stopped' if stop_reason['reason'] else ('smoke_passed_paused' if smoke_passed else ('completed' if result['complete'] else 'stopped_incomplete')),
                                                'summary':str(run/'summary.json'),'automatic_restart':False,
                                                'stop_reason':stop_reason['reason'],'minimum_available_unified_memory_gib_sampled':minimum_memory['bytes']/2**30,
                                                'owned_children_exited':all(p.poll() is not None for p in owned)})
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
        fcntl.flock(project_lock,fcntl.LOCK_UN);project_lock.close()

if __name__=='__main__':main()
