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

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(HERE/'config.json'))
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--start-approved',action='store_true');parser.add_argument('--retry-interrupted',action='store_true')
    cli=parser.parse_args();config=load_config(cli.config)
    require_ready(config,cli.start_approved)
    # The authoritative config records the chosen concurrency, which may be reduced to one.
    if cli.workers>config['workers']:raise RuntimeError('Requested more workers than reviewed configuration')
    root=Path(config['project_root']);run=run_path(config);run.mkdir(parents=True,exist_ok=True)
    lock=(run/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    config_hash=fingerprint(config);frozen=run/'approved-config.json'
    if frozen.exists() and fingerprint(load_config(frozen))!=config_hash:raise RuntimeError('Configuration differs from existing run; use a new directory')
    atomic_json(frozen,config)
    manifest=run/'launch-manifest.json'
    if not manifest.exists():
        atomic_json(manifest,{'label':config['label'],'config_sha256':config_hash,'actual_workers':cli.workers,
            'model_repo':config['model_repo'],'model_path':str((root/config['model_relative']).resolve()),
            'parameters_updated':False,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
            'preparation_manifest':str(run/'preparation-manifest.json'),'automatic_batching':False})
    previous_invocations=run/'invocations.jsonl'
    from common import append_event
    append_event(previous_invocations,{'event':'approved_launch','actual_workers':cli.workers,'retry_interrupted':cli.retry_interrupted,'config_sha256':config_hash})
    stop=threading.Event();children={};children_lock=threading.RLock();drainers=[];summary_lock=threading.Lock();stop_reason={'reason':None}

    def spawn(argv,name,cwd,env,limit):
        p=subprocess.Popen(argv,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        with children_lock:children[name]=p
        def drain():
            path=run/'logs'/f'{name}.log'
            kept=path.stat().st_size if path.exists() else 0;truncated=kept>=limit
            with path.open('ab') as out:
                while True:
                    chunk=p.stdout.read(8192)
                    if not chunk:break
                    room=max(0,limit-kept)
                    if room:out.write(chunk[:room]);out.flush();kept+=min(room,len(chunk))
                    if len(chunk)>room and not truncated:
                        out.write(b'\n[Text log capped; outcomes remain in structured ledgers]\n');truncated=True
            p.stdout.close()
        thread=threading.Thread(target=drain,daemon=True);thread.start();drainers.append(thread)
        return p

    def terminate(p):
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass

    def request_stop(signum=None,frame=None):
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
    try:
        # Do not contact or reuse a potentially unrelated resident service.
        with socket.socket() as s:s.bind(('127.0.0.1',config['port']))
        service=spawn([str(root/'envs/vla/bin/python'),'-B','-u',str(HERE/'serve.py'),'--config',str(frozen),'--start-approved'],
            'inference',root/'repos/lingbot-vla-v2',env,16*1024*1024)
        deadline=time.monotonic()+1200
        while not stop.is_set():
            if service.poll() is not None:raise RuntimeError('Inference service exited before readiness')
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{config['port']}/healthz",timeout=2) as response:
                    if response.status==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('Inference service startup timed out')
            stop.wait(1)
        simenv=env.copy();simenv['LD_LIBRARY_PATH']=str(root/'envs/mplib-build/lib')
        queues=[config['tasks'][i::cli.workers] for i in range(cli.workers)]
        def worker(worker_id):
            for task in queues[worker_id]:
                if stop.is_set():return
                argv=[str(root/'envs/sim/bin/python'),'-B','-u',str(HERE/'evaluate_task.py'),
                      '--config',str(frozen),'--task',task,'--worker-id',str(worker_id),'--start-approved']
                if cli.retry_interrupted:argv.append('--retry-interrupted')
                p=spawn(argv,f'eval-{task}',root/'repos/RoboTwin',simenv,2*1024*1024)
                while p.poll() is None:
                    if stop.wait(1):
                        request_stop();break
                    if service.poll() is not None:
                        stop_reason['reason']='Inference service exited; no restart'
                        request_stop();raise RuntimeError(stop_reason['reason'])
                if stop.is_set():return
                if p.returncode!=0:
                    stop_reason['reason']=f'{task} exited {p.returncode}; no retry or seed substitution'
                    request_stop();raise RuntimeError(f'{task} exited {p.returncode}; no retry or seed substitution')
                with summary_lock:overall(config)
        with ThreadPoolExecutor(max_workers=cli.workers) as pool:
            futures=[pool.submit(worker,i) for i in range(cli.workers)]
            for future in futures:future.result()
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
        result=overall(config)
        atomic_json(run/'execution-status.json',{'state':'completed' if result['complete'] else 'stopped_incomplete',
                                                'summary':str(run/'summary.json'),'automatic_restart':False,
                                                'stop_reason':stop_reason['reason']})
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
