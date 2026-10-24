"""Explicit future CPU export and isolated strict reload. Importing starts nothing."""
import argparse,fcntl,json,os,resource,signal,subprocess,threading,time
from pathlib import Path
from common import HERE,load_config,atomic_json
from validation import validate_config,digest
from model_source import verify_source
from runtime_resources import ResourceGuard,startup_check,thread_env,suite_cache_baseline
from bounded_log import start_drain

def available():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024

def idle():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    if result.stdout.strip():raise RuntimeError('GPU occupied; other projects are left untouched')

def runtime_env(root,cfg):
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        LINGBOT_TRAINING_APPROVED='1',LINGBOT_TRAIN_ATTEMPT='eval-step20000-prepare-only',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        QWEN3VL_PATH=str(root/'models/Qwen3-VL-4B-Instruct'),CUDA_HOME='/usr/local/cuda',
        TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=str(root/cfg['training_tools_relative'])+':'+str(root/'repos/lingbot-vla-v2'))
    for key,rel in {'HF_HOME':'huggingface','TORCH_HOME':'torch','TORCH_EXTENSIONS_DIR':'torch_extensions',
                    'TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg','MPLCONFIGDIR':'matplotlib',
                    'TMPDIR':'tmp','HF_DATASETS_CACHE':'huggingface/datasets'}.items():
        path=root/'cache'/rel;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    env.update(thread_env(root,'prepare'))
    return env

def run_phase(argv,root,env,path,phase,baseline_cache_bytes=None):
    guard=ResourceGuard(root,baseline_cache_bytes=baseline_cache_bytes);log_errors=[]
    child=subprocess.Popen(argv,cwd=root/'repos/lingbot-vla-v2',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    def failed_log(error):
        log_errors.append(error)
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    thread=start_drain(child.stdout,path,16*1024**2,failed_log);deadline=time.monotonic()+1800
    try:
        while child.poll() is None:
            guard.check()
            if log_errors:raise OSError('Model preparation log failed: '+log_errors[-1])
            if time.monotonic()>deadline:raise TimeoutError(phase+' exceeded 30 minutes')
            time.sleep(2)
        if child.returncode:raise RuntimeError(phase+' failed; inspect its bounded log')
        thread.join(timeout=5)
        if log_errors:raise OSError('Model preparation log failed: '+log_errors[-1])
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        thread.join(timeout=5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'config-clean.json'));p.add_argument('--prepare-approved',action='store_true');args=p.parse_args()
    if not args.prepare_approved:raise RuntimeError('Explicit future model preparation approval required')
    cfg=load_config(args.config);validate_config(cfg);root=Path(cfg['project_root']);train=root/cfg['training_relative']
    prep=root/cfg['suite_relative']/'preparation';prep.mkdir(parents=True,exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock=(root/'runs/finetune-clean-v3-gb10/training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise KeyboardInterrupt('Model preparation interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        idle()
        startup_check(root,90,required_cpus=8)
        import shutil
        if shutil.disk_usage(root).free<40*2**30:raise RuntimeError('Need 40GiB free for the single inference export and reserve')
        hf=root/cfg['model_relative'];env=runtime_env(root,cfg);tools=root/cfg['training_tools_relative']
        baseline=suite_cache_baseline(root,root/cfg['suite_relative'])
        if hf.with_name('hf_ckpt.incomplete').exists():raise RuntimeError('Incomplete export preserved; inspect before retrying')
        for name,gpu,needed in [('export_cpu','',not hf.exists()),('verify_reload','0',True)]:
            if not needed:continue
            idle();env['CUDA_VISIBLE_DEVICES']=gpu
            atomic_json(prep/'model-preparation-status.json',{'state':name,'started_unix':time.time()})
            argv=[str(root/'envs/vla/bin/python'),'-B','-u',str(tools/(name+'.py')),
                  '--config',str(train/'frozen-config.yaml'),'--control',str(tools/'formal20000-control.json')]
            run_phase(argv,root,env,prep/(name+'.log'),name,baseline)
        manifest=json.loads((hf/'export-manifest.json').read_text());shards={}
        for name,entry in manifest['weights'].items():
            path=hf/name
            if digest(path)!=entry['sha256']:raise ValueError('Export checksum mismatch')
            shards[name]={**entry,'mtime_ns':path.stat().st_mtime_ns}
        small=[hf/'export-manifest.json',hf/'config.json',train/'lingbotvla_cli.yaml',train/'reload-verification.json',
               train/'checkpoints/global_step_20000/complete.json',train/'frozen-config.yaml']
        record={'training_step':20000,'model_repo':cfg['model_repo'],'hf_path':str(hf),
                'configuration_signature':cfg['training_configuration_signature'],'shards':shards,
                'small_files':[{'path':str(path),'sha256':digest(path)} for path in small],
                'parameters_updated_during_evaluation':False,'one_export_shared_by_settings_and_workers':True}
        atomic_json(root/cfg['model_source_relative'],record);verify_source(cfg,full_hash=False);idle()
        atomic_json(prep/'model-preparation-status.json',{'state':'prepared_and_released','export_bytes':sum(e['bytes'] for e in shards.values()),
                                                        'simulation_or_policy_inference_executed':False})
        print(json.dumps({'prepared':True,'export_bytes':sum(e['bytes'] for e in shards.values()),'gpu_released':True}))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
