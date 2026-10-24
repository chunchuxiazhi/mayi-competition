"""Explicit CPU export then isolated strict reload; never starts training or evaluation."""
import argparse,fcntl,json,os,resource,signal,subprocess,time
from pathlib import Path
from common import HERE,load_config,run_path,atomic_json
from model_source import digest,TRAIN_SIGNATURE,verify_source

def available():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024

def idle():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if p.stdout.strip():raise RuntimeError('GPU already has compute clients; stop preparation without affecting them')

def runtime_env(root):
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        LINGBOT_TRAINING_APPROVED='1',LINGBOT_TRAIN_ATTEMPT='eval-step10000-prepare',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        QWEN3VL_PATH=str(root/'models/Qwen3-VL-4B-Instruct'),CUDA_HOME='/usr/local/cuda',
        TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH=str(root/'tools/finetune-clean-v3-gb10')+':'+str(root/'repos/lingbot-vla-v2'))
    for key,rel in {'HF_HOME':'huggingface','TORCH_HOME':'torch','TORCH_EXTENSIONS_DIR':'torch_extensions',
        'TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg',
        'MPLCONFIGDIR':'matplotlib','TMPDIR':'tmp','HF_DATASETS_CACHE':'huggingface/datasets'}.items():
        p=root/'cache'/rel;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
    return env

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-approved',action='store_true');a=p.parse_args()
    if not a.prepare_approved:raise RuntimeError('Explicit model preparation approval required')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    cfg=load_config();root=Path(cfg['project_root']);run=run_path(cfg);prep=run/'preparation';prep.mkdir(parents=True,exist_ok=True)
    lock=(root/'runs/finetune-clean-v3-gb10/training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    idle()
    if available()<90*2**30:raise MemoryError('Need 90 GiB available unified memory before preparation')
    train=root/'runs/finetune-clean-v3-gb10/clean50-10000-fixed1000';tools=root/'tools/finetune-clean-v3-gb10'
    if digest(train/'frozen-config.yaml')!=TRAIN_SIGNATURE:raise ValueError('Unexpected frozen training config')
    hf=root/cfg['model_relative'];env=runtime_env(root);minimum=available();child=None
    def stop(signum,frame):raise KeyboardInterrupt('Preparation interrupted')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        for name,gpu,needed in [('export_cpu','',not hf.exists()),('verify_reload','0',True)]:
            if not needed:continue
            idle();env['CUDA_VISIBLE_DEVICES']=gpu
            atomic_json(prep/'model-preparation-status.json',{'state':name,'started_unix':time.time()})
            with (prep/(name+'.log')).open('ab') as log:
                child=subprocess.Popen([str(root/'envs/vla/bin/python'),'-B','-u',str(tools/(name+'.py')),
                    '--config',str(train/'frozen-config.yaml'),'--control',str(tools/'formal10000-control.json')],
                    cwd=root/'repos/lingbot-vla-v2',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+1800
                while child.poll() is None:
                    minimum=min(minimum,available())
                    if available()<16*2**30:raise MemoryError('Unified memory reserve below 16 GiB')
                    if time.monotonic()>deadline:raise TimeoutError(name+' exceeded 30 minutes')
                    time.sleep(2)
                if child.returncode:raise RuntimeError(name+' failed; inspect phase log')
                child=None
        manifest=json.loads((hf/'export-manifest.json').read_text())
        small=[hf/'export-manifest.json',hf/'config.json',train/'lingbotvla_cli.yaml',train/'reload-verification.json',
               train/'checkpoints/global_step_10000/complete.json',train/'frozen-config.yaml']
        shards={}
        for name,m in manifest['weights'].items():
            f=hf/name
            if digest(f)!=m['sha256']:raise ValueError('Export checksum mismatch')
            shards[name]={**m,'mtime_ns':f.stat().st_mtime_ns}
        record={'model_repo':cfg['model_repo'],'training_step':10000,'hf_path':str(hf),
            'training_configuration_signature':manifest['configuration_signature'],'frozen_configuration_file_sha256':TRAIN_SIGNATURE,'shards':shards,
            'small_files':[{'path':str(f),'sha256':digest(f)} for f in small],
            'normalization_sha256':cfg['normalization']['sha256'],'parameters_updated_during_evaluation':False}
        atomic_json(run/'model-source.json',record);verify_source(cfg)
        idle();atomic_json(prep/'model-preparation-status.json',{'state':'prepared_and_released',
            'minimum_available_unified_memory_gib_sampled':minimum/2**30,'export_bytes':sum(s['bytes'] for s in shards.values())})
        print(json.dumps({'prepared':True,'export_bytes':sum(s['bytes'] for s in shards.values()),'gpu_released':True}),flush=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
