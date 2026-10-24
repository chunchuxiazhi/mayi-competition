"""Bounded GPU probe, followed by zero-update reload and test-only cleanup."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import time
from storage_policy import GIB,atomic_json,checked_checkpoint
from gb10_runtime import available_memory

PROJECT=Path('/home/zzj/competitions/lingbot')
ROOT=PROJECT/'runs/finetune-clean-v3-gb10'
PROBE=ROOT/'save-continuation-probe-20261016'
TOOLS=PROBE/'tools'


def fingerprint():
    files=[]
    for run in ('pilot','clean50-10000-fixed1000'):
        folder=ROOT/run
        files.extend(folder.glob('frozen-config.yaml'))
        files.extend((folder/'checkpoints').rglob('*'))
    files.extend((PROJECT/'tools/finetune-clean-v3-gb10').glob('*.py'))
    result={}
    for path in files:
        if path.is_file():
            stat=path.stat()
            result[str(path)]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            if stat.st_size<1024*1024:
                result[str(path)]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock=(ROOT/'training.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (PROBE/'result.json').exists():
        raise RuntimeError('Probe already finished; do not overwrite evidence')
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
    if gpu.returncode or gpu.stdout.strip():
        raise RuntimeError('GPU is in use; no overlap allowed')
    if available_memory()<90*GIB or shutil.disk_usage(PROBE).free<180*GIB:
        raise RuntimeError('Insufficient starting memory/disk reserve')
    before=fingerprint();atomic_json(PROBE/'protected-files-before.json',before)
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
        PYTHONPATH=str(TOOLS)+':'+str(PROJECT/'repos/lingbot-vla-v2'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        LINGBOT_TRAINING_APPROVED='1',LINGBOT_SAVE_PROBE_RUN=str(PROBE),LINGBOT_VIDEO_AUDIT=str(PROBE/'video-audit'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',QWEN3VL_PATH=str(PROJECT/'models/Qwen3-VL-4B-Instruct'),
        CUDA_HOME='/usr/local/cuda',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    for key,relative in {'HF_HOME':'cache/huggingface','HF_DATASETS_CACHE':'cache/huggingface/datasets','TORCH_HOME':'cache/torch',
        'TORCH_EXTENSIONS_DIR':'cache/torch_extensions','TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda',
        'XDG_CACHE_HOME':'cache/xdg','MPLCONFIGDIR':'cache/matplotlib','TMPDIR':'cache/tmp'}.items():
        env[key]=str(PROJECT/relative)
    child=None;stop_reason=None;stopped_at=None;minimum=available_memory();statuses=[]
    def stop(signum=None,frame=None):
        nonlocal stop_reason,stopped_at
        if child is not None and child.poll() is None and stopped_at is None:
            stopped_at=time.monotonic();stop_reason=stop_reason or 'stop requested'
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    error=None
    try:
        for mode in ('continue','reload'):
            env['LINGBOT_SAVE_PROBE_MODE']=mode;env['LINGBOT_TRAIN_ATTEMPT']='save-probe-'+mode
            stopped_at=None;started=time.monotonic()
            argv=[str(PROJECT/'envs/vla/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=1',
                '--max-restarts=0',str(TOOLS/'save_probe_entry.py')]
            with (PROBE/(mode+'.log')).open('ab',buffering=0) as log, (PROBE/'memory-samples.jsonl').open('a',buffering=1) as samples:
                child=subprocess.Popen(argv,cwd=PROJECT/'repos/lingbot-vla-v2',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(PROBE/'status.json',{'state':'running','mode':mode,'pid':child.pid,'supervisor_pid':os.getpid()})
                while child.poll() is None:
                    available=available_memory();minimum=min(minimum,available)
                    sample={'unix_time':time.time(),'mode':mode,'unified_mem_available_gib':available/GIB}
                    pidfile=PROBE/'child-pid.json'
                    if pidfile.exists():
                        pidinfo=json.loads(pidfile.read_text());proc=Path('/proc')/str(pidinfo['pid'])/'status'
                        if pidinfo['mode']==mode and proc.exists():
                            status=proc.read_text();sample['rss_kib']=next((int(s.split()[1]) for s in status.splitlines() if s.startswith('VmRSS:')),None)
                    samples.write(json.dumps(sample)+'\n')
                    if available<16*GIB:
                        stop_reason='Unified memory below 16 GiB: '+str(sample);stop()
                    if time.monotonic()-started>1800:
                        stop_reason='Probe child exceeded 30 minute limit';stop()
                    if stopped_at is not None and time.monotonic()-stopped_at>20:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    time.sleep(0.5)
            statuses.append({'mode':mode,'exit_code':child.returncode,'elapsed_seconds':time.monotonic()-started})
            if child.returncode!=0 or stop_reason:
                raise RuntimeError(str(statuses[-1])+' '+str(stop_reason))
            evidence=PROBE/('continuation-result.json' if mode=='continue' else 'reload-result.json')
            if not evidence.exists() or not json.loads(evidence.read_text()).get('passed'):
                raise RuntimeError('Probe did not produce passing evidence: '+mode)
            # torchrun has reaped the GPU worker before the next child is launched.
        if fingerprint()!=before:
            raise RuntimeError('Protected source configuration/checkpoint files changed')
    except BaseException as exc:
        error=repr(exc)
        stop()
        if child is not None and child.poll() is None:
            try:child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
    finally:
        removed=[];reclaimed=0
        checkpoints=PROBE/'checkpoints'
        if checkpoints.exists():
            if checkpoints.is_symlink() or checkpoints.resolve()!=PROBE.resolve()/'checkpoints':
                raise RuntimeError('Unsafe test cleanup root')
            for path in checkpoints.iterdir():
                target=checked_checkpoint(checkpoints,path)
                if target.name not in ('global_step_22','global_step_24'):
                    raise RuntimeError('Unexpected cleanup target: '+str(target))
                reclaimed+=sum(p.stat().st_size for p in target.rglob('*') if p.is_file() and not p.is_symlink())
                shutil.rmtree(target);removed.append(str(target))
        unchanged=fingerprint()==before
        result={'passed':error is None and unchanged,'error':error,'children':statuses,'min_unified_mem_available_gib':minimum/GIB,
            'protected_source_files_unchanged':unchanged,'test_checkpoint_directories_removed':removed,
            'bytes_removed_at_final_cleanup':reclaimed,'original_step20_and_formal_step1000_retained':True,
            'formal_training_started':False}
        atomic_json(PROBE/'result.json',result);atomic_json(PROBE/'status.json',{'state':'passed' if result['passed'] else 'failed',**result})
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
