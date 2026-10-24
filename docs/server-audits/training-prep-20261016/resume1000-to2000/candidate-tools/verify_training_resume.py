"""Explicit zero-update resume verification of the committed 20-step pilot.

This starts one torchrun child, restores real model/Adam/loader/RNG state, and
exits at the guarded boundary before any model forward or optimizer update.
"""
from pathlib import Path
import argparse,fcntl,json,os,resource,signal,subprocess,threading,time
from storage_policy import GIB,atomic_json

HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-check',action='store_true');cli=parser.parse_args()
    if not cli.run_check:raise SystemExit('Use --run-check explicitly; this check loads a model but executes zero updates')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    ctl=json.loads((HERE/'pilot-control.json').read_text());r=Path(ctl['project_root']);root=Path(ctl['training_root'])
    lock=(root/'training.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
    if gpu.returncode or gpu.stdout.strip():raise RuntimeError('GPU has another compute client; do not overlap the probe')
    from gb10_runtime import available_memory
    if available_memory()<90*GIB:raise MemoryError('Insufficient unified-memory reserve')
    probe=root/'resume-verification';probe.mkdir(exist_ok=True);result=probe/'result.json'
    if result.exists():raise RuntimeError('Resume probe evidence already exists; inspect rather than overwrite')
    pilot=root/'pilot';update_file=pilot/'logs/updates.jsonl';before=update_file.read_bytes()
    checkpoint_files={str(p):p.stat().st_size for p in (pilot/'checkpoints').rglob('*') if p.is_file()}
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
        PYTHONPATH=str(HERE)+':'+str(r/'repos/lingbot-vla-v2'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        LINGBOT_TRAINING_APPROVED='1',LINGBOT_VERIFY_RESUME_ONLY='1',LINGBOT_RESUME_PROBE_RESULT=str(result),
        LINGBOT_TRAIN_ATTEMPT='resume-check-'+str(time.time_ns()),LINGBOT_VIDEO_AUDIT=str(probe/'video-audit'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',QWEN3VL_PATH=str(r/'models/Qwen3-VL-4B-Instruct'),
        CUDA_HOME='/usr/local/cuda',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    for key,relative in {'HF_HOME':'cache/huggingface','HF_DATASETS_CACHE':'cache/huggingface/datasets','TORCH_HOME':'cache/torch',
        'TORCH_EXTENSIONS_DIR':'cache/torch_extensions','TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda',
        'XDG_CACHE_HOME':'cache/xdg','MPLCONFIGDIR':'cache/matplotlib','TMPDIR':'cache/tmp'}.items():env[key]=str(r/relative)
    argv=[str(r/'envs/vla/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=1',
        '--max-restarts=0',str(HERE/'train_entry.py'),'--config',str(pilot/'frozen-config.yaml'),'--control',str(HERE/'pilot-control.json')]
    atomic_json(probe/'status.json',{'state':'restoring','optimizer_updates_allowed':0})
    with (probe/'probe.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(argv,cwd=r/'repos/lingbot-vla-v2',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        reason=None;stop_at=None
        def stop(signum=None,frame=None):
            nonlocal stop_at,reason
            if stop_at is None:
                stop_at=time.monotonic();reason=reason or 'stop requested'
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        while child.poll() is None:
            if available_memory()<16*GIB:reason='Unified-memory reserve reached';stop()
            if stop_at is not None and time.monotonic()-stop_at>20:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            time.sleep(2)
    after_files={str(p):p.stat().st_size for p in (pilot/'checkpoints').rglob('*') if p.is_file()}
    unchanged=before==update_file.read_bytes() and checkpoint_files==after_files
    passed=child.returncode==0 and reason is None and unchanged and result.is_file() and json.loads(result.read_text()).get('passed')
    atomic_json(probe/'status.json',{'state':'passed' if passed else 'failed','exit_code':child.returncode,'reason':reason,
        'optimizer_update_ledger_and_checkpoint_files_unchanged':unchanged,'optimizer_steps_executed':0 if unchanged else None})
    if not passed:raise SystemExit('Zero-update resume verification failed; inspect '+str(probe/'probe.log'))
    print(result.read_text())
if __name__=='__main__':main()
