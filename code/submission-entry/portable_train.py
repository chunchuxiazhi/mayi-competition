"""Relocate the unchanged two-stage GB10 recipe. Default: plan only, never train."""
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from portable_eval import BUNDLE,RUNTIME,CONFIGS,ORIGINAL_CONFIGS,SOURCES,TOOLS,absolute,link_directory
sys.path.insert(0,str(RUNTIME))
from common import atomic_json

OLD='/home/zzj/competitions/lingbot'

def relocate(value,root):
    if isinstance(value,str):return value.replace(OLD,str(root))
    if isinstance(value,dict):return {k:relocate(v,root) for k,v in value.items()}
    if isinstance(value,list):return [relocate(v,root) for v in value]
    return value

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir',type=absolute,required=True)
    p.add_argument('--stage',choices=('10000','20000','both'),default='both')
    p.add_argument('--pretrained',type=absolute,required=True,help='Designated original robbyant/lingbot-vla-v2-6b including depth/ and dino_video/')
    p.add_argument('--data',type=absolute,required=True,help='The exact allowed 50-task clean V3 LeRobot dataset')
    p.add_argument('--qwen-root',type=absolute,required=True)
    p.add_argument('--moge-root',type=absolute,required=True)
    p.add_argument('--vla-python',type=absolute,required=True)
    p.add_argument('--execute',action='store_true')
    return p

def make_config(root,step):
    first=json.loads((CONFIGS/'training_step_10000.json').read_text(encoding='utf-8'))
    cfg=relocate(first,root)
    if step=='20000':cfg['train'].update(output_dir=str(root/'runs/finetune-clean-v3-gb10/clean50-20000-from10000-20261018'),
        max_steps=20000,lr='1e-05',lr_min='5e-06',lr_warmup_ratio=0.0)
    return cfg

def setup(a):
    root=a.workdir.resolve();root.mkdir(parents=True,exist_ok=True)
    for dest,source in [('models/lingbot-vla-v2-6b',a.pretrained),('models/Qwen3-VL-4B-Instruct',a.qwen_root),
                        ('models/moge-2-vitb-normal',a.moge_root),('data/RoboTwin_lerobot_v30',a.data),
                        ('envs/vla',a.vla_python.parent.parent),('repos/lingbot-vla-v2',SOURCES/'lingbot-vla-v2')]:
        link_directory(source,root/dest)
    for rel in ['normalization/robotwin_clean_v3.json','robots/clean-v3-gb10/robotwin.yaml']:
        target=root/'configs'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        body=(ORIGINAL_CONFIGS/rel).read_text(encoding='utf-8').replace(OLD,str(root))
        if target.exists() and target.read_text(encoding='utf-8')!=body:raise ValueError('Existing relocated robot/statistics config differs')
        if not target.exists():target.write_text(body,encoding='utf-8')
    for step in ('10000','20000'):
        name='finetune-clean-v3-gb10' if step=='10000' else 'finetune-clean-v3-gb10-continue20000-20261018'
        src=TOOLS/name;dest=root/'tools'/name
        if not dest.exists():shutil.copytree(src,dest)
        else:
            for p in src.glob('*.py'):
                if not (dest/p.name).is_file() or (dest/p.name).read_bytes()!=p.read_bytes():raise ValueError('Existing training adapter differs')
        cfg=make_config(root,step)
        control=relocate(json.loads((src/f'formal{step}-control.json').read_text(encoding='utf-8')),root)
        control['task_episode_map']=str(CONFIGS/'clean-episode-task-map.json')
        control['native_config_path']=str(dest/f'portable-{step}.json')
        if step=='20000':
            source_run=root/'runs/finetune-clean-v3-gb10/clean50-10000-fixed1000'
            if not (source_run/'frozen-config.json').is_file():continue
            source_cfg=json.loads((source_run/'frozen-config.json').read_text())
            control['continuation'].update(source_run=str(source_run),source_config_sha256=hashlib.sha256((source_run/'frozen-config.json').read_bytes()).hexdigest(),
                source_signature=hashlib.sha256(json.dumps(source_cfg,sort_keys=True).encode()).hexdigest(),
                source_commit_sha256=hashlib.sha256((source_run/'checkpoints/global_step_10000/complete.json').read_bytes()).hexdigest())
        atomic_json(dest/f'portable-{step}.json',cfg)
        atomic_json(dest/f'portable-{step}-control.json',control)
    return root

def run_child(argv,root,env,logpath,minimum_memory=16):
    from bounded_log import start_drain
    from runtime_resources import ResourceGuard
    guard=ResourceGuard(root,memory_floor_gib=minimum_memory,disk_floor_gib=120)
    errors=[];child=subprocess.Popen(argv,cwd=root/'repos/lingbot-vla-v2',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    def stop(signum=None,frame=None):
        if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        if signum is not None:raise KeyboardInterrupt('Training reproduction interrupted')
    def log_error(message):errors.append(message);stop()
    drainer=start_drain(child.stdout,logpath,50*1024**2,log_error)
    previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        while child.poll() is None:
            guard.check()
            if errors:raise OSError(errors[-1])
            time.sleep(2)
        if child.returncode:raise RuntimeError('Child failed; inspect bounded log: '+str(logpath))
    finally:
        if child.poll() is None:
            stop()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        drainer.join(timeout=5)
        for s,h in previous.items():signal.signal(s,h)

def main(argv=None):
    a=parser().parse_args(argv);root=a.workdir.resolve()
    if root==BUNDLE or root.is_relative_to(BUNDLE):raise ValueError('Choose an external reproduction directory')
    stages=('10000','20000') if a.stage=='both' else (a.stage,)
    print(json.dumps({'source':'original pretrained initialization; clean 50-task cotrain','stages':list(stages),
        'optimizer_updates_total':int(stages[-1]),'save_every':1000,'retention':2,
        'runtime_started':False,'new_portable_entry_not_runtime_verified':True,'workdir':str(root)},ensure_ascii=False),flush=True)
    if not a.execute:return 0
    if sys.platform!='linux' or a.vla_python.parent.name!='bin':raise RuntimeError('Use the Linux NVIDIA training environment bin/python')
    for path in [a.pretrained/'model.safetensors.index.json',a.pretrained/'depth/model.pt',a.pretrained/'dino_video/teacher_step_10000.pth',
                 a.qwen_root/'config.json',a.moge_root/'model.pt',a.data/'meta/info.json',a.vla_python]:
        if not path.is_file():raise FileNotFoundError(path)
    if not a.workdir.exists():a.workdir.mkdir(parents=True)
    from runtime_resources import startup_check,thread_env
    startup_check(root,90,required_cpus=4)
    if shutil.disk_usage(root).free<120*2**30:raise OSError('Need 120GiB free disk reserve for reproduction training')
    from prepare_model import idle
    idle()
    import fcntl
    lock=(root/'portable-runtime.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for step in stages:
            setup(a)
            name='finetune-clean-v3-gb10' if step=='10000' else 'finetune-clean-v3-gb10-continue20000-20261018'
            tool=root/'tools'/name;cfgpath=tool/f'portable-{step}.json';ctl=tool/f'portable-{step}-control.json'
            if not cfgpath.is_file():raise FileNotFoundError('Finish stage10000 with full DCP and CPU HF export before continuation')
            cfg=json.loads(cfgpath.read_text());run=Path(cfg['train']['output_dir'])
            complete=run/f'checkpoints/global_step_{step}/complete.json'
            if complete.exists():raise RuntimeError('Requested training stage already completed; refusing extra updates')
            run.mkdir(parents=True,exist_ok=True);(run/'logs').mkdir(exist_ok=True)
            frozen=run/'frozen-config.json'
            if frozen.exists() and json.loads(frozen.read_text())!=cfg:raise ValueError('Frozen training config differs')
            if not frozen.exists():atomic_json(frozen,cfg)
            env=os.environ.copy();env.update(thread_env(root,'train'))
            env.update(PYTHONNOUSERSITE='1',PYTHONPATH=str(tool)+os.pathsep+str(root/'repos/lingbot-vla-v2'),
                HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LINGBOT_TRAINING_APPROVED='1',
                LINGBOT_TRAIN_ATTEMPT=str(time.time_ns()),LINGBOT_VIDEO_AUDIT=str(run/'logs/video-tail'),
                CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/usr/local/cuda',QWEN3VL_PATH=str(a.qwen_root),
                TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            for key in ['LINGBOT_VERIFY_RESUME_ONLY','LINGBOT_SEGMENT_STOP_AT','LINGBOT_EXPECT_RESUME_STEP']:env.pop(key,None)
            argv=[str(a.vla_python),'-B','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--max-restarts=0',
                  str(tool/'train_entry.py'),'--config',str(cfgpath),'--control',str(ctl)]
            run_child(argv,root,env,run/'logs/portable-train.log')
            if not complete.is_file():raise RuntimeError('Training ended without the requested complete checkpoint')
            cpu=env.copy();cpu['CUDA_VISIBLE_DEVICES']=''
            run_child([str(a.vla_python),'-B',str(tool/'export_cpu.py'),'--config',str(cfgpath),'--control',str(ctl)],root,cpu,run/'logs/portable-export.log',28)
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
