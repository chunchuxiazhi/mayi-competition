"""Bounded host resource checks; imports no numerical or simulation libraries."""
import os,shutil,subprocess,time
from pathlib import Path

GIB=2**30
CACHE_NAMES=('torch_extensions','triton','cuda','xdg','tmp','torch','huggingface','inductor','optix')

def cgroup_paths():
    root=Path('/sys/fs/cgroup');paths=[root]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            current=root/line.split('::',1)[1].lstrip('/')
            if current.is_dir() and current.resolve().is_relative_to(root.resolve()):
                while current!=root:paths.append(current);current=current.parent
    return paths

def available_memory():
    amount=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    for group in cgroup_paths():
        maximum=group/'memory.max';current=group/'memory.current'
        if maximum.is_file() and current.is_file() and maximum.read_text().strip()!='max':
            amount=min(amount,max(0,int(maximum.read_text())-int(current.read_text())))
    return amount

def effective_cpus():
    count=len(os.sched_getaffinity(0))
    for group in cgroup_paths():
        path=group/'cpu.max'
        if path.is_file():
            quota,period=path.read_text().split()
            if quota!='max':count=min(count,float(quota)/float(period))
    return count

def thread_env(root,role):
    env={k:str(v) for k,v in {'OMP_NUM_THREADS':8 if role=='inference' else 4,'MKL_NUM_THREADS':8 if role=='inference' else 4,
        'OPENBLAS_NUM_THREADS':1,'NUMEXPR_NUM_THREADS':1,'TOKENIZERS_PARALLELISM':'false',
        'CUDA_CACHE_MAXSIZE':512*1024**2,'PYTHONDONTWRITEBYTECODE':1,'PYTHONUNBUFFERED':1}.items()}
    paths={'HF_HOME':'huggingface','HF_DATASETS_CACHE':'huggingface/datasets','TORCH_HOME':'torch',
           'TORCH_EXTENSIONS_DIR':'torch_extensions','TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda',
           'TORCHINDUCTOR_CACHE_DIR':'inductor','OPTIX_CACHE_PATH':'optix','XDG_CACHE_HOME':'xdg',
           'MPLCONFIGDIR':'matplotlib','TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp'}
    for key,name in paths.items():
        path=Path(root)/'cache'/name;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    return env

def cache_bytes(root):
    paths=[Path(root)/'cache'/name for name in CACHE_NAMES if (Path(root)/'cache'/name).is_dir()]
    if not paths:return 0
    result=subprocess.run(['du','-sx','-B1',*[str(p) for p in paths]],capture_output=True,text=True,timeout=15,check=True)
    return sum(int(line.split()[0]) for line in result.stdout.splitlines())

def suite_cache_baseline(root,suite):
    from common import atomic_json
    import json
    path=Path(suite)/'preparation/resource-baseline.json'
    if not path.is_file():atomic_json(path,{'cache_allocated_bytes':cache_bytes(root),'created_unix':time.time()})
    return int(json.loads(path.read_text())['cache_allocated_bytes'])

def host_snapshot(root):
    fs=os.statvfs(root)
    return {'unified_mem_available_bytes':available_memory(),'disk_free_bytes':shutil.disk_usage(root).free,
            'free_inodes':fs.f_favail,'effective_cpus':effective_cpus(),'load_average':list(os.getloadavg())}

class ResourceGuard:
    def __init__(self,root,baseline_cache_bytes=None,memory_floor_gib=16,disk_floor_gib=8,cache_growth_gib=4):
        self.root=Path(root);self.memory_floor=memory_floor_gib*GIB;self.disk_floor=disk_floor_gib*GIB
        self.cache_growth=cache_growth_gib*GIB;self.baseline_cache=cache_bytes(root) if baseline_cache_bytes is None else baseline_cache_bytes
        self.last_cache=self.baseline_cache;self.next_cache_check=0;self.minimum_memory=available_memory()
    def check(self):
        snapshot=host_snapshot(self.root);self.minimum_memory=min(self.minimum_memory,snapshot['unified_mem_available_bytes'])
        if snapshot['unified_mem_available_bytes']<self.memory_floor:raise MemoryError('Unified memory reserve below 16GiB')
        if snapshot['disk_free_bytes']<self.disk_floor:raise OSError('Project disk free reserve below 8GiB; preserve existing results')
        if snapshot['free_inodes']<10000:raise OSError('Filesystem inode reserve below 10000')
        now=time.monotonic()
        if now>=self.next_cache_check:
            self.last_cache=cache_bytes(self.root);self.next_cache_check=now+60
            if self.last_cache-self.baseline_cache>self.cache_growth:raise OSError('Project cache grew more than 4GiB during this suite')
        snapshot.update(cache_allocated_bytes=self.last_cache,cache_growth_bytes=self.last_cache-self.baseline_cache)
        return snapshot

def startup_check(root,minimum_memory_gib,required_cpus=16):
    snapshot=host_snapshot(root)
    if snapshot['effective_cpus']<required_cpus:raise RuntimeError(f'This CPU profile requires at least {required_cpus} effective CPUs')
    if snapshot['unified_mem_available_bytes']<minimum_memory_gib*GIB:raise MemoryError('Insufficient available unified memory for startup')
    if snapshot['disk_free_bytes']<40*GIB:raise OSError('Need 40GiB disk reserve before preparing/starting evaluation')
    if snapshot['free_inodes']<10000:raise OSError('Insufficient inode reserve')
    return snapshot
