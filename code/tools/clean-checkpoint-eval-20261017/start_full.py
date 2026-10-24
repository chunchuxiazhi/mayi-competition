"""Manual detached full-suite entry point. Never called by smoke or model preparation."""
import argparse,hashlib,os,subprocess,time
from common import HERE,load_config,run_path,fingerprint,require_ready
from model_source import verify_source
from prepare_model import idle,available

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start-approved',action='store_true')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--retry-interrupted',action='store_true')
    a=parser.parse_args();cfg=load_config();require_ready(cfg,a.start_approved);run=run_path(cfg)
    smoke=load_config(run/'smoke-result.json')
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    if not smoke['passed'] or smoke['config_sha256']!=fingerprint(cfg) or smoke['model_source_sha256']!=verify_source(cfg) or smoke['code_sha256']!=code:
        raise RuntimeError('Matching successful smoke required')
    subprocess.run(['python3','-B',str(HERE/'static_check.py')],check=True,cwd=HERE)
    idle()
    if available()<85*2**30:raise MemoryError('Need 85 GiB available unified memory; other jobs are left untouched')
    argv=['python3','-B','-u',str(HERE/'driver.py'),'--full-suite','--start-approved','--workers',str(a.workers)]
    if a.retry_interrupted:argv.append('--retry-interrupted')
    with (run/'logs/full-launcher.log').open('ab') as log:
        p=subprocess.Popen(argv,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(2)
    if p.poll() is not None:raise RuntimeError('Launcher exited early; inspect full-launcher.log')
    print(f'Full-suite supervisor dispatched: PID {p.pid}; status: {run}/execution-status.json; no automatic restart')

if __name__=='__main__':main()
