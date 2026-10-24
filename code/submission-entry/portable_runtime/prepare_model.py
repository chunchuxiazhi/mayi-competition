"""Resource metadata only. This entry never exports or loads a model."""
import subprocess
from runtime_resources import available_memory

def available():return available_memory()

def idle():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=10)
    if result.stdout.strip():raise RuntimeError('GPU is occupied; existing processes are left untouched')
