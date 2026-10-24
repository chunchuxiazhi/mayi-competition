"""Future entry point. Starts nothing unless all review gates are satisfied."""
import argparse
import asyncio
import os
from pathlib import Path
import sys
import time
import uuid
from common import load_config,require_ready

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--start-approved',action='store_true')
    args=parser.parse_args();config=load_config(args.config);require_ready(config,args.start_approved)
    from model_source import verify_source
    source_digest=verify_source(config)
    root=Path(config['project_root']);repo=root/'repos/lingbot-vla-v2'
    os.chdir(repo);sys.path.insert(0,str(repo))
    os.environ['QWEN3VL_PATH']=str(root/'models/Qwen3-VL-4B-Instruct')
    os.environ['TRITON_PTXAS_PATH']='/usr/local/cuda/bin/ptxas'
    os.environ['TRITON_CACHE_DIR']=str(root/'cache/triton')
    import torch
    import numpy as np
    import websockets.asyncio.server as ws_server
    import websockets
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from deploy.msgpack_numpy import Packer,unpackb
    from session_policy import SessionPolicy
    torch.set_num_threads(8)
    wrapper=root/config['model_wrapper_relative']
    if wrapper.resolve()!=(root/config['model_relative']).resolve():raise ValueError('Checkpoint wrapper does not resolve to designated step10000 export')
    policy=LingbotVLAv2Server(str(wrapper),robot_norm_path=config['normalization']['path'],
        use_length=config['use_length'],chunk_ret=config['chunk_ret'],use_bf16=False,use_fp32=True,use_compile=False)
    policy.vla.requires_grad_(False);policy.vla.eval()
    sessions=SessionPolicy(policy,torch,np,max_sessions=2)

    async def handler(ws):
        cid=uuid.uuid4().hex
        try:sessions.open(cid)
        except RuntimeError:
            await ws.close(code=1013,reason='Only two simulator sessions allowed');return
        packer=Packer()
        try:
            await ws.send(packer.pack({'model':config['model_repo'],'label':config['label'],'session_id':cid,'automatic_batching':False,'training_step':config['training_step'],'model_source_sha256':source_digest}))
            while True:
                obs=unpackb(await ws.recv());started=time.monotonic()
                # No await inside inference/state swap. One request is fully completed at a time.
                result=sessions.infer(cid,obs)
                result['server_timing']={'infer_ms':(time.monotonic()-started)*1000}
                result['session_id']=cid
                await ws.send(packer.pack(result))
        except websockets.ConnectionClosed:pass
        except Exception as exc:
            await ws.send(str(exc));await ws.close(code=1011,reason='Session failed; no automatic retry')
        finally:sessions.close(cid)

    async def health(connection,request):
        if request.path=='/healthz':
            import http
            return connection.respond(http.HTTPStatus.OK,'OK\n')

    async def run():
        async with ws_server.serve(handler,'127.0.0.1',config['port'],compression=None,max_size=16*1024*1024,
                                   ping_interval=None,ping_timeout=None,process_request=health) as server:
            await server.serve_forever()
    asyncio.run(run())

if __name__=='__main__':main()
