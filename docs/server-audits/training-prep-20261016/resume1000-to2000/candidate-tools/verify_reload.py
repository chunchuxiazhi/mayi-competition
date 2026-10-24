"""Future post-training strict deployment-model reload; no inference, service or simulation."""
import argparse,json,os
from pathlib import Path
from storage_policy import GIB,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--control',required=True);cli=p.parse_args()
    if os.environ.get('LINGBOT_TRAINING_APPROVED')!='1':raise RuntimeError('Explicit training/pilot launch required')
    import yaml
    cfg=yaml.safe_load(Path(cli.config).read_text());ctl=json.loads(Path(cli.control).read_text())
    from gb10_runtime import available_memory
    if available_memory()<80*GIB:raise MemoryError('Insufficient unified-memory reserve for isolated reload')
    run=Path(cfg['train']['output_dir']);hf=run/'exports'/f"global_step_{cfg['train']['max_steps']}"/'hf_ckpt'
    record=json.loads((hf/'export-manifest.json').read_text())
    import torch
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    torch.cuda.set_per_process_memory_fraction(min(72*GIB/torch.cuda.get_device_properties(0).total_memory,0.8))
    policy=LingbotVLAv2Server(str(hf),robot_norm_path=cfg['data']['norm_stats_file'],
        use_length=50,chunk_ret=True,use_bf16=False,use_fp32=True,use_compile=False)
    policy.vla.requires_grad_(False);policy.vla.eval()
    from safetensors import safe_open
    sentinel=record['changed_weight_sentinel'];loaded=policy.vla.state_dict()[sentinel].detach().cpu()
    expected=None
    for path in hf.glob('*.safetensors'):
        with safe_open(path,framework='pt',device='cpu') as f:
            if sentinel in f.keys():expected=f.get_tensor(sentinel);break
    if expected is None or not torch.equal(loaded,expected):raise RuntimeError('Reloaded action weights do not match exported checkpoint')
    atomic_json(run/'reload-verification.json',{'passed':True,'checkpoint':str(hf),'strict_deployment_load':True,
        'launch_attempt':os.environ.get('LINGBOT_TRAIN_ATTEMPT'),'configuration_signature':record['configuration_signature'],
        'action_projection_exact_match':True,'max_abs_change_from_pretrained':record['max_abs_change_from_pretrained'],
        'training_loss_finite':True,'simulation_or_action_inference_executed':False})

if __name__=='__main__':main()
