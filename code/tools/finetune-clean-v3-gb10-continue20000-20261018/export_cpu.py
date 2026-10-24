"""CPU-only, sequential export; never keeps a training model resident."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
from storage_policy import GIB,atomic_json,check_space,validate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--control',required=True)
    cli=parser.parse_args()
    import yaml
    config=yaml.safe_load(Path(cli.config).read_text());control=json.loads(Path(cli.control).read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Export must have GPU visibility disabled')
    from gb10_runtime import signature,available_memory
    if available_memory()<56*GIB:raise MemoryError('Insufficient CPU/unified-memory reserve for sequential export')
    budget={**control,'checkpoint_reserve_gib':32};check_space(budget)
    run=Path(config['train']['output_dir']);root=run/'checkpoints';expected=signature(config)
    completed=sorted((p for p in root.glob('global_step_*') if (p/'complete.json').is_file()),key=lambda p:int(p.name.rsplit('_',1)[1]))
    if not completed:raise RuntimeError('No committed checkpoint to export')
    checkpoint=completed[-1];validate(root,checkpoint,expected)
    if int(checkpoint.name.rsplit('_',1)[1])!=config['train']['max_steps']:raise RuntimeError('Final planned optimizer update not reached')
    import torch
    from lingbotvla.checkpoint import ckpt_to_state_dict
    from lingbotvla.models import save_model_weights
    destination=run/'exports'/checkpoint.name/'hf_ckpt';temporary=destination.with_name('hf_ckpt.incomplete')
    if destination.exists() or temporary.exists():raise RuntimeError('Export already exists or incomplete; preserve for review')
    temporary.parent.mkdir(parents=True,exist_ok=True)
    state=ckpt_to_state_dict(str(checkpoint),str(run),ckpt_manager='dcp')
    save_model_weights(str(temporary),state,save_dtype=torch.float32,model_assets=None)
    del state
    assets=run/'model_assets'
    if not (assets/'config.json').is_file():raise RuntimeError('Saved model configuration missing')
    if any(p.suffix in ('.safetensors','.pt','.pth','.bin') for p in assets.rglob('*') if p.is_file()):
        raise RuntimeError('Unexpected weight files in model_assets')
    shutil.copytree(assets,temporary,dirs_exist_ok=True)
    files=list(temporary.glob('*.safetensors'))
    if not files:raise RuntimeError('Export produced no safetensors')
    index=temporary/'model.safetensors.index.json'
    if index.exists():
        expected_files=set(json.loads(index.read_text())['weight_map'].values())
        if expected_files!={p.name for p in files}:raise RuntimeError('HF shard/index mismatch')
    manifest={}
    for p in files:
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        manifest[p.name]={'bytes':p.stat().st_size,'sha256':digest}
    from safetensors import safe_open
    sentinel='model.action_out_proj.weight'
    def read_tensor(folder):
        for p in Path(folder).glob('*.safetensors'):
            with safe_open(p,framework='pt',device='cpu') as f:
                if sentinel in f.keys():return f.get_tensor(sentinel)
        raise RuntimeError('Action output projection sentinel missing')
    difference=float((read_tensor(temporary)-read_tensor(config['model']['model_path'])).abs().max())
    if not difference>0:raise RuntimeError('Pilot did not change the action output projection weights')
    atomic_json(temporary/'export-manifest.json',{'configuration_signature':expected,'checkpoint':str(checkpoint),
        'weights':manifest,'changed_weight_sentinel':sentinel,'max_abs_change_from_pretrained':difference,
        'static_export_integrity_checked':True,'actual_model_reload_verified':False})
    os.replace(temporary,destination)
    print(json.dumps({'exported':str(destination),'training_process_not_resident':True,'actual_model_reload_verified':False}))

if __name__=='__main__':main()
