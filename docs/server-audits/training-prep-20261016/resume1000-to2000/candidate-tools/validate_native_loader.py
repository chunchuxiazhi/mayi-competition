"""Check the official normalized DataLoader's accumulation batches and serializable state, CPU only."""
import sys,json,os
from pathlib import Path

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','formal10000'],default='pilot');phase=parser.parse_args().phase
    import torch
    torch.set_num_threads(1);torch.multiprocessing.set_start_method('spawn',force=True)
    import safe_video
    safe_video.install()
    from tasks.vla.train_lingbotvla import Arguments,parse_args
    from lingbotvla.models import build_processor
    from lingbotvla.models.config_registry import get_config_registry
    from lingbotvla.data import build_vla_dataset,build_dataloader,VLADataCollatorWithPacking
    import lingbotvla.data.data_loader as loader_module
    from types import SimpleNamespace
    # Test the known one-rank layout without initializing NCCL or loading a model.
    loader_module.get_parallel_state=lambda:SimpleNamespace(dp_size=1,dp_rank=0,sp_size=1,sp_enabled=False)
    r=Path('/home/zzj/competitions/lingbot');sys.argv=['check',str(r/f'configs/training/clean-v3-gb10/{phase}.yaml')]
    args=parse_args(Arguments)
    config=get_config_registry().get_config_cls_from_config_key(args.model.config_key)(**{**vars(args.model),**vars(args.train)})
    processor=build_processor(args.model.tokenizer_path);args.data.chunk_size=args.train.chunk_size
    dataset=build_vla_dataset(args.data,args.model,config,processor,use_depth_align=True)
    args.train.compute_train_steps(args.data.max_seq_len,args.data.train_size,len(dataset))
    loader=build_dataloader(dataset=dataset,micro_batch_size=args.train.micro_batch_size,global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,seed=args.train.seed,collate_fn=[VLADataCollatorWithPacking()],
        max_seq_len=args.data.max_seq_len,train_steps=args.train.train_steps,rmpad=False,rmpad_with_pos_ids=False,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,dyn_bsz_margin=args.train.dyn_bsz_margin,dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,drop_last=True,pin_memory=False,
        prefetch_factor=args.data.prefetch_factor if args.data.num_workers>0 else None)
    iterator=iter(loader);micro=next(iterator)
    assert len(micro)==args.train.gradient_accumulation_steps
    for item in micro:assert tuple(item['actions'].shape)==(1,50,55)
    state=loader.state_dict()
    import io
    stream=io.BytesIO();torch.save(state,stream)
    shutdown=getattr(iterator,'_shutdown_workers',None)
    if shutdown is not None:shutdown()
    exitcodes=[w.exitcode for w in getattr(iterator,'_workers',[])]
    assert exitcodes==[],exitcodes
    print(json.dumps({'passed':True,'native_micro_batches':len(micro),'normalized_actions_shape':list(micro[0]['actions'].shape),
        'serialized_loader_state_bytes':stream.tell(),'layout_stub':'one rank, CPU data-only check; FSDP not executed',
        'data_workers':args.data.num_workers,'worker_exit_codes':exitcodes,
        'model_weights_loaded':False,'optimizer_updates':0,'simulation_started':False}))
    del iterator,loader

if __name__=='__main__':main()
