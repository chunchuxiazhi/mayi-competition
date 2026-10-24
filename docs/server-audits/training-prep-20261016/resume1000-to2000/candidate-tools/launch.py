"""Validate by default. Training and CPU export require the explicit --start option."""
import argparse,fcntl,hashlib,json,os,resource,signal,subprocess,sys,threading,time
from pathlib import Path
from storage_policy import GIB,atomic_json,check_space,resolve_schedule,planned_saves

HERE=Path(__file__).resolve().parent

def validate_config(config,control):
    import yaml
    r=Path(control['project_root']).resolve();data=config['data'];train=config['train'];model=config['model']
    if model['model_path']!=str(r/'models/lingbot-vla-v2-6b'):raise ValueError('Only designated original initialization is allowed')
    if data['train_path']!=str(r/'data/RoboTwin_lerobot_v30') or data['data_name']!='robotwin':raise ValueError('Only the reviewed clean V3 dataset is allowed')
    if train['micro_batch_size']!=1 or train['global_batch_size']!=train['gradient_accumulation_steps']:
        raise ValueError('Single-rank micro batch/global batch mismatch')
    if data['num_workers']!=0 or data['prefetch_factor']!=1 or data['pin_memory']:raise ValueError('Use the verified synchronous data-loader profile')
    if train['use_compile'] or train['attention_implementation']!='eager' or model.get('vit_attn_implementation')!='eager' or not train['train_expert_only'] or not train['enable_fp32'] or not train['enable_mixed_precision']:
        raise ValueError('Unreviewed model resource settings')
    if train['save_hf_weights'] or train['async_save_hf_weights'] or config['eval']['enabled']:
        raise ValueError('In-training HF export or evaluation is prohibited')
    if train['data_parallel_mode']!='fsdp2' or train['optimizer']!='adamw':raise ValueError('Unreviewed optimizer/distributed mode')
    if control['world_size']!=1 or control['keep_checkpoints'] not in (1,2):raise ValueError('Unexpected training concurrency/retention')
    points=planned_saves(train['max_steps'],train['save_steps'])
    if control.get('checkpoint_schedule_mode') not in (None,'fixed','pilot_calibrated'):raise ValueError('Unknown save scheduling mode')
    if control.get('checkpoint_schedule_mode')=='fixed' and points!=control['expected_checkpoint_steps']:
        raise ValueError('Fixed checkpoint schedule changed')
    output=Path(train['output_dir']).resolve();training_root=Path(control['training_root']).resolve()
    if not output.is_relative_to(training_root) or not training_root.is_relative_to(r/'runs'):raise ValueError('Output escaped project run directory')
    records=json.loads(Path(control['task_episode_map']).read_text());tasks={row['task'] for row in records}
    if len(tasks)!=50 or set(control['selected_tasks'])!=tasks or len(control['selected_tasks'])!=50:
        raise ValueError('Joint training must include all 50 distinct tasks')
    if len(records)!=2500 or sum(e['length'] for e in records)!=548893:raise ValueError('Unexpected task/frame manifest')
    norm=r/'configs/normalization/robotwin_clean_v3.json'
    if hashlib.sha256(norm.read_bytes()).hexdigest()!=control['normalization_sha256']:raise ValueError('Clean normalization changed')
    robot=yaml.safe_load((Path(data['robot_config_root'])/'robotwin.yaml').read_text())
    if robot['norm_stats']!=str(norm):raise ValueError('Training robot mapping uses unreviewed statistics')
    for path in (model['model_path'],model['tokenizer_path'],data['train_path'],train['align_params']['depth']['moge_path'],
                 train['align_params']['depth']['morgbd_path'],train['align_params']['video']['ckpt_path'],train['align_params']['video']['config_path']):
        if not Path(path).exists():raise FileNotFoundError(path)
    source=r/'repos/lingbot-vla-v2/tasks/vla/train_lingbotvla.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=control['upstream_trainer_sha256']:raise ValueError('Upstream trainer changed; review adapter compatibility')
    return {'passed':True,'tasks':50,'episodes':2500,'frames':548893,'world_size':1,
        'micro_batch':1,'gradient_accumulation':train['gradient_accumulation_steps'],
        'optimizer_updates':train['max_steps'],'save_every_updates':train['save_steps'],
        'checkpoint_keep':control['keep_checkpoints'],'save_schedule_mode':control.get('checkpoint_schedule_mode','legacy_calibrated'),
        'configured_save_steps':points,'training_started':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','stage1','formal10000'],default='pilot')
    parser.add_argument('--stop-at-step',type=int);parser.add_argument('--expected-resume-step',type=int)
    parser.add_argument('--start',action='store_true');cli=parser.parse_args()
    import yaml
    ctl_path=HERE/f'{cli.phase}-control.json';control=json.loads(ctl_path.read_text())
    cfg_path=Path(control['native_config_path']);config=yaml.safe_load(cfg_path.read_text())
    print(json.dumps(validate_config(config,control),indent=2),flush=True)
    segment=None
    if cli.stop_at_step is not None or cli.expected_resume_step is not None:
        if cli.phase!='formal10000':raise ValueError('Segment control is restricted to the reviewed formal run')
        from segment_stop import validate_budget
        segment=validate_budget(config,cli.stop_at_step,cli.expected_resume_step)
        print(json.dumps({'segment':segment}),flush=True)
    if not cli.start:return
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    root=Path(control['project_root']);training_root=Path(control['training_root']);training_root.mkdir(parents=True,exist_ok=True)
    lock=(training_root/'training.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=cli.stop_at_step if segment else config['train']['max_steps']
    planned=Path(config['train']['output_dir'])/'checkpoints'/f"global_step_{target}"/'complete.json'
    if planned.exists():raise RuntimeError('Planned optimizer updates already completed; do not train another step. Inspect/export the existing checkpoint separately.')
    schedule=None
    if cli.phase in ('stage1','formal10000'):
        evidence=training_root/'pilot/reload-verification.json'
        if not evidence.is_file() or not json.loads(evidence.read_text()).get('passed'):
            raise RuntimeError('First pass the 20-update pilot, export and actual model reload')
        pilot=training_root/'pilot'
        pilot_config=json.loads((pilot/'frozen-config.json').read_text())
        pilot_attempt=json.loads(evidence.read_text())['launch_attempt']
        timings=[e for s in (pilot/'logs/updates.jsonl').read_text().splitlines() if s.strip()
                 for e in [json.loads(s)] if e.get('launch_attempt')==pilot_attempt]
        schedule=resolve_schedule(config,control,timings,pilot_config['train']['global_batch_size'])
        config['train']['save_steps']=schedule['save_every_updates']
        validate_config(config,control)
        print(json.dumps({'checkpoint_schedule':schedule}),flush=True)
        if cli.phase=='formal10000':
            evidence_path=Path(control['resume_verification_path'])
            verified=json.loads(evidence_path.read_text())
            if not verified.get('passed') or verified.get('optimizer_steps_executed')!=0:
                raise RuntimeError('Pass the zero-update real checkpoint restoration check before long training')
            if verified['checkpoint_signature']!=json.loads(evidence.read_text())['configuration_signature']:
                raise RuntimeError('Resume verification does not match the reviewed pilot checkpoint')
            if verified['resume_guard_sha256']!=hashlib.sha256((HERE/'resume_guard.py').read_bytes()).hexdigest():
                raise RuntimeError('Resume guard changed after verification')
            probe_status=json.loads((evidence_path.parent/'status.json').read_text())
            if probe_status.get('state')!='passed' or not probe_status.get('optimizer_update_ledger_and_checkpoint_files_unchanged'):
                raise RuntimeError('Resume probe did not finish with unchanged update/checkpoint records')
    if schedule is None and control.get('checkpoint_schedule_mode')=='fixed':schedule=resolve_schedule(config,control)
    active=subprocess.run(['pgrep','-af',str(root)+'/tools/clean-quick-baseline-20261016/(serve|evaluate_task).py'],capture_output=True,text=True).stdout
    if active:raise RuntimeError('Project evaluation is active; training must run separately')
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
    if gpu.returncode or gpu.stdout.strip():raise RuntimeError('GPU already has compute clients or status is unavailable; refusing concurrent training')
    from gb10_runtime import available_memory
    if available_memory()<control['startup_memory_min_gib']*GIB:raise MemoryError('Insufficient free unified memory for startup')
    check_space(control)
    run=Path(config['train']['output_dir']);run.mkdir(parents=True,exist_ok=True);(run/'logs').mkdir(exist_ok=True)
    previous_manifest=run/'launch-manifest.json'
    if previous_manifest.exists() and json.loads(previous_manifest.read_text())['control']!=control:
        raise ValueError('Run resource/retention control changed; review before resuming')
    from gb10_runtime import signature
    frozen=run/'frozen-config.json'
    if frozen.exists() and json.loads(frozen.read_text())!=config:raise ValueError('Existing run configuration differs')
    if not frozen.exists():atomic_json(frozen,config)
    cfg_path=run/'frozen-config.yaml'
    if cfg_path.exists():
        if yaml.safe_load(cfg_path.read_text())!=config:raise ValueError('Frozen YAML configuration differs')
    else:atomic_json(cfg_path,config)
    attempt=str(time.time_ns())
    launch_record={'configuration_signature':signature(config),'control':control,'launch_attempt':attempt,
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        'checkpoint_schedule':schedule,'segment':segment,'model_and_dataset_copied':False,'automatic_evaluation':False,'resumes_same_config_only':True}
    atomic_json(run/'launch-manifest.json',launch_record)
    from gb10_runtime import append
    append(run/'launch-invocations.jsonl',launch_record)
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
        PYTHONPATH=str(HERE)+':'+str(root/'repos/lingbot-vla-v2'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        LINGBOT_TRAINING_APPROVED='1',LINGBOT_VIDEO_AUDIT=str(run/'logs/video-tail'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        LINGBOT_TRAIN_ATTEMPT=attempt,
        TOKENIZERS_PARALLELISM='false',QWEN3VL_PATH=str(root/'models/Qwen3-VL-4B-Instruct'),CUDA_HOME='/usr/local/cuda',
        TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2',PYTHONFAULTHANDLER='1')
    env.pop('LINGBOT_SEGMENT_STOP_AT',None);env.pop('LINGBOT_EXPECT_RESUME_STEP',None)
    if segment:
        env.update(LINGBOT_SEGMENT_STOP_AT=str(cli.stop_at_step),LINGBOT_EXPECT_RESUME_STEP=str(cli.expected_resume_step))
    for key,relative in {'HF_HOME':'cache/huggingface','HF_DATASETS_CACHE':'cache/huggingface/datasets','TORCH_HOME':'cache/torch',
        'TORCH_EXTENSIONS_DIR':'cache/torch_extensions','TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda',
        'XDG_CACHE_HOME':'cache/xdg','MPLCONFIGDIR':'cache/matplotlib','TMPDIR':'cache/tmp'}.items():
        path=root/relative;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    python=str(root/'envs/vla/bin/python')
    argv=[python,'-B','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=1','--max-restarts=0',str(HERE/'train_entry.py'),
          '--config',str(cfg_path),'--control',str(ctl_path)]
    atomic_json(run/'execution-status.json',{'state':'training','phase':cli.phase,'automatic_restart':False})
    child=subprocess.Popen(argv,cwd=root/'repos/lingbot-vla-v2',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    reason=None;signalled=False;stop_time=None
    def stop(signum=None,frame=None):
        nonlocal reason,signalled,stop_time
        if signalled:return
        signalled=True;stop_time=time.monotonic();reason=reason or 'User/process stop requested'
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def drain():
        path=run/'logs/train.log';kept=path.stat().st_size if path.exists() else 0
        with path.open('ab') as log:
            while chunk:=child.stdout.read1(8192):
                room=max(0,50*1024**2-kept)
                if room:log.write(chunk[:room]);log.flush();kept+=min(room,len(chunk))
    thread=threading.Thread(target=drain,daemon=True);thread.start()
    while child.poll() is None:
        current_free=available_memory()
        if current_free<control['memory_free_floor_gib']*GIB:
            atomic_json(run/'resource-stop-details.json',{'unix_time':time.time(),'unified_mem_available_gib':current_free/GIB,
                'floor_gib':control['memory_free_floor_gib'],'launch_attempt':attempt})
            reason='Unified memory fell below reserve; stopped without emergency HF export';stop()
        if stop_time is not None and time.monotonic()-stop_time>20:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        time.sleep(2)
    thread.join(timeout=10)
    if child.returncode or reason:
        atomic_json(run/'execution-status.json',{'state':'stopped','exit_code':child.returncode,'reason':reason,'automatic_restart':False})
        raise SystemExit(child.returncode or 1)
    if segment:
        from segment_stop import verify_completion
        completion=verify_completion(run,config,cli.stop_at_step,attempt)
        atomic_json(run/'execution-status.json',completion)
        raise SystemExit(0)
    # The entire training process group has exited before the export process starts.
    atomic_json(run/'execution-status.json',{'state':'exporting_on_cpu','phase':cli.phase})
    export_env=env.copy();export_env['CUDA_VISIBLE_DEVICES']=''
    with (run/'logs/export.log').open('ab') as log:
        result=subprocess.run([python,'-B',str(HERE/'export_cpu.py'),'--config',str(cfg_path),'--control',str(ctl_path)],
            cwd=root/'repos/lingbot-vla-v2',env=export_env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        atomic_json(run/'execution-status.json',{'state':'export_failed','exit_code':result.returncode,'evaluation_started':False})
        raise SystemExit(result.returncode)
    # Third process: actual strict reload, with no simulation or action inference.
    atomic_json(run/'execution-status.json',{'state':'verifying_model_reload','phase':cli.phase})
    with (run/'logs/reload.log').open('ab') as log:
        result=subprocess.run([python,'-B',str(HERE/'verify_reload.py'),'--config',str(cfg_path),'--control',str(ctl_path)],
            cwd=root/'repos/lingbot-vla-v2',env=env,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(run/'execution-status.json',{'state':'training_export_reload_complete' if result.returncode==0 else 'reload_failed',
        'reload_exit_code':result.returncode,'evaluation_started':False,'actual_model_reload_verified':result.returncode==0})
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
