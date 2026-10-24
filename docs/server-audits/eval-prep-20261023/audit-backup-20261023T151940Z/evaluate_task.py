"""Prepared dual-setting/seen client; imports simulator only AFTER explicit gates."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import uuid
from common import load_config,require_ready,run_path,seeds,append_event,read_events,summarize
from recovery import TaskIncomplete,TaskSliceCompleted,recorded_expert,generate_seen,global_resource_fault

CONFIG=None
WORKER_ID=None
RETRY_INTERRUPTED=False
EPISODE_INDICES=None
STOPPING=False

def stop_requested(signum,frame):
    global STOPPING
    if STOPPING:return
    STOPPING=True
    raise KeyboardInterrupt(f'Stop requested by signal {signum}')

def eval_policy(task_name,TASK_ENV,args,model,st_seed,test_num=None,video_size=None,
                video_fps='10',instruction_type=None,usr_args=None):
    import fcntl
    import numpy as np
    import torch
    from generate_episode_instructions import generate_episode_descriptions
    config=CONFIG
    if test_num!=config['test_num'] or instruction_type!='seen':raise ValueError('Unexpected episode count or instruction type')
    directory=run_path(config)/'tasks'/task_name
    directory.mkdir(parents=True,exist_ok=True)
    policy_file=directory/'episodes.jsonl';expert_file=directory/'expert.jsonl'
    task_index=config['tasks'].index(task_name)
    lock_file=(directory/'task.lock').open('a')
    fcntl.flock(lock_file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.update(eval_mode=True,save_data=False,collect_data=False,eval_video_log=False)
    args.pop('eval_video_save_dir',None)
    args['save_path']=str(directory/'unused-observation-output')
    TASK_ENV.suc=0;TASK_ENV.test_num=0
    try:
        events=read_events(policy_file)
        starts={e['attempt_id']:e for e in events if e['event']=='policy_started'}
        ends={e['attempt_id'] for e in events if e['event'] in ('policy_completed','policy_exception','policy_interrupted')}
        for attempt,event in starts.items():
            if attempt not in ends:
                ready=[e for e in events if e.get('attempt_id')==attempt and e['event']=='policy_instruction_ready']
                append_event(policy_file,{**(ready[-1] if ready else event),'event':'policy_interrupted','success':None,
                    'exception':'Prior process ended without terminal event; steps unknown','execution_steps':None})
        incomplete=[]
        for index in (EPISODE_INDICES if EPISODE_INDICES is not None else range(test_num)):
            scene_seeds=seeds(config,task_index,index)
            base={'task':task_name,'setting':config['task_config'],'instruction_type':config['instruction_type'],'instruction':None,
                  'episode_index':index,'worker_id':WORKER_ID,**scene_seeds,
                  'execution_steps':0,'elapsed_seconds':0.0,'exception':None,'success':None}
            events=read_events(policy_file)
            prior=[e for e in events if e['episode_index']==index]
            terminals=[e for e in prior if e['event'] in ('policy_completed','policy_exception')]
            if len(terminals)>1:raise RuntimeError('Duplicate episode terminal')
            if terminals:
                if terminals[0]['event']=='policy_exception':
                    if terminals[0].get('failure_scope')!='task':raise RuntimeError('Prior global policy fault requires review')
                    incomplete.append(index)
                continue
            if any(e['event']=='policy_interrupted' for e in prior) and not RETRY_INTERRUPTED:
                raise RuntimeError('Interrupted episode preserved; use explicit --retry-interrupted for same seed only')
            expert_events=[e for e in read_events(expert_file) if e['seed']==scene_seeds['seed']]
            expert=recorded_expert(expert_events,base)
            if expert:
                info=expert.get('episode_info')
            else:
                attempts=sum(e['event']=='expert_started' for e in expert_events)
                if any(e['event']=='expert_exception' for e in expert_events):
                    if any(e.get('failure_scope')=='global' for e in expert_events):raise RuntimeError('Prior global expert fault requires review')
                    incomplete.append(index);continue
                if attempts and not RETRY_INTERRUPTED:raise RuntimeError('Unfinished expert attempt requires explicit review/retry')
                if attempts>=config['expert_attempt_limit_per_scene']:raise RuntimeError('Expert local attempt guard reached; review global quota scope')
                expert_id=uuid.uuid4().hex;start=time.monotonic();expert_ok=False;info=None
                expert_base={**base,'attempt_id':expert_id,'local_expert_attempt':attempts+1}
                append_event(expert_file,{**expert_base,'event':'expert_started'})
                render_freq=args['render_freq'];args['render_freq']=0
                try:
                    random.seed(scene_seeds['seed'])
                    TASK_ENV.setup_demo(now_ep_num=index,seed=scene_seeds['seed'],is_test=True,**args)
                    episode_info=TASK_ENV.play_once()
                    # Preserve official expert feasibility/success checks. No scene or success changes.
                    TASK_ENV.close_env()
                    expert_ok=bool(TASK_ENV.plan_success and TASK_ENV.check_success())
                    info=episode_info['info']
                    append_event(expert_file,{**expert_base,'event':'expert_passed' if expert_ok else 'expert_failed',
                        'elapsed_seconds':time.monotonic()-start,'episode_info':info,
                        'expert_success':expert_ok})
                except BaseException as exc:
                    append_event(expert_file,{**expert_base,'event':'expert_interrupted' if isinstance(exc,KeyboardInterrupt) else 'expert_exception',
                        'failure_scope':'global' if global_resource_fault(exc) else 'task',
                        'elapsed_seconds':time.monotonic()-start,'exception':traceback.format_exc()[-16000:]})
                    if not isinstance(exc,KeyboardInterrupt):
                        append_event(policy_file,{**base,'event':'episode_blocked','attempt_id':expert_id,
                            'exception':'Expert precheck raised; policy round not executed and seed not substituted: '+str(exc)[:4000]})
                    try:TASK_ENV.close_env()
                    except Exception:pass
                    if isinstance(exc,KeyboardInterrupt):raise
                    if global_resource_fault(exc):raise
                    incomplete.append(index)
                    continue
                finally:args['render_freq']=render_freq
                expert={'attempt_id':expert_id,'expert_success':expert_ok}
            try:
                instruction=generate_seen(task_name,info,scene_seeds['instruction_seed'],config['instruction_candidates'],
                    Path(config['project_root'])/'repos/RoboTwin')
            except TaskIncomplete as exc:
                append_event(policy_file,{**base,'event':'episode_blocked','attempt_id':uuid.uuid4().hex,
                    'block_reason':'metadata_invalid','exception':str(exc)})
                incomplete.append(index);continue
            from model_source import expected_episode
            expected_episode(config,task_name,index,scene_seeds,instruction)
            prior_instructions=[e['instruction'] for e in prior if e.get('instruction')]
            if prior_instructions and any(s!=instruction for s in prior_instructions):raise RuntimeError('Resume instruction drift; do not run')
            append_event(policy_file,{**base,'event':'expert_metadata_validated','attempt_id':uuid.uuid4().hex,
                'expert_attempt_id':expert['attempt_id'],'expert_success':expert['expert_success'],
                'instruction':instruction,'historical_block_released':any(e['event']=='episode_blocked' for e in prior)})
            attempt=uuid.uuid4().hex;episode={**base,'attempt_id':attempt,'instruction':instruction};started=time.monotonic()
            append_event(policy_file,{**episode,'event':'policy_started'})
            outcome=None
            try:
                random.seed(scene_seeds['seed'])
                try:
                    TASK_ENV.setup_demo(now_ep_num=index,seed=scene_seeds['seed'],is_test=True,**args)
                except Exception as exc:
                    if global_resource_fault(exc):raise
                    raise TaskIncomplete('Scene setup failed: '+str(exc)) from exc
                episode['instruction']=instruction
                append_event(policy_file,{**episode,'event':'policy_instruction_ready'})
                TASK_ENV.set_instruction(instruction=instruction)
                from model_source import verify_source
                metadata=model.get_server_metadata();session_id=metadata['session_id']
                if metadata.get('training_step')!=config['training_step'] or metadata.get('model_source_sha256')!=verify_source(config):
                    raise RuntimeError('Server is not the verified configured checkpoint')
                reset=model.infer({'reset':True,'robo_name':'robotwin','episode_seed':scene_seeds['policy_seed']})
                if reset['session_id']!=session_id:raise RuntimeError('Reset session mismatch')
                if reset['_session_state']!={'global_step':0,'episode_seed':scene_seeds['policy_seed']}:raise RuntimeError('Reset state mismatch')
                barrier=os.environ.get('LINGBOT_SMOKE_BARRIER_DIR')
                if barrier:
                    from common import atomic_json
                    bd=Path(barrier);bd.mkdir(parents=True,exist_ok=True)
                    atomic_json(bd/f'worker-{WORKER_ID}.json',{'session_id':session_id,'task':task_name,'policy_seed':scene_seeds['policy_seed']})
                    deadline=time.monotonic()+180
                    while len(list(bd.glob('worker-*.json')))<2:
                        if time.monotonic()>deadline:raise TimeoutError('Second smoke session did not reset in time')
                        time.sleep(.2)
                expected_chunk=0
                success=False
                while TASK_ENV.take_action_cnt<TASK_ENV.step_lim and not success:
                    observation=TASK_ENV.get_obs()
                    formatted={
                        'observation.images.cam_high':observation['observation']['head_camera']['rgb'],
                        'observation.images.cam_left_wrist':observation['observation']['left_camera']['rgb'],
                        'observation.images.cam_right_wrist':observation['observation']['right_camera']['rgb'],
                        'observation.state':observation['joint_action']['vector'],'task':TASK_ENV.get_instruction()}
                    reply=model.infer(formatted)
                    if reply['session_id']!=session_id:raise RuntimeError('Inference session mismatch')
                    expected_chunk+=1
                    if reply['_session_state']!={'global_step':expected_chunk,'episode_seed':scene_seeds['policy_seed']}:
                        raise RuntimeError('Session counter or RNG seed changed across clients')
                    action=reply['action']
                    if expected_chunk==1:
                        append_event(policy_file,{**episode,'event':'policy_first_action_verified','session_id':session_id,
                            'action_shape':list(action.shape),'session_state':reply['_session_state'],
                            'elapsed_seconds':time.monotonic()-started})
                    actions=action if len(action.shape)==2 else [action]
                    for act in actions:
                        TASK_ENV.take_action(act)
                        if TASK_ENV.eval_success:success=True;break
                outcome={**episode,'event':'policy_completed','success':bool(success),
                         'session_id':session_id,'verified_action_chunks':expected_chunk,
                         'execution_steps':TASK_ENV.take_action_cnt,'step_limit':TASK_ENV.step_lim,
                         'elapsed_seconds':time.monotonic()-started,'exception':None}
            except BaseException as exc:
                outcome={**episode,'event':'policy_interrupted' if isinstance(exc,KeyboardInterrupt) else 'policy_exception',
                    'failure_scope':'task' if isinstance(exc,TaskIncomplete) else 'global',
                    'success':None,'execution_steps':getattr(TASK_ENV,'take_action_cnt',None),
                    'step_limit':getattr(TASK_ENV,'step_lim',None),'elapsed_seconds':time.monotonic()-started,
                    'exception':traceback.format_exc()[-16000:]}
                append_event(policy_file,outcome);summarize(config,task_name)
                if isinstance(exc,TaskIncomplete):incomplete.append(index)
                else:raise
            else:
                # Durably record the outcome before environment cleanup or starting another round.
                append_event(policy_file,outcome)
            finally:
                try:TASK_ENV.close_env(clear_cache=((index+2)%args['clear_cache_freq']==0))
                except Exception:
                    append_event(policy_file,{**episode,'event':'cleanup_exception','exception':traceback.format_exc()[-16000:]})
                    raise
            row=summarize(config,task_name)
            TASK_ENV.suc=row['successes'];TASK_ENV.test_num=row['completed_valid_policy_episodes']
            print(f"{task_name} completed {TASK_ENV.test_num}/{test_num}; successes {TASK_ENV.suc}/{TASK_ENV.test_num}",flush=True)
        row=summarize(config,task_name)
        if incomplete:raise TaskIncomplete(f'{task_name}: unresolved slots {incomplete}; no seed substitution')
        if EPISODE_INDICES is not None:raise TaskSliceCompleted()
        raise TaskSliceCompleted()  # prevent legacy result text export with a planned-count denominator
    finally:
        summarize(config,task_name)
        model._ws.close()
        fcntl.flock(lock_file.fileno(),fcntl.LOCK_UN);lock_file.close()

def main():
    global CONFIG,WORKER_ID,RETRY_INTERRUPTED,EPISODE_INDICES
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--task',required=True)
    parser.add_argument('--worker-id',type=int,required=True);parser.add_argument('--start-approved',action='store_true')
    parser.add_argument('--retry-interrupted',action='store_true')
    parser.add_argument('--episode-indices',type=int,nargs='+')
    cli=parser.parse_args();CONFIG=load_config(cli.config);require_ready(CONFIG,cli.start_approved)
    if cli.task not in CONFIG['tasks'] or cli.worker_id not in (0,1):raise ValueError('Unexpected task or worker')
    WORKER_ID=cli.worker_id;RETRY_INTERRUPTED=cli.retry_interrupted
    EPISODE_INDICES=cli.episode_indices
    if EPISODE_INDICES is not None and (len(set(EPISODE_INDICES))!=len(EPISODE_INDICES) or any(i not in range(CONFIG['test_num']) for i in EPISODE_INDICES)):
        raise ValueError('Invalid smoke slice')
    root=Path(CONFIG['project_root']);repo=root/'repos/RoboTwin'
    os.chdir(repo);sys.path[:0]=[str(repo),str(repo/'script'),str(repo/'description/utils')]
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    # Match the already validated native library import order.
    import torch,sapien,mplib
    import eval_policy_client_lingbotvla as official
    official.eval_policy=eval_policy
    usr={'task_name':cli.task,'policy_name':'LingbotVLA_v2_step20000_quick','task_config':CONFIG['task_config'],
         'ckpt_setting':f"finetuned-step{CONFIG['training_step']}-{CONFIG['setting_name']}-quick-{CONFIG['test_num']}",'seed':0,'instruction_type':'seen','test_num':CONFIG['test_num'],
         'train_config_name':str(CONFIG['training_step']),'port':CONFIG['port'],'host':'127.0.0.1','robo_name':'robotwin',
         'eval_video_log':False,'output_dir':str(run_path(CONFIG)/'tasks')}
    try:official.main(usr)
    except TaskSliceCompleted:raise SystemExit(11)
    except TaskIncomplete as exc:
        print('TASK_INCOMPLETE: '+str(exc),flush=True)
        raise SystemExit(10)

if __name__=='__main__':main()
