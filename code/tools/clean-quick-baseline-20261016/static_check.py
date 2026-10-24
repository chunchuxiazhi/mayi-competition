"""Static checks only: text/AST/configuration/filesystem, no model or simulator import."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from common import HERE,load_config,run_path,seeds,atomic_json,ready_errors

def main():
    config=load_config();root=Path(config['project_root']);repo=root/'repos/lingbot-vla-v2';sim=root/'repos/RoboTwin'
    checks={};errors=[]
    def check(name,ok,detail=None):
        checks[name]={'passed':bool(ok),'detail':detail}
        if not ok:errors.append(name)
    for p in HERE.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        compile(p.read_text(encoding='utf-8'),str(p),'exec')  # No execution and no pyc writing.
    check('python_syntax',True)
    tasks=config['tasks'];launcher=repo/config['task_list_source']['path']
    official=shlex.split(re.search(r'^task_list_all=\((.*?)\)',launcher.read_text(),re.M).group(1))
    check('official_50_tasks_no_duplicates',len(tasks)==len(set(tasks))==50)
    check('official_task_list_order_and_membership',tasks==official,{'missing':sorted(set(official)-set(tasks)),'extra':sorted(set(tasks)-set(official))})
    check('task_list_source_hash',hashlib.sha256(launcher.read_text().encode()).hexdigest()==config['task_list_source']['sha256'])
    check('task_export_matches_config',json.loads((HERE/'tasks50.json').read_text())==tasks)
    check('all_task_modules_present',all((sim/'envs'/f'{t}.py').is_file() for t in tasks))
    direct_savers={}
    for task in tasks:
        node=ast.parse((sim/'envs'/f'{task}.py').read_text())
        calls=[n.func.attr for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr in ('save_camera_rgb','save_traj_data','save_pkl','save_img','merge_pkl_to_hdf5_video')]
        if calls:direct_savers[task]=calls
    check('task_code_has_no_direct_media_or_trajectory_savers',not direct_savers,direct_savers)
    seen={t:len(json.loads((sim/'description/task_instruction'/f'{t}.json').read_text()).get('seen',[])) for t in tasks}
    check('all_tasks_have_seen_instructions',all(n>0 for n in seen.values()),seen)
    import yaml
    limits=yaml.safe_load((sim/'task_config/_eval_step_limit.yml').read_text())
    check('all_official_step_limits_present',all(type(limits.get(t)) is int and limits[t]>0 for t in tasks),{t:limits.get(t) for t in tasks})
    protected=['task_config/demo_clean.yml','task_config/_eval_step_limit.yml','envs/_base_task.py']+[f'envs/{t}.py' for t in tasks]
    unchanged=subprocess.run(['git','-C',str(sim),'diff','--quiet','HEAD','--',*protected]).returncode==0
    check('official_scenes_step_limits_success_and_action_code_unchanged',unchanged)
    clean=yaml.safe_load((sim/'task_config/demo_clean.yml').read_text())
    check('clean_scene_unchanged',not clean['domain_randomization']['random_background'] and not clean['domain_randomization']['cluttered_table'] and not clean['domain_randomization']['random_light'])
    check('suite_count_500',config['test_num']==10 and len(tasks)*config['test_num']==500)
    check('clean_seen_only',config['task_config']=='demo_clean' and config['instruction_type']=='seen')
    check('two_workers_single_fallback',config['workers']==2 and config['allowed_workers']==[1,2])
    check('original_fp32_settings',config['model_repo']=='robbyant/lingbot-vla-v2-6b' and config['use_fp32'] and not config['use_bf16'] and not config['use_compile'] and config['use_length']==50 and config['chunk_ret'])
    check('media_saving_disabled',all(config[k] is False for k in ['eval_video_log','save_data','save_frames','save_observation_trajectory']))
    wrapper=root/config['model_wrapper_relative']
    check('checkpoint_symlink_original',wrapper.is_symlink() and wrapper.resolve()==(root/config['model_relative']).resolve())
    check('runtime_paths_exist',all((root/p).exists() for p in ['envs/sim/bin/python','envs/vla/bin/python',config['model_relative'],'models/Qwen3-VL-4B-Instruct']))
    source=json.loads((HERE/'model-source.json').read_text())
    check('original_weight_shards_present_sizes',source['repo']==config['model_repo'] and len(source['files'])==6 and all((root/config['model_relative']/p['path']).stat().st_size==p['bytes'] for p in source['files']))
    construction=yaml.safe_load((run_path(config)/'lingbotvla_cli.yaml').read_text())
    check('wrapper_has_no_unverified_normalization_fallback',construction['data']['norm_stats_file']==str(root/'configs/normalization/robotwin_clean_v3.json'))
    check('model_construction_preserved_fp32_fused',construction['model']['post_training'] is False and construction['model']['moe_implementation']=='fused')
    check('official_expert_check_not_disabled',config['expert_check'] is True and config['expert_failure_policy']=='validate_metadata_then_policy_same_seed')
    client=sim/'script/eval_policy_client_lingbotvla.py'
    text=client.read_text();tree=ast.parse(text)
    main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assigns=[n for n in ast.walk(main_node) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='test_num' for t in n.targets)]
    check('legacy_client_count_configurable',len(assigns)==1 and not isinstance(assigns[0].value,ast.Constant))
    check('no_standard_score_claim',config['label']=='每任务10回合的预训练模型 clean 快速基线')
    plan=[{'task':t,'episode_index':i,'worker_id':ti%2,**seeds(config,ti,i),
           'setting':'demo_clean','instruction_type':'seen','result_directory':str(run_path(config)/'tasks'/t)}
          for ti,t in enumerate(tasks) for i in range(config['test_num'])]
    check('unique_episode_keys',len({(p['task'],p['episode_index']) for p in plan})==500)
    queues=[tasks[i::2] for i in range(2)]
    check('two_disjoint_queues',len(queues[0])==len(queues[1])==25 and not set(queues[0])&set(queues[1]) and set(queues[0]+queues[1])==set(tasks))
    check('no_output_directory_collision',len({p['result_directory'] for p in plan})==50)
    check('previous_unseen_results_excluded',run_path(config).resolve()!=(root/config['prior_unseen_run_excluded']).resolve())
    check('approval_state_recorded',type(config['execution_approval']['approved']) is bool)
    for name in ('serve.py','evaluate_task.py','driver.py'):
        source=(HERE/name).read_text();tree=ast.parse(source)
        check(name+'_imports_gated',all(not isinstance(n,(ast.Import,ast.ImportFrom)) or not any(a.name.split('.')[0] in ('torch','sapien','mplib','numpy','deploy') for a in n.names) for n in tree.body))
        check(name+'_explicit_ready_gate','require_ready(' in source)
    session=(HERE/'session_policy.py').read_text()
    check('session_state_design_markers',all(s in session for s in ['threading.RLock','copy.copy','ctx[\'transform\']','ctx[\'rng\']','cuda.synchronize','path_to_pi_model','episode_seed']))
    policy_tree=ast.parse((repo/'deploy/lingbot_vla_v2_policy.py').read_text())
    policy_class=next(n for n in policy_tree.body if isinstance(n,ast.ClassDef) and n.name=='LingbotVLAv2Server')
    assigned=set()
    for method in policy_class.body:
        if isinstance(method,ast.FunctionDef) and method.name in ('infer','reset'):
            for node in ast.walk(method):
                if isinstance(node,(ast.Assign,ast.AugAssign)):
                    for target in node.targets if isinstance(node,ast.Assign) else [node.target]:
                        if isinstance(target,ast.Attribute):assigned.add(ast.unparse(target))
    known={'self.global_step','self.last_action_chunk','self.last_normalized_action_chunk','self.action_key','self.robot_config','self.vla','self.vla.feature_transform'}
    check('upstream_mutable_session_fields_reviewed',assigned<=known,sorted(assigned))
    model_tree=ast.parse((repo/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py').read_text())
    samples=[n for n in ast.walk(model_tree) if isinstance(n,ast.FunctionDef) and n.name=='sample_actions']
    check('request_local_kv_cache',any('past_key_values=None' in ast.unparse(n) for n in samples))
    gates=ready_errors(config,approval=config['execution_approval']['approved'])
    report={'scope':'Static checks only; no simulation, model load, service, inference or training executed',
            'checks':checks,'static_passed':not errors,'static_failures':errors,
            'launch_readiness':'ready_for_approved_local_baseline' if not gates else 'blocked_pending_review_items','launch_blockers':gates,
            'dynamic_session_isolation_tested':False,'runtime_episode_count':0,
            'config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()}
    atomic_json(run_path(config)/'episodes-plan.json',plan)
    atomic_json(run_path(config)/'static-check-report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
