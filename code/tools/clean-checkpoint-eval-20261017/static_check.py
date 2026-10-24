"""AST/config/source/space checks only. No torch, model or simulator imports."""
import ast,json,shutil
from pathlib import Path
from common import HERE,load_config,run_path,require_ready,atomic_json,seeds
from model_source import verify_source,digest

def main():
    cfg=load_config();root=Path(cfg['project_root']).resolve();run=run_path(cfg).resolve()
    require_ready(cfg,True);source=verify_source(cfg)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    assert cfg['test_num']==10 and len(set(cfg['tasks']))==50 and len(cfg['tasks'])==50
    assert cfg['task_config']=='demo_clean' and cfg['instruction_type']=='seen'
    assert cfg['workers']==2 and cfg['allowed_workers']==[1,2]
    assert cfg['use_fp32'] and not cfg['use_bf16'] and not cfg['use_compile']
    assert cfg['chunk_ret'] and cfg['use_length']==50
    assert not any(cfg[k] for k in ('save_data','eval_video_log','save_frames','save_observation_trajectory'))
    assert run.is_relative_to(root/'runs') and run!=(root/cfg['baseline_relative']).resolve()
    assert not run.is_relative_to((root/cfg['model_relative']).resolve())
    plan=json.loads((run/'episodes-plan.json').read_text());assert len(plan['episodes'])==500
    seen=set()
    for e in plan['episodes']:
        key=(e['task'],e['episode_index']);assert key not in seen;seen.add(key)
        assert all(e[k]==v for k,v in seeds(cfg,cfg['tasks'].index(e['task']),e['episode_index']).items())
    prov=json.loads((run/'preparation/input-provenance.json').read_text())
    for path,sha in prov['official_files_sha256'].items():assert digest(path)==sha,path
    for entry in plan['baseline_ledgers']:assert digest(entry['path'])==entry['sha256']
    assert shutil.disk_usage(root).free>32*2**30
    report={'passed':True,'planned_tasks':50,'planned_slots':500,'model_source_sha256':source,
            'code_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},'no_model_or_simulator_loaded':True,
            'official_task_files_unchanged':True,'baseline_files_unchanged':True,
            'expert_quota_official_scope_confirmed':False}
    atomic_json(run/'preparation/static-check.json',report)
    print(json.dumps(report))

if __name__=='__main__':main()
