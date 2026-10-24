"""CPU-only clean state/action statistics, using upstream RunningStats.

No robot simulation, model construction, image decoding, or optimizer.
"""
from pathlib import Path
import collections
import hashlib
import ast
import json
import re
import sys
import time
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

def main():
    started=time.monotonic();root=Path('/home/zzj/competitions/lingbot')
    run=root/'runs/pretrained-clean-quick-50x10-20261016';outdir=run/'preparation'
    dataset=root/'data/RoboTwin_lerobot_v30';repo=root/'repos/lingbot-vla-v2';sim=root/'repos/RoboTwin'
    config=json.loads((Path(__file__).parent/'config.json').read_text())
    tasks=sorted(config['tasks'])
    meta_files=sorted((dataset/'meta/episodes').rglob('*.parquet'))
    episodes=pa.concat_tables([pq.read_table(p,columns=['episode_index','tasks','length','dataset_from_index','dataset_to_index']) for p in meta_files]).to_pylist()
    episodes.sort(key=lambda e:e['episode_index'])
    assert len(episodes)==2500 and [e['episode_index'] for e in episodes]==list(range(2500))
    expected_counts=collections.Counter();mapping=[];bad=[];block_checks=[]
    def canonical(text):return re.sub(r'\s+',' ',text.strip()).rstrip('.!?').casefold()
    def patterns(task):
        templates=json.loads((sim/'description/task_instruction'/f'{task}.json').read_text())['seen']
        return [re.compile('^'+''.join('.+?' if s.startswith('{') and s.endswith('}') else re.escape(s) for s in re.split(r'(\{[^}]+\})',canonical(t)))+'$',re.I) for t in templates]
    regex={t:patterns(t) for t in tasks}
    # The merged dataset preserves 50-episode task blocks, but not alphabetical order.
    # Identify each block from independent official instruction-template matches.
    block_tasks=[]
    for begin in range(0,2500,50):
        block=episodes[begin:begin+50]
        scores={t:sum(all(any(p.fullmatch(canonical(text)) for p in regex[t]) for text in e['tasks']) for e in block) for t in tasks}
        ranking=sorted(scores,key=scores.get,reverse=True);task=ranking[0]
        if scores[task]<40 or scores[task]<=scores[ranking[1]]:
            raise RuntimeError('Task block mapping ambiguous: '+json.dumps({'begin':begin,'scores':scores}))
        block_tasks.append(task);block_checks.append({'first_episode':begin,'task':task,'matched':scores[task],'total':50,'runner_up':scores[ranking[1]]})
    assert len(set(block_tasks))==50 and set(block_tasks)==set(tasks)
    cursor=0
    for e in episodes:
        task=block_tasks[e['episode_index']//50]
        assert e['dataset_from_index']==cursor and e['dataset_to_index']-cursor==e['length']
        cursor=e['dataset_to_index']
        matches=all(any(p.fullmatch(canonical(text)) for p in regex[task]) for text in e['tasks'])
        if not matches:bad.append({'episode':e['episode_index'],'expected_sorted_block_task':task,'instruction':e['tasks']})
        expected_counts[task]+=1
        mapping.append({'episode_index':e['episode_index'],'task':task,'length':e['length'],'instructions':e['tasks'],
                        'from':e['dataset_from_index'],'to':e['dataset_to_index']})
    assert cursor==548893 and all(n==50 for n in expected_counts.values())
    cols=['observation.state','action','episode_index','frame_index','index']
    data_files=sorted((dataset/'data').rglob('*.parquet'))
    table=pa.concat_tables([pq.read_table(p,columns=cols) for p in data_files])
    assert table.num_rows==cursor
    state=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
    action=np.asarray(table['action'].to_pylist(),dtype=np.float32)
    epids=table['episode_index'].to_numpy();frameids=table['frame_index'].to_numpy();indices=table['index'].to_numpy()
    assert state.shape==action.shape==(548893,14)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert np.array_equal(indices,np.arange(cursor))
    for e in episodes:
        a,b=e['dataset_from_index'],e['dataset_to_index']
        assert np.all(epids[a:b]==e['episode_index']) and np.array_equal(frameids[a:b],np.arange(b-a))
    robot=yaml.safe_load((repo/'configs/robot_configs/robotwin.yaml').read_text())
    for feature in robot['actions']:assert next(iter(feature.values()))['subtract_state'] is False
    upstream=repo/'lingbotvla/utils/normalize.py'
    # Reuse the upstream numerical class unchanged, without importing its optional
    # numpydantic serialization dependency or the training/model package.
    source_tree=ast.parse(upstream.read_text())
    numeric_class=next(n for n in source_tree.body if isinstance(n,ast.ClassDef) and n.name=='RunningStats')
    numeric_tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),numeric_class],type_ignores=[])
    ast.fix_missing_locations(numeric_tree)
    namespace={'np':np,'NormStats':SimpleNamespace}
    exec(compile(numeric_tree,str(upstream),'exec'),namespace)
    module=SimpleNamespace(RunningStats=namespace['RunningStats'])
    stats={k:module.RunningStats() for k in ['observation.state.arm.position','observation.state.effector.position','action.arm.position','action.effector.position']}
    arm=[0,1,2,3,4,5,7,8,9,10,11,12];eff=[6,13]
    # Matches upstream current-state and future 0..49 absolute-action windows.
    # Episode tail indices clamp to the last action, exactly as LeRobot query padding.
    for begin in range(0,len(episodes),64):
        batch=episodes[begin:begin+64]
        a,b=batch[0]['dataset_from_index'],batch[-1]['dataset_to_index']
        current=state[a:b]
        stats['observation.state.arm.position'].update(current[:,arm])
        stats['observation.state.effector.position'].update(current[:,eff])
        windows=[]
        for e in batch:
            x,y=e['dataset_from_index'],e['dataset_to_index']
            idx=np.minimum(np.arange(x,y)[:,None]+np.arange(50)[None,:],y-1)
            windows.append(action[idx].reshape(-1,14))
        future=np.concatenate(windows,axis=0)
        stats['action.arm.position'].update(future[:,arm])
        stats['action.effector.position'].update(future[:,eff])
        print('CLEAN_STATS_EPISODES',min(begin+64,2500),'/2500',flush=True)
    result={k:s.get_statistics() for k,s in stats.items()}
    output=root/'configs/normalization/robotwin_clean_v3.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    payload={'norm_stats':{key:{k:np.asarray(v).tolist() for k,v in value.__dict__.items()} for key,value in result.items()},'count':cursor}
    output.write_text(json.dumps(payload,indent=2,allow_nan=False))
    parsed=json.loads(output.read_text())
    assert parsed['count']==548893
    for key,entry in parsed['norm_stats'].items():
        expected=12 if '.arm.' in key else 2
        assert len(entry['mean'])==expected
        for name in ['mean','std','q01','q99','q02','q98','min','max']:assert np.isfinite(entry[name]).all()
    (outdir/'clean-episode-task-map.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2))
    evidence={'status':'computed_from_downloaded_demo_clean_v3_state_action_only','dataset':str(dataset),
      'source_statement':'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/discussions/95',
      'total_episodes':2500,'total_frames':cursor,'task_episode_counts':dict(expected_counts),
      'mapping_validation':'50-episode blocks independently identified by >=40/50 official seen-template matches, uniquely covering all 50 tasks; punctuation normalized only for mapping validation; frame ranges, IDs and lengths checked against raw parquet.',
      'task_block_checks':block_checks,'unmatched_instruction_variations':bad,
      'method':'Official RunningStats/5000 histogram bins; deterministic batches of 64 episodes; current state; absolute future actions at offsets 0..49, clamped within each episode; raw feature slices before padding; no state subtraction.',
      'normalization_scope':'clean-derived preprocessing for local pretrained baseline; no claim that official rules have separately approved this exact statistics procedure',
      'count_by_feature':{k:s._count for k,s in stats.items()},
      'input_sha256':{str(p.relative_to(dataset)):hashlib.sha256(p.read_bytes()).hexdigest() for p in meta_files+data_files},
      'official_algorithm_sha256':hashlib.sha256(upstream.read_bytes()).hexdigest(),
      'robot_feature_map_sha256':hashlib.sha256((repo/'configs/robot_configs/robotwin.yaml').read_bytes()).hexdigest(),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'output':str(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
      'images_decoded':0,'model_loaded':False,'weight_updates':False,'elapsed_seconds':time.monotonic()-started}
    (outdir/'clean-normalization-provenance.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['input_sha256','task_episode_counts']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
