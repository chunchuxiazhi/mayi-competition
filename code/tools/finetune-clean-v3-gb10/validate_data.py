"""CPU data checks only, including two spawned workers. No model is constructed."""
from pathlib import Path
import json,os

def collate(batch):return batch

def main():
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from torch.utils.data import DataLoader,Subset
    import safe_video
    r=Path('/home/zzj/competitions/lingbot');data=r/'data/RoboTwin_lerobot_v30'
    info=json.loads((data/'meta/info.json').read_text());fps=info['fps']
    meta=pa.concat_tables([pq.read_table(p) for p in sorted((data/'meta/episodes').rglob('*.parquet'))]).to_pylist()
    keys=[k for k,v in info['features'].items() if v['dtype']=='video']
    for e in meta:
        for key in keys:safe_video.query_plan(e,key,[0,(e['length']-1)/fps],fps)
    safe_video.install()
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
    dataset=LeRobotDataset(repo_id=data.name,root=data,load_image=True,video_backend='pyav',
        delta_timestamps={'action':[i/fps for i in range(50)],**{k:[0,49/fps] for k in keys}})
    # Every task block's final episode exercises its actual video boundary.
    indices=[int(meta[i]['dataset_to_index'])-1 for i in range(49,2500,50)]
    indices=[0,len(dataset)//2,*indices]
    loader=DataLoader(Subset(dataset,indices),batch_size=1,num_workers=2,prefetch_factor=1,
        pin_memory=False,multiprocessing_context='spawn',collate_fn=collate)
    count=0
    for rows in loader:
        item=rows[0];count+=1
        assert tuple(item['action'].shape)==(50,14)
        for k in keys:
            assert tuple(item[k].shape)==(2,3,240,320) and item[k].dtype==torch.uint8
    print(json.dumps({'passed':True,'metadata_intervals_checked':len(meta)*3,'actual_samples_decoded':count,
        'coverage':'first, middle, and each of 50 task-block final episode tails','loader_workers':2,'worker_start_method':'spawn',
        'model_loaded':False,'training_or_simulation_started':False}))

if __name__=='__main__':main()
