"""Training-only V3 video reads: retain samples and repeat a missing tail frame explicitly."""
import json
import math
import os
from pathlib import Path

_logged=set()

def query_plan(episode,key,relative_times,fps):
    start=float(episode[f'videos/{key}/from_timestamp'])
    end=float(episode[f'videos/{key}/to_timestamp'])
    count_float=(end-start)*fps
    count=round(count_float)
    if not math.isclose(count_float,count,abs_tol=1e-4) or count<1:
        raise ValueError('Non-integral or empty per-episode video interval')
    length=int(episode['length']);missing=length-count
    if missing not in (0,1,2):raise ValueError(f'Unreviewed video/state length difference: {missing}')
    original=[];selected=[]
    for timestamp in relative_times:
        frame_float=float(timestamp)*fps;index=round(frame_float)
        if abs(frame_float-index)>0.001 or not 0<=index<length:
            raise ValueError('Query is not a valid frame within this episode')
        original.append(index);selected.append(min(index,count-1))
    return {'start':start,'end':end,'last':start+(count-1)/fps,
        'timestamps':[start+i/fps for i in selected],'original':original,'selected':selected,'missing':missing}

def decode_within_episode(path,plan,tolerance=0.0001):
    import av
    import numpy as np
    import torch
    targets=sorted(set(plan['timestamps']));found={}
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        container.seek(int(targets[0]/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('Frame without presentation timestamp')
            timestamp=float(frame.pts*stream.time_base)
            # Never select a frame from the preceding or following episode.
            if timestamp<plan['start']-1e-7:continue
            if timestamp>plan['last']+1e-7:break
            matching=[t for t in targets if abs(timestamp-t)<tolerance]
            if matching:
                rgb=np.ascontiguousarray(frame.to_ndarray(format='rgb24'))
                tensor=torch.from_numpy(rgb).permute(2,0,1)
                for t in matching:found[t]=tensor
            if timestamp>=targets[-1]-tolerance:break
    if any(t not in found for t in targets):
        raise ValueError(f'No in-episode frame within original tolerance for {path}: {targets}')
    return torch.stack([found[t] for t in plan['timestamps']])

def safe_query_videos(self,query_timestamps,ep_idx):
    from lingbotvla.data.vla_data.base_dataset import LEROBOT_DATASET_API
    if LEROBOT_DATASET_API!='v3':raise ValueError('This reviewed adapter is limited to the V3 dataset')
    episode=self.meta.episodes[ep_idx];fps=float(self.meta.fps);item={}
    for key,times in query_timestamps.items():
        plan=query_plan(episode,key,times,fps)
        path=Path(self.root)/self.meta.get_video_file_path(ep_idx,key)
        frames=decode_within_episode(path,plan,float(self.tolerance_s))
        item[key]=frames.squeeze(0)
        marker=(str(self.root),int(ep_idx),key)
        if plan['original']!=plan['selected'] and marker not in _logged:
            _logged.add(marker)
            audit=os.environ.get('LINGBOT_VIDEO_AUDIT')
            if audit:
                p=Path(audit)/f'video-tail-worker-{os.getpid()}.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists() and p.stat().st_size>16*1024**2:raise RuntimeError('Video audit log budget exceeded')
                with p.open('a') as f:f.write(json.dumps({'episode_index':int(ep_idx),'camera':key,
                    'policy':'repeat_last_valid_frame_for_missing_1_or_2_tail_frames',
                    'original_indices':plan['original'],'selected_indices':plan['selected'],'missing_frames':plan['missing']})+'\n')
    return item

from lingbotvla.data.vla_data.base_dataset import LeRobotDataset as OriginalLeRobotDataset

class TrainingLeRobotDataset(OriginalLeRobotDataset):
    # A module-level subclass survives DataLoader spawn; parent-only monkeypatches do not.
    _query_videos=safe_query_videos

def install():
    import lingbotvla.data.vla_data.base_dataset as module
    module.LeRobotDataset=TrainingLeRobotDataset
