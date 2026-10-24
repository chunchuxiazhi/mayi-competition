"""On-demand status only. Does not monitor, start, resume, or modify a run."""
import json
from pathlib import Path
import subprocess
from common import load_config,run_path

def main():
    config=load_config();run=run_path(config)
    out={'label':config['label'],'planned_episodes':len(config['tasks'])*config['test_num'],
         'execution':json.loads((run/'execution-status.json').read_text()),'tasks':{},'valid_completed':0,'successes':0,'exceptions':0,'interrupted':0,'expert_attempts':0}
    for task in config['tasks']:
        directory=run/'tasks'/task;rows=[]
        p=directory/'episodes.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():rows.append(json.loads(line))
        completed=[r for r in rows if r['event']=='policy_completed']
        if len({r['episode_index'] for r in completed})!=len(completed):raise RuntimeError('Duplicate completed episode')
        exceptions=[r for r in rows if r['event']=='policy_exception']
        interruptions=[r for r in rows if r['event']=='policy_interrupted']
        experts=directory/'expert.jsonl'
        expert_rows=[json.loads(s) for s in experts.read_text().splitlines() if s.strip()] if experts.exists() else []
        row={'completed':len(completed),'successes':sum(r['success'] for r in completed),
             'exceptions':len(exceptions),'interrupted':len(interruptions),'expert_attempts':sum(r['event']=='expert_started' for r in expert_rows),
             'last_event':rows[-1]['event'] if rows else None}
        if rows or expert_rows:out['tasks'][task]=row
        out['valid_completed']+=row['completed'];out['successes']+=row['successes'];out['exceptions']+=row['exceptions'];out['interrupted']+=row['interrupted'];out['expert_attempts']+=row['expert_attempts']
    out['success_rate']=out['successes']/out['valid_completed'] if out['valid_completed'] else None
    out['sessions']=subprocess.run(['tmux','list-sessions'],capture_output=True,text=True).stdout
    out['overall_competition_score']=None
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
