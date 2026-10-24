"""Structural/source/count checks; missing export is a visible runtime blocker."""
import argparse,ast,json,shutil
from common import HERE,load_config,run_path,atomic_json
from validation import validate_config,validate_inputs,digest
from model_source import verify_source

def check(cfg,require_model=False):
    report=validate_config(cfg);validate_inputs(cfg)
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    if shutil.disk_usage(cfg['project_root']).free<40*2**30:raise RuntimeError('Less than 40GiB disk reserve')
    blockers=[];source=None
    try:source=verify_source(cfg)
    except Exception as exc:blockers.append(str(exc))
    report.update(static_checks_passed=True,runtime_ready=not blockers,runtime_blockers=blockers,
                  model_source_sha256=source,no_model_or_simulator_loaded=True,
                  code_sha256={p.name:digest(p) for p in HERE.glob('*.py')},
                  expert_quota_official_scope_confirmed=False,normalization_method_officially_confirmed=False,
                  runtime_smoke_executed=False)
    atomic_json(run_path(cfg)/'preparation/static-check.json',report)
    if require_model and blockers:raise RuntimeError('Runtime preparation incomplete: '+'; '.join(blockers))
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--require-ready',action='store_true');args=p.parse_args()
    print(json.dumps(check(load_config(args.config),args.require_ready)))

if __name__=='__main__':main()
