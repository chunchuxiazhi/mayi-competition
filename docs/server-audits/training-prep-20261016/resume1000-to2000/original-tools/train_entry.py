"""Explicitly launched child entry. The upstream trainer file remains unchanged."""
import argparse,json,os,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--control',required=True)
    cli=parser.parse_args()
    import yaml
    config=yaml.safe_load(Path(cli.config).read_text());control=json.loads(Path(cli.control).read_text())
    if os.environ.get('LINGBOT_TRAINING_APPROVED')!='1':raise RuntimeError('Start approval flag absent')
    sys.path.insert(0,str(Path(control['project_root'])/'repos/lingbot-vla-v2'))
    import tasks.vla.train_lingbotvla as official
    import gb10_runtime
    shutdown_loaders=gb10_runtime.install(official,config,control)
    # Keep model selection, loss, action semantics and native optimization loop upstream.
    # Fail on a resume error instead of accepting the upstream fallback to a fresh optimizer.
    original_parse=official.parse_args
    def strict_parse(rootclass):
        args=original_parse(rootclass)
        if args.train.load_checkpoint_path:raise ValueError('Use the run-owned automatic resume path only')
        return args
    official.parse_args=strict_parse
    # Main's resume loop catches exceptions. Force a failed managed load to escape that loop.
    managed=official.build_checkpointer
    def strict_factory(*args,**kwargs):
        cls=managed(*args,**kwargs);load=cls.load
        @classmethod
        def strict_load(inner,*a,**k):
            try:return load(*a,**k)
            except Exception as exc:raise SystemExit('Resume rejected; no fresh-optimizer fallback: '+repr(exc)) from exc
        cls.load=strict_load
        return cls
    official.build_checkpointer=strict_factory
    import resume_guard
    resume_guard.install(official,config,control)
    sys.argv=[str(Path(control['project_root'])/'repos/lingbot-vla-v2/tasks/vla/train_lingbotvla.py'),cli.config]
    try:official.main()
    finally:shutdown_loaders()

if __name__=='__main__':main()
