import ast,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import driver
from common import HERE,load_config,atomic_json,append_event
from model_source import expected_episode,compare

class Tests(unittest.TestCase):
    def test_explicit_mode_required_before_any_launch(self):
        for argv in [['driver.py'],['driver.py','--start-approved'],['driver.py','--smoke-then-run','--start-approved'],
                     ['driver.py','--smoke-only','--full-suite','--start-approved']]:
            with self.subTest(argv=argv),patch('sys.argv',argv),patch('subprocess.Popen') as spawn:
                with self.assertRaises(SystemExit):driver.main()
                spawn.assert_not_called()

    def test_smoke_return_precedes_full_queue(self):
        tree=ast.parse((HERE/'driver.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        trial=next(n for n in main.body if isinstance(n,ast.Try))
        smoke=next(n for n in trial.body if isinstance(n,ast.If) and ast.unparse(n.test)=='cli.smoke_only' and isinstance(n.body[-1],ast.Return))
        self.assertIsInstance(smoke.body[-1],ast.Return)

    def test_seed_instruction_drift_rejected_and_comparison_missing_visible(self):
        with tempfile.TemporaryDirectory() as d:
            cfg={'project_root':d,'run_relative':'run','tasks':['lift_pot'],'test_num':1}
            p=Path(d)/'run';p.mkdir()
            row={'task':'lift_pot','episode_index':0,'seed':100000,'policy_seed':1000000,'instruction_seed':2000000,
                 'baseline_instruction':'lift it','baseline_success':False}
            atomic_json(p/'episodes-plan.json',{'episodes':[row],'baseline_valid_episodes':1})
            ss={k:row[k] for k in ('seed','policy_seed','instruction_seed')}
            expected_episode(cfg,'lift_pot',0,ss,'lift it')
            with self.assertRaises(ValueError):expected_episode(cfg,'lift_pot',0,ss,'wrong instruction')
            with self.assertRaises(ValueError):expected_episode(cfg,'lift_pot',0,{**ss,'seed':1},'lift it')
            self.assertEqual(compare(cfg)['missing_valid_trained_slots'],1)
            e={'event':'policy_completed',**row,'instruction':'lift it','success':True}
            append_event(p/'tasks/lift_pot/episodes.jsonl',e)
            out=compare(cfg);self.assertEqual(out['matched_valid_episodes'],1);self.assertEqual(out['trained_successes_on_matched'],1)
            append_event(p/'tasks/lift_pot/episodes.jsonl',e)
            with self.assertRaises(ValueError):compare(cfg)

if __name__=='__main__':unittest.main()
