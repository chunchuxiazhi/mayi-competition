import ast
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from segment_stop import transform_tree,validate_budget

SAMPLE='''
def main():
    __gb10_resume_boundary(locals())
    saved=[]
    performed=[]
    reached_max_steps=False
    for global_step in range(1001,args.train.max_steps+1):
        performed.append(global_step)
        if global_step % args.train.save_steps == 0:
            saved.append(global_step)
        if args.train.max_steps is not None and global_step >= args.train.max_steps:
            logger.info_rank0(f'Reached max_steps={args.train.max_steps}, stopping training.')
            reached_max_steps=True
            break
    return performed,saved,reached_max_steps,args.train.max_steps
'''


class SegmentStopTest(unittest.TestCase):
    def test_native_save_occurs_before_exact_stop_and_original_schedule_is_unchanged(self):
        args=SimpleNamespace(train=SimpleNamespace(max_steps=10000,save_steps=1000))
        namespace={'args':args,'__gb10_resume_boundary':lambda scope:None,
                   'logger':SimpleNamespace(info_rank0=lambda message:None)}
        tree=transform_tree(ast.parse(SAMPLE),2000)
        with patch('segment_stop.resume_boundary') as boundary:
            exec(compile(tree,'<segment-test>','exec'),namespace)
            performed,saved,stopped,max_steps=namespace['main']()
            boundary.assert_called_once()
        self.assertEqual(performed,list(range(1001,2001)))
        self.assertEqual(saved,[2000])
        self.assertTrue(stopped)
        self.assertEqual(max_steps,10000)
        # Schedule construction still sees the original maximum, not 2000.
        self.assertEqual(args.train.max_steps,10000)

    def test_transform_preserves_all_other_statements(self):
        before=ast.parse(SAMPLE);after=transform_tree(deepcopy(before),2000)
        before_fn=before.body[0];after_fn=after.body[0]
        self.assertEqual(ast.dump(before_fn.body[0]),ast.dump(after_fn.body[0]))
        self.assertEqual(ast.dump(before_fn.body[-1]),ast.dump(after_fn.body[-1]))
        before_loop=next(n for n in before_fn.body if isinstance(n,ast.For))
        after_loop=next(n for n in after_fn.body if isinstance(n,ast.For))
        self.assertEqual(ast.dump(before_loop.iter),ast.dump(after_loop.iter))
        self.assertEqual(ast.dump(before_loop.body[0]),ast.dump(after_loop.body[0]))
        self.assertEqual(ast.dump(before_loop.body[1]),ast.dump(after_loop.body[1]))
        self.assertEqual(ast.dump(before_loop.body[2].body[1]),ast.dump(after_loop.body[2].body[1]))
        self.assertEqual(ast.dump(before_loop.body[2].body[2]),ast.dump(after_loop.body[2].body[2]))

    def test_budget_leaves_configuration_unchanged(self):
        config={'train':{'max_steps':10000,'save_steps':1000,'lr':2e-5}}
        copy=deepcopy(config)
        self.assertEqual(validate_budget(config,2000,1000)['original_max_steps'],10000)
        self.assertEqual(config,copy)

    def test_reject_missing_or_non_integer_boundary(self):
        config={'train':{'max_steps':10000,'save_steps':1000}}
        for stop,start in ((None,1000),(2000,None),('2000',1000),(True,1000)):
            with self.subTest(stop=stop,start=start),self.assertRaises(ValueError):validate_budget(config,stop,start)

    def test_reject_unsafe_limits_or_unscheduled_saves(self):
        config={'train':{'max_steps':10000,'save_steps':1000}}
        for stop,start in ((1000,1000),(500,1000),(11000,1000),(2500,1000),(2000,999),(2000,0)):
            with self.subTest(stop=stop,start=start),self.assertRaises(ValueError):validate_budget(config,stop,start)

    def test_reject_upstream_termination_structure_changes(self):
        source=SAMPLE.replace('global_step >= args.train.max_steps','global_step > args.train.max_steps')
        with self.assertRaises(RuntimeError):transform_tree(ast.parse(source),2000)

    def test_reject_absent_resume_guard(self):
        source=SAMPLE.replace('__gb10_resume_boundary(locals())','pass')
        with self.assertRaises(RuntimeError):transform_tree(ast.parse(source),2000)


if __name__=='__main__':unittest.main()
