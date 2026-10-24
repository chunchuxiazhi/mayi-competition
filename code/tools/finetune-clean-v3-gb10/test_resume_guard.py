import ast,unittest
from resume_guard import transform_source

SOURCE='''def main():
    candidates=['example']
    loaded=False
    if candidates:
        for cp in candidates:
            try:
                Checkpointer.load(cp)
                scheduler.load_state_dict({})
                loaded=True
                break
            except Exception as e:
                continue
        if not loaded:
            sink.append('unsafe fallback')
    else:
        sink.append('fresh')
    sink.append('training loop')
'''
class ResumeGuardTests(unittest.TestCase):
    def test_pytorch_zero_lr_initialization_does_not_update_weights_or_training_count(self):
        import torch
        from torch.distributed.checkpoint.state_dict import _init_optim_state
        from gb10_runtime import is_restore_initialization
        weight=torch.nn.Parameter(torch.tensor([1.0,2.0]));before=weight.detach().clone()
        optimizer=torch.optim.AdamW([weight],lr=0.01,weight_decay=0.1)
        state={'restoring_optimizer':True,'training_updates':0}
        def hook(opt,a,k):
            if not is_restore_initialization(state,opt):state['training_updates']+=1
        optimizer.register_step_post_hook(hook)
        _init_optim_state(optimizer)
        self.assertTrue(torch.equal(weight.detach(),before))
        self.assertEqual(state['training_updates'],0)
        self.assertEqual(state['restore_initialization_calls'],1)
        self.assertEqual(optimizer.param_groups[0]['lr'],0.01)
        self.assertIsNone(weight.grad)
    def test_nonzero_lr_step_is_not_silently_ignored_during_restore(self):
        from gb10_runtime import is_restore_initialization
        import types
        with self.assertRaises(RuntimeError):is_restore_initialization({'restoring_optimizer':True},types.SimpleNamespace(param_groups=[{'lr':0.01}]))
    def execute(self,fail=False):
        import types
        sink=[]
        def load(x):
            if fail:raise ValueError('scheduler restore failed')
        namespace={'Checkpointer':types.SimpleNamespace(load=lambda p:None),'scheduler':types.SimpleNamespace(load_state_dict=load),
            'sink':sink,'__gb10_resume_boundary':lambda scope:sink.append(('boundary',scope['loaded']))}
        exec(compile(transform_source(SOURCE),'<resume-test>','exec'),namespace)
        return sink,namespace['main']
    def test_complete_resume_precedes_training_boundary(self):
        sink,main=self.execute();main();self.assertEqual(sink,[('boundary',True),'training loop'])
    def test_error_after_model_load_cannot_fall_back_or_train(self):
        sink,main=self.execute(True)
        with self.assertRaisesRegex(SystemExit,'no fallback'):main()
        self.assertEqual(sink,[])
    def test_unrecognized_upstream_source_refused(self):
        with self.assertRaises(RuntimeError):transform_source('def main():\n    pass\n')
    def test_training_statement_is_preserved(self):
        tree=transform_source(SOURCE)
        self.assertEqual(ast.unparse(tree.body[0].body[-1]),"sink.append('training loop')")
if __name__=='__main__':unittest.main(verbosity=2)
