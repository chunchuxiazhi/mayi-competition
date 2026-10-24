import json,tempfile,unittest
from pathlib import Path
from storage_policy import atomic_json,commit,prune,validate,checked_checkpoint,checkpoint_schedule,planned_saves,resolve_schedule

class ScheduleTests(unittest.TestCase):
    def test_fixed_1000_has_ten_points_and_does_not_calibrate(self):
        result=resolve_schedule({'train':{'max_steps':10000,'save_steps':1000,'global_batch_size':8}},
            {'checkpoint_schedule_mode':'fixed','purpose':'joint_50_tasks'})
        self.assertEqual(result['planned_saves'],list(range(1000,10001,1000)))
        self.assertEqual(result['save_every_updates'],1000)
    def test_fixed_5000_has_two_points(self):
        self.assertEqual(planned_saves(10000,5000),[5000,10000])
    def test_last_step_is_saved_when_not_divisible(self):
        self.assertEqual(planned_saves(10001,5000),[5000,10000,10001])
    def test_invalid_save_interval_is_rejected(self):
        for count in (0,-1,1.5):
            with self.assertRaises(ValueError):planned_saves(10000,count)
    def test_scales_four_sample_pilot_to_eight_sample_training_and_excludes_startup(self):
        rows=[{'optimizer_step':i,'seconds_since_previous_update':1000 if i==1 else 12} for i in range(1,21)]
        schedule=checkpoint_schedule(rows,4,8)
        self.assertEqual(schedule['estimated_training_update_seconds'],24)
        self.assertEqual(schedule['save_every_updates'],70)
    def test_missing_timing_cannot_silently_start_large_run(self):
        with self.assertRaises(ValueError):checkpoint_schedule([],4,8)
from safe_video import query_plan,decode_within_episode

class VideoPolicyTests(unittest.TestCase):
    def episode(self,length=4,frames=3):
        return {'length':length,'videos/cam/from_timestamp':0.0,'videos/cam/to_timestamp':frames/15}
    def test_one_missing_tail_frame_repeats_last_and_keeps_both_queries(self):
        p=query_plan(self.episode(),'cam',[0,3/15],15)
        self.assertEqual(p['original'],[0,3]);self.assertEqual(p['selected'],[0,2])
    def test_two_missing_tail_frames(self):
        self.assertEqual(query_plan(self.episode(5,3),'cam',[3/15,4/15],15)['selected'],[2,2])
    def test_unreviewed_missing_frames_fail(self):
        with self.assertRaises(ValueError):query_plan(self.episode(6,3),'cam',[0],15)
    def test_cross_episode_request_fails(self):
        with self.assertRaises(ValueError):query_plan(self.episode(),'cam',[4/15],15)
    def test_actual_decode_never_uses_next_episode(self):
        import av,numpy as np
        from fractions import Fraction
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'two-episodes.mp4'
            with av.open(str(path),'w') as c:
                stream=c.add_stream('mpeg4',rate=15);stream.width=32;stream.height=32;stream.pix_fmt='yuv420p'
                for i in range(6):
                    rgb=np.zeros((32,32,3),dtype=np.uint8);rgb[:,:,0 if i<3 else 2]=240
                    f=av.VideoFrame.from_ndarray(rgb,format='rgb24');f.pts=i;f.time_base=Fraction(1,15)
                    for packet in stream.encode(f):c.mux(packet)
                for packet in stream.encode():c.mux(packet)
            frames=decode_within_episode(path,query_plan(self.episode(),'cam',[0,3/15],15))
            self.assertEqual(tuple(frames.shape),(2,3,32,32))
            self.assertGreater(float(frames[:,0].float().mean()),200)
            self.assertLess(float(frames[:,2].float().mean()),30)

class CheckpointTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'checkpoints';self.root.mkdir()
    def tearDown(self):self.tmp.cleanup()
    def make(self,step,complete=True):
        p=self.root/f'global_step_{step}'
        for name in ('model/.metadata','model/__0_0.distcp','optimizer/.metadata','optimizer/__0_0.distcp','extra_state/extra_state_rank_0.pt'):
            f=p/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'example-state')
        if complete:commit(self.root,p,'test-signature')
        return p
    def test_keep_two_and_never_remove_incomplete(self):
        old=self.make(250);self.make(500);self.make(750);incomplete=self.make(999,False)
        removed=prune(self.root,'test-signature',2)
        self.assertEqual(removed,[str(old)]);self.assertTrue(incomplete.exists())
    def test_corrupted_component_refuses_load_and_retention(self):
        p=self.make(250);(p/'model/__0_0.distcp').write_bytes(b'wrong-size')
        with self.assertRaises(RuntimeError):validate(self.root,p,'test-signature')
        with self.assertRaises(RuntimeError):prune(self.root,'test-signature',1)
    def test_external_target_rejected(self):
        with self.assertRaises(ValueError):checked_checkpoint(self.root,Path(self.tmp.name)/'global_step_250')
    def test_other_config_refused(self):
        p=self.make(250)
        with self.assertRaises(ValueError):validate(self.root,p,'other-config')
    def test_incomplete_checkpoint_not_committed(self):
        p=self.make(250,False);(p/'optimizer/.metadata').unlink()
        with self.assertRaises(RuntimeError):commit(self.root,p,'test-signature')

if __name__=='__main__':unittest.main(verbosity=2)
