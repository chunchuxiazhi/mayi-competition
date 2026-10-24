"""CPU/loopback-only checks; no model, CUDA or simulator imports."""
import argparse,errno,json,socket,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
import dispatch,port_release

class Ports(unittest.TestCase):
    def test_free_port_passes_plain_bind(self):
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        self.assertEqual(port_release.wait_for_port_release(port)['port'],port)
    def test_active_listener_is_rejected_and_left_running(self):
        with socket.socket() as service:
            service.bind(('127.0.0.1',0));service.listen();port=service.getsockname()[1]
            with self.assertRaises(port_release.PortOccupied):port_release.wait_for_port_release(port,timeout=0.1)
            with socket.create_connection(('127.0.0.1',port),timeout=1) as client:
                connection,_=service.accept();connection.close()
    def test_closing_state_waits_then_requires_successful_plain_bind(self):
        events=[]
        with patch('port_release.plain_bind',side_effect=[OSError(errno.EADDRINUSE,'busy'),None]) as bind,patch('port_release.socket_states',return_value=['06']):
            result=port_release.wait_for_port_release(12345,timeout=1,interval=0.001,on_event=events.append)
        self.assertEqual(bind.call_count,2);self.assertGreaterEqual(result['wait_seconds'],0)
        self.assertEqual([e['event'] for e in events],['waiting_for_port_release','port_released'])
    def test_closing_state_has_bounded_timeout(self):
        with patch('port_release.plain_bind',side_effect=OSError(errno.EADDRINUSE,'busy')),patch('port_release.socket_states',return_value=['06']):
            with self.assertRaises(TimeoutError):port_release.wait_for_port_release(12345,timeout=0.01,interval=0.002)
    def test_unidentified_or_live_connection_is_not_reused(self):
        for states in ([],['01'],['06','0A']):
            with patch('port_release.plain_bind',side_effect=OSError(errno.EADDRINUSE,'busy')),patch('port_release.socket_states',return_value=states):
                with self.assertRaises(port_release.PortOccupied):port_release.wait_for_port_release(12345)
    def test_other_bind_error_is_not_suppressed(self):
        with patch('port_release.plain_bind',side_effect=PermissionError(errno.EACCES,'denied')):
            with self.assertRaises(PermissionError):port_release.wait_for_port_release(12345)
    def test_wait_can_be_interrupted(self):
        stop=threading.Event();stop.set()
        with self.assertRaises(InterruptedError):port_release.wait_for_port_release(12345,stop=stop)
    def test_resource_failure_stops_port_wait(self):
        with self.assertRaises(MemoryError):port_release.wait_for_port_release(12345,check_resources=lambda:(_ for _ in ()).throw(MemoryError('reserve')))

class Scheduling(unittest.TestCase):
    def test_default_only_checks_without_launching(self):
        cfgs=[{'task_config':'demo_clean'},{'task_config':'demo_randomized'}]
        with patch('sys.argv',['dispatch.py']),patch('dispatch.check',return_value=(cfgs,{})),patch('dispatch.run') as run,patch('dispatch.subprocess.Popen') as popen:
            dispatch.main();run.assert_not_called();popen.assert_not_called()
    def test_full_requires_explicit_approval(self):
        with patch('sys.argv',['dispatch.py','--mode','full']),patch('dispatch.check') as check:
            with self.assertRaises(RuntimeError):dispatch.main()
            check.assert_not_called()
    def exercise(self,fail_first=False):
        with tempfile.TemporaryDirectory(prefix='lingbot-dispatch-cpu-') as temporary:
            root=Path(temporary);suite=root/'suite';suite.mkdir()
            cfgs=[{'project_root':str(root),'suite_relative':'suite','run_relative':'suite/'+name,'setting_name':name,'task_config':setting,'port':12345}
                  for name,setting in (('clean','demo_clean'),('randomized','demo_randomized'))]
            sequence=[]
            for cfg in cfgs:(suite/cfg['setting_name']).mkdir()
            def wait(port,**kwargs):sequence.append('wait');return {'port':port}
            def call(argv):
                name='clean' if 'config-clean.json' in ' '.join(argv) else 'randomized';sequence.append(name)
                dispatch.atomic_json(suite/name/'execution-status.json',{'owned_children_exited':True,'stop_reason':'fake fault' if fail_first and name=='clean' else None})
                return 1 if fail_first else 0
            summary={'complete':False,'valid_policy_episodes':4}
            args=argparse.Namespace(workers=2,retry_interrupted=False)
            with patch('dispatch.ROOT',root),patch('dispatch.idle'),patch('dispatch.startup_check'),patch('dispatch.ResourceGuard'),patch('dispatch.suite_cache_baseline',return_value=0),patch('dispatch.wait_for_port_release',side_effect=wait),patch('dispatch.call',side_effect=call),patch('dispatch.collect',return_value=summary),patch('dispatch.export',return_value={'test':True}),patch('dispatch.signal.signal'):
                if fail_first:
                    with self.assertRaises(RuntimeError):dispatch.run(args,cfgs,{})
                else:dispatch.run(args,cfgs,{})
            status=json.loads((suite/'execution-status.json').read_text())
            self.assertFalse(status['automatic_restart'])
            self.assertEqual(status['state'],'stopped' if fail_first else 'stopped_incomplete')
            return sequence
    def test_each_setting_is_preceded_by_port_release_check(self):self.assertEqual(self.exercise(),['wait','clean','wait','randomized'])
    def test_first_setting_fault_does_not_launch_second(self):self.assertEqual(self.exercise(True),['wait','clean'])

if __name__=='__main__':unittest.main(verbosity=2)
