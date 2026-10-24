"""Two tiny TCP servers on one ephemeral port; no model or simulation."""
import errno,json,socket,time
from pathlib import Path
from port_release import plain_bind,socket_states,wait_for_port_release

def main():
    started=time.monotonic();events=[]
    with socket.socket() as first:
        first.bind(('127.0.0.1',0));first.listen();port=first.getsockname()[1]
        with socket.create_connection(('127.0.0.1',port),timeout=2) as client:
            connection,_=first.accept()
            with connection:
                connection.sendall(b'phase-one');connection.shutdown(socket.SHUT_WR)
            while client.recv(1024):pass
    try:plain_bind(port)
    except OSError as exc:
        if exc.errno!=errno.EADDRINUSE:raise
        blocked=True
    else:raise RuntimeError('Did not reproduce the closing-port condition; test is inconclusive')
    states=socket_states(port)
    if '06' not in states:raise RuntimeError('TIME_WAIT not observed; test is inconclusive')
    report={'state':'waiting_for_real_time_wait','port':port,'original_plain_bind_blocked':blocked,'socket_states':states}
    output=Path('/home/zzj/competitions/lingbot/docs/eval-prep-20261023/port-transition-real-test.json')
    output.write_text(json.dumps(report))
    released=wait_for_port_release(port,timeout=90,interval=0.5,on_event=events.append)
    with socket.socket() as second:
        second.bind(('127.0.0.1',port));second.listen()
        with socket.create_connection(('127.0.0.1',port),timeout=2) as client:
            connection,_=second.accept()
            with connection:connection.sendall(b'phase-two')
            if client.recv(1024)!=b'phase-two':raise RuntimeError('Second phase failed')
    report.update(state='passed',passed=True,second_service_started_on_same_port=True,events=events,wait_seconds=released['wait_seconds'],total_seconds=time.monotonic()-started,model_loaded=False,simulation_started=False)
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
