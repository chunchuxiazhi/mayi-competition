"""Wait for closing loopback TCP endpoints; never reuse or kill a listener."""
import errno,socket,threading,time
from pathlib import Path

CLOSING_STATES={'04','05','06','09','0B'}

class PortOccupied(RuntimeError):pass

def socket_states(port):
    rows=[]
    for name in ('tcp','tcp6'):
        path=Path('/proc/net')/name
        if not path.is_file():continue
        for line in path.read_text().splitlines()[1:]:
            fields=line.split()
            if len(fields)>3 and int(fields[1].rsplit(':',1)[1],16)==port:rows.append(fields[3])
    return rows

def plain_bind(port):
    # Match the unchanged driver's check. SO_REUSEADDR would conceal TIME_WAIT
    # from this probe while the driver's subsequent plain bind could still fail.
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:probe.bind(('127.0.0.1',port))

def wait_for_port_release(port,timeout=90,interval=0.5,stop=None,on_event=None,check_resources=None):
    if not isinstance(port,int) or isinstance(port,bool) or not 0<port<65536:raise ValueError('Invalid port')
    if timeout<0 or interval<=0:raise ValueError('Invalid wait bounds')
    stop=stop if stop is not None else threading.Event()
    start=time.monotonic();wait_logged=False
    while True:
        if stop.is_set():raise InterruptedError('Port-release wait interrupted')
        if check_resources:check_resources()
        try:plain_bind(port)
        except OSError as exc:
            if exc.errno!=errno.EADDRINUSE:raise
            states=socket_states(port)
            if not states or any(state not in CLOSING_STATES for state in states):
                raise PortOccupied(f'Port {port} is occupied by an active or unidentified endpoint; no service contacted or stopped; states={states}') from exc
            elapsed=time.monotonic()-start
            if elapsed>=timeout:raise TimeoutError(f'Port {port} remained in closing states for {timeout}s; no automatic restart') from exc
            if not wait_logged and on_event:
                on_event({'event':'waiting_for_port_release','port':port,'states':states,'maximum_wait_seconds':timeout});wait_logged=True
            stop.wait(min(interval,timeout-elapsed))
        else:
            record={'event':'port_released','port':port,'wait_seconds':time.monotonic()-start}
            if on_event:on_event(record)
            return record
