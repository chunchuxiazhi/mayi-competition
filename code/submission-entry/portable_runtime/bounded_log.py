"""Drain subprocess pipes even when a text log reaches its cap; report I/O faults."""
from pathlib import Path
import threading

def drain_to_file(stream,path,limit,on_error):
    path=Path(path);log=None
    try:
        kept=path.stat().st_size if path.exists() else 0
        log=path.open('ab')
        while True:
            block=stream.read1(8192)
            if not block:break
            room=max(0,limit-kept)
            if room:log.write(block[:room]);log.flush();kept+=min(room,len(block))
    except BaseException as exc:
        on_error(type(exc).__name__+': '+str(exc))
        # The callback terminates the owned process; consume pending bytes to
        # avoid blocking its shutdown on a full stdout pipe.
        try:
            while stream.read1(8192):pass
        except Exception:pass
    finally:
        if log is not None:
            try:log.close()
            except Exception as exc:on_error(type(exc).__name__+': '+str(exc))
        try:stream.close()
        except Exception:pass

def start_drain(stream,path,limit,on_error):
    thread=threading.Thread(target=drain_to_file,args=(stream,path,limit,on_error),daemon=True)
    thread.start();return thread
