"""Exit without waiting on vLLM teardown, and without orphaning its EngineCore.

vLLM 0.31's engine teardown can hang after all work is finished, so the eval scripts exit
hard once results are on disk. os._exit() alone leaves the EngineCore child alive holding
GPU memory, so SIGKILL direct children first.
"""
import os
import signal


def hard_exit(code=0):
    me = os.getpid()
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == me:
            try:
                os.kill(int(p), signal.SIGKILL)
            except ProcessLookupError:
                pass
    os._exit(code)
