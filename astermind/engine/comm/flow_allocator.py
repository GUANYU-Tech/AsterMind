"""Native FlowSim allocation core; explicit provenance, no silent fallback."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys


def build():
    source=Path(__file__).with_name("native_allocator.cc")
    directory=source.parent.parent/"build"
    directory.mkdir(exist_ok=True)
    target=directory/("comm_allocator.dll" if sys.platform=="win32" else "libcomm_allocator.so")
    command=[os.environ.get("CXX","g++"),"-std=c++17","-O2","-shared",str(source),"-o",str(target)]
    if sys.platform!="win32": command.insert(1,"-fPIC")
    subprocess.run(command,check=True)
    return target


class FlowAllocator:
    provenance="flowsim_db16261_progressive_filling_native_extraction"

    def __init__(self):
        path=Path(__file__).parent.parent/"build"/("comm_allocator.dll" if sys.platform=="win32" else "libcomm_allocator.so")
        if not path.exists():
            raise RuntimeError("Build native allocation core: python -m astermind.engine.comm.flow_allocator")
        self.lib=ctypes.CDLL(str(path))
        ptr=ctypes.POINTER(ctypes.c_double)
        self.lib.comm_allocate.argtypes=[ctypes.c_int,ctypes.c_int,ptr,ptr,ptr]
        self.lib.comm_allocate.restype=ctypes.c_int

    def rates(self, demands, capacities):
        if not demands: return {}
        ids=list(demands)
        keys=list(dict.fromkeys(k for demand in demands.values() for k in demand))
        array=lambda values:(ctypes.c_double*len(values))(*values)
        result=array([0.]*len(ids))
        status=self.lib.comm_allocate(len(ids),len(keys),array([capacities[k] for k in keys]),
            array([demands[i].get(k,0.) for i in ids for k in keys]),result)
        if status: raise RuntimeError(f"FlowSim allocation failed ({status})")
        return dict(zip(ids,result))


if __name__=="__main__": print(build())
