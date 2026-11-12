"""Build the two pinned CPU helpers used by compute and communication."""
import os
import subprocess
from astermind.common.bootstrap import activate


def main():
    root=activate()
    directory=root/'build';directory.mkdir(exist_ok=True)
    target=directory/'lru_sim.so'
    # Exact upstream lru_sim.c build recipe; no native predictor changes.
    subprocess.run([os.environ.get('CC','gcc'),'-O3','-shared','-fPIC','-o',str(target),
                    str(root/'src/kernel/csrc/lru_sim.c')],check=True)
    # Keep the common build entry independent of runtime imports.
    import sys
    print(target)
    subprocess.run([sys.executable,'-m','astermind.engine.comm.flow_allocator'],check=True)
if __name__=='__main__':main()
