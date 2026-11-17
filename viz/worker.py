"""One frontend-owned physical preparation job on the unchanged V1 engine."""
from pathlib import Path
import argparse
import json

def main():
    from runtime_environment import activate
    activate()
    from paths import owned
    from astermind.builder.physical.network import run_network_experiment
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();output=owned(args.out)
    result=run_network_experiment(json.loads(args.manifest.read_text()),output,base_dir=args.manifest.parent,
                                  _workers=1,export_workers=1,progress=lambda x:print(json.dumps(x),flush=True))
    print(json.dumps({'checks':result['checks'],'output':str(output)}),flush=True)

if __name__=='__main__':main()
