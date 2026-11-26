"""One frontend-owned physical preparation job on the unchanged V1 engine."""
from pathlib import Path
import argparse
import json


def prepare_physical(manifest, output, runner, *, prepare_fast_frames=False):
    # Let the backend apply its CPU affinity budget and per-stage task limits.
    # _workers is a private test override, not a frontend deployment default.
    options = {'prepare_fast_frames': True} if prepare_fast_frames else {}
    result = runner(json.loads(manifest.read_text(encoding='utf-8')), output,
                    base_dir=manifest.parent,
                    progress=lambda stage: print(stage, flush=True), **options)
    keys = ('passed', 'errors', 'snapshot_count', 'physical_satellites', 'ground_nodes', 'event_frame_count')
    checks = {key: result['checks'][key] for key in keys if key in result['checks']}
    print(json.dumps({'checks': checks, 'output': str(output)},
                     ensure_ascii=False), flush=True)
    return result


def main():
    from runtime_environment import activate
    activate()
    from paths import owned
    from astermind.builder.physical.network import run_network_experiment
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare-fast-frames', action='store_true')
    args=p.parse_args();output=owned(args.out)
    prepare_physical(args.manifest, output, run_network_experiment, prepare_fast_frames=args.prepare_fast_frames)

if __name__=='__main__':main()
