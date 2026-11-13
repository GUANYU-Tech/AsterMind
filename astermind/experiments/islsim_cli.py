"""Generate an ISLSim JSON input without installing or invoking ISLSim."""
import argparse
import json
from pathlib import Path

from astermind.engine import prepare_islsim_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Native parameters or an OrbitServe link profile JSON.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    parameters = json.loads(args.input.read_text(encoding='utf-8')) if args.input else {}
    if isinstance(parameters, dict) and 'islsim_config' in parameters:
        parameters = parameters['islsim_config']
    result = prepare_islsim_config(parameters)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
