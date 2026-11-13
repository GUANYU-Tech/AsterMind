"""Verify frozen source identity and every imported/adapted target file."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'astermind/engine/compute/SOURCE.json').read_text())
    unchanged=0
    for row in manifest['files']:
        actual=hashlib.sha256((root/row['target']).read_bytes()).hexdigest()
        assert actual==row['target_sha256'],row['target']
        if args.source:
            assert hashlib.sha256((args.source/row['source']).read_bytes()).hexdigest()==row['source_sha256'],row['source']
        unchanged+=actual==row['source_sha256']
    print(json.dumps(dict(commit=manifest['commit'],files=len(manifest['files']),byte_identical=unchanged,
                          adapted=len(manifest['files'])-unchanged)))
if __name__=='__main__':main()
