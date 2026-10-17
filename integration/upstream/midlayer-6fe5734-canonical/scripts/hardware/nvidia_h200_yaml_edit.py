#!/usr/bin/env python3
"""Comment-preserving edits of the H200 hardware yaml used by the H200 port.

All edits fail loudly unless their target occurs exactly once.  Values are
always taken from `microbench/nvidia_h200/references/*_sm90.json` by the
per-package apply scripts; this module only knows where they go.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
YAML = ROOT / 'configs/hardware/nvidia_h200.yaml'
REFS = ROOT / 'microbench/nvidia_h200/references'
SM_HZ = 1.98e9


def ref(name):
    return json.loads((REFS / name).read_text())


def _replace_line(text, pattern, value):
    regex = re.compile(pattern, re.M)
    hits = regex.findall(text)
    if len(hits) != 1:
        raise SystemExit(f'{pattern!r} matched {len(hits)} times')
    return regex.sub(lambda m: f'{m.group(1)}{value}', text, count=1)


def set_key(text, key, value, indent=2):
    """Set a unique `<indent spaces><key>: value` line (keeps no trailing comment)."""
    return _replace_line(text, r'^(' + ' ' * indent + re.escape(key) + r':[ \t]*)[^\n#]*?(?:[ \t]*#[^\n]*)?$', value)


def _block(text, header_regex, child_marker):
    match = re.search(header_regex, text, re.M)
    if match is None or len(re.findall(header_regex, text, re.M)) != 1:
        raise SystemExit(f'block {header_regex!r} not unique')
    start = match.start()
    nxt = re.search(child_marker, text[match.end():], re.M)
    end = match.end() + nxt.start() if nxt else len(text)
    return start, end


def set_component_throughput(text, name, value):
    start, end = _block(text, r'^  - name: ' + re.escape(name) + r'\s*$', r'^  - name: |^\S')
    block = _replace_line(text[start:end], r'^(    throughput_per_clk:[ \t]*)[^\n#]*?(?:[ \t]*#[^\n]*)?$', value)
    return text[:start] + block + text[end:]


def set_service(text, service, capacity, evidence_ref=None):
    start, end = _block(text, r'^    - service: ' + re.escape(service) + r'\s*$', r'^    - service: |^  \S|^\S')
    block = _replace_line(text[start:end],
                          r'^(      logical_byte_capacity_per_s:[ \t]*)[^\n#]*?(?:[ \t]*#[^\n]*)?$', capacity)
    if evidence_ref is not None:
        block = _replace_line(block, r'^(      evidence_ref:[ \t]*)[^\n#]*?(?:[ \t]*#[^\n]*)?$', evidence_ref)
    return text[:start] + block + text[end:]


def load():
    return YAML.read_text()


def save(text):
    YAML.write_text(text)
