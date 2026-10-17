#!/usr/bin/env python3
"""Set scalar keys inside one top-level-indented block of a hardware YAML.

Line-based on purpose: the hardware configs carry extensive provenance
comments that a YAML round-trip would drop.  A block is located by its
exact ``<indent><name>:`` header line; each ``key: value`` is replaced at
the block's child indent only, and the call fails if a key is not found
exactly once.

Usage:
    python scripts/hardware/set_yaml_block_values.py <yaml> <block> key=value [...]
"""
import re
import sys


def set_block_values(path, block, assignments):
    lines = open(path).read().split('\n')
    header = [i for i, line in enumerate(lines)
              if re.fullmatch(r'\s*' + re.escape(block) + r':\s*', line)]
    if len(header) != 1:
        raise SystemExit(f'block {block!r} found {len(header)} times')
    start = header[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    child_indent = None
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line.lstrip().startswith('#'):
            current = len(line) - len(line.lstrip())
            if current <= indent:
                break
            if child_indent is None:
                child_indent = current
        end += 1
    for key, value in assignments:
        pattern = re.compile(r'^' + ' ' * child_indent + re.escape(key) + r':(\s*)([^#]*?)(\s*(#.*)?)$')
        hits = [i for i in range(start + 1, end) if pattern.match(lines[i])]
        if len(hits) != 1:
            raise SystemExit(f'{block}.{key} found {len(hits)} times')
        match = pattern.match(lines[hits[0]])
        lines[hits[0]] = (' ' * child_indent + key + ':' + (match.group(1) or ' ')
                          + value + (match.group(3) or ''))
    open(path, 'w').write('\n'.join(lines))


if __name__ == '__main__':
    yaml_path, block_name = sys.argv[1], sys.argv[2]
    set_block_values(yaml_path, block_name,
                     [tuple(item.split('=', 1)) for item in sys.argv[3:]])
