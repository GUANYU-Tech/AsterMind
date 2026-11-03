"""Corrupt block input fails closed; returned metadata cannot poison shared state."""
import json
from pathlib import Path
import sqlite3
import struct
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import unittest
from unittest.mock import patch
import zlib

from orbitserve.runtime.state.attribute_cache import AttributeBlock
from orbitserve.runtime.state.attribute_codec import MAGIC, decode, encode
from orbitserve.runtime.state.physical_dataset import PhysicalAttributes, SCHEMA, _decode_block, _encode_block


def rows():
    return [(time_s, {'geometry': {'distance_km': 1., 'notes': ['original']},
                      'directions': [{'source_node_id': 'a', 'target_node_id': 'b',
                                      'bandwidth_bps': 10., 'sources': ['reference']}]})
            for time_s in (0., 1., 2.)]


class PhysicalBlockContractTests(unittest.TestCase):
    def test_concurrent_preview_queries_decode_and_charge_a_block_once(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'shared.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE metadata (value TEXT)')
                db.execute('INSERT INTO metadata VALUES (?)', (json.dumps({'schema_version': SCHEMA}),))
                db.execute('CREATE TABLE attributes (link TEXT,time REAL,value BLOB)')
                db.execute('INSERT INTO attributes VALUES (?,?,?)', ('ab', 0., zlib.compress(encode(AttributeBlock(rows())))))
            reader = PhysicalAttributes(path)
            original = reader._read_block

            def slow_read(link, start_s):
                sleep(.02)  # Force peers to query while a disk/decode miss is in flight.
                return original(link, start_s)

            with patch.object(reader, '_read_block', side_effect=slow_read), ThreadPoolExecutor(max_workers=8) as pool:
                values = list(pool.map(lambda unused: reader.geometry_at('ab', .5), range(32)))
            self.assertTrue(all(value == rows()[0][1]['geometry'] for value in values))
            self.assertEqual(reader.block_loads, 1)
            self.assertEqual(reader.decoded_bytes, sum(block.nbytes for block in reader._blocks.values()))
    def test_round_trip_and_nested_metadata_isolation(self):
        block = decode(encode(AttributeBlock(rows())))
        for index, (_, expected) in enumerate(rows()):
            self.assertEqual(block.at(index), expected)
        value = block.at(0)
        value['geometry']['notes'].append('poisoned')
        value['directions'][0]['sources'].clear()
        self.assertEqual(block.at(0), rows()[0][1])
        self.assertEqual(block.at(1), rows()[1][1])

    def test_negative_column_references_and_nonfinite_times_are_rejected(self):
        data = encode(AttributeBlock(rows()))
        offset = len(MAGIC) + 4
        size = struct.unpack_from('<I', data, len(MAGIC))[0]
        header = json.loads(data[offset:offset + size])
        header['times'] = -len(header['arrays'])
        metadata = json.dumps(header).encode()
        invalid = MAGIC + struct.pack('<I', len(metadata)) + metadata + data[offset + size:]
        with self.assertRaisesRegex(ValueError, 'column reference'):
            decode(invalid)
        for time_s in (float('nan'), float('inf'), -1., 1.):
            block = AttributeBlock(rows())
            block.times[0] = time_s
            with self.subTest(time_s=time_s), self.assertRaisesRegex(ValueError, 'block times'):
                decode(encode(block))

    def test_truncated_trailing_and_oversized_compressed_blocks(self):
        valid = zlib.compress(encode(AttributeBlock(rows())))
        self.assertEqual(_decode_block(valid).at(0), rows()[0][1])
        for invalid in (valid[:-1], valid + b'junk', b'not zlib'):
            with self.subTest(invalid=invalid[:8]), self.assertRaises(ValueError):
                _decode_block(invalid)
        with patch('orbitserve.runtime.state.physical_dataset.MAX_BLOCK_BYTES', 1024):
            with self.assertRaisesRegex(ValueError, 'size limit'):
                _decode_block(zlib.compress(b' ' * 1025))
            oversized = rows()
            oversized[0][1]['geometry']['notes'] = ['x' * 4096]
            with self.assertRaisesRegex(ValueError, 'size limit'):
                _encode_block(oversized)

    def test_public_query_results_are_independent_and_nonfinite_queries_fail(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'state.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE metadata (value TEXT)')
                db.execute('INSERT INTO metadata VALUES (?)', (json.dumps({'schema_version': SCHEMA}),))
                db.execute('CREATE TABLE attributes (link TEXT,time REAL,value BLOB)')
                db.execute('INSERT INTO attributes VALUES (?,?,?)', ('ab', 0., zlib.compress(encode(AttributeBlock(rows())))))
            reader = PhysicalAttributes(path)
            value = reader.at('ab', .5)
            value['geometry']['distance_km'] = 999.
            value['geometry']['notes'].append('changed')
            self.assertEqual(reader.at('ab', .5), rows()[0][1])
            self.assertEqual(reader.geometry_at('ab', .5), rows()[0][1]['geometry'])
            for time_s in (float('nan'), float('inf'), -1.):
                with self.subTest(time_s=time_s), self.assertRaises(ValueError):
                    reader.at('ab', time_s)
