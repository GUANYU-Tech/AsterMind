"""Bounded-memory indexing and per-batch reads of large saved replay JSON."""
import json
from pathlib import Path
import sqlite3
import threading
import zlib

LIMIT = 64 * 1024 * 1024
_lock = threading.Lock()
_building = set()


class JsonStream:
    def __init__(self, stream, chunk=65536):
        self.stream, self.chunk = stream, chunk
        self.buffer, self.pos, self.eof = '', 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        self.buffer = self.buffer[self.pos:]
        self.pos = 0
        more = self.stream.read(self.chunk)
        self.eof = not more
        self.buffer += more

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():self.pos += 1
            if self.pos < len(self.buffer):return self.buffer[self.pos]
            if self.eof:return ''
            self.fill()

    def take(self, token):
        if self.peek() != token:raise ValueError('Invalid replay JSON delimiter')
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                # A numeric token at a chunk boundary may not be complete yet.
                if not self.eof and (end == len(self.buffer) or self.buffer[end] not in ' \t\r\n,]}:'):
                    self.fill();continue
                self.pos = end
                return value
            except json.JSONDecodeError:
                if self.eof:raise
                self.fill()

    def fields(self):
        self.take('{')
        while self.peek() != '}':
            key = self.value();self.take(':')
            if self.peek() == '[':
                self.take('[')
                yield key, 'array', None
                while self.peek() != ']':
                    yield key, 'item', self.value()
                    if self.peek() != ']':self.take(',')
                self.take(']')
            else:yield key, 'value', self.value()
            if self.peek() != '}':self.take(',')
        self.take('}')
        if self.peek():raise ValueError('Trailing data in replay JSON')


def packed(value):
    return zlib.compress(json.dumps(value,separators=(',',':')).encode(),1)


def unpacked(value):
    return json.loads(zlib.decompress(value))


def build_index(source):
    source = Path(source)
    target = source.with_suffix('.pages.sqlite3')
    temporary = target.with_suffix('.tmp')
    stat = source.stat()
    with sqlite3.connect(temporary) as db:
        db.executescript('''
        DROP TABLE IF EXISTS meta; DROP TABLE IF EXISTS batches; DROP TABLE IF EXISTS steps;
        DROP TABLE IF EXISTS transfers; DROP TABLE IF EXISTS hops; DROP TABLE IF EXISTS events;
        CREATE TABLE meta (id INTEGER PRIMARY KEY, data BLOB, size INTEGER, stamp INTEGER);
        CREATE TABLE batches (id INTEGER PRIMARY KEY, cell TEXT, step TEXT, offset INTEGER, start REAL, end REAL, data BLOB);
        CREATE TABLE steps (cell TEXT, step TEXT, data BLOB);
        CREATE TABLE transfers (id TEXT PRIMARY KEY, correlation TEXT, data BLOB);
        CREATE TABLE hops (flow TEXT, data BLOB);
        CREATE TABLE events (time REAL, data BLOB);
        ''')
        meta, calls, current, offset, index, serial = {}, [], None, 0, 0, 0
        def flush():
            nonlocal serial
            if not calls:return
            db.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?)',
                       (serial,*current,offset,min(c['start'] for c in calls),max(c['end'] for c in calls),packed(calls)))
            serial += 1
            calls.clear()
        count = 0
        with source.open(encoding='utf-8') as stream:
            for key, kind, value in JsonStream(stream).fields():
                if kind == 'array':
                    if key not in {'calls','hops','steps','transfers','events'}:meta[key]=[]
                    continue
                if key == 'calls':
                    batch = value['cell'],value['batch']
                    if batch != current:flush();current=batch;offset=index
                    calls.append(value);index+=1
                elif key == 'steps':db.execute('INSERT INTO steps VALUES (?,?,?)',(value['cell'],value['id'],packed(value)))
                elif key == 'transfers':db.execute('INSERT INTO transfers VALUES (?,?,?)',(value['id'],value.get('correlation_id'),packed(value)))
                elif key == 'hops':db.execute('INSERT INTO hops VALUES (?,?)',(value.get('flow'),packed(value)))
                elif key == 'events':db.execute('INSERT INTO events VALUES (?,?)',(value['time'],packed(value)))
                elif kind == 'item':meta[key].append(value)
                else:meta[key]=value
                count += 1
                if count % 10000 == 0:db.commit()
        flush()
        db.execute('INSERT INTO meta VALUES (1,?,?,?)',(packed(meta),stat.st_size,stat.st_mtime_ns))
        db.executescript('CREATE INDEX hop_flow ON hops(flow); CREATE INDEX event_time ON events(time); CREATE INDEX step_batch ON steps(cell,step); CREATE INDEX transfer_correlation ON transfers(correlation);')
        db.commit()
    temporary.replace(target)
    return target


def ready_index(source):
    source=Path(source);target=source.with_suffix('.pages.sqlite3')
    if target.is_file():return target
    with _lock:
        if target.is_file():return target
        if str(source) not in _building:
            _building.add(str(source))
            def build():
                try:build_index(source)
                except Exception:
                    import traceback
                    target.with_suffix('.error.log').write_text(traceback.format_exc())
                finally:
                    with _lock:_building.discard(str(source))
            threading.Thread(target=build,name='index-replay',daemon=True).start()
    raise ValueError('正在整理按批次读取的回放索引，请稍后重试')


def catalog(source):
    with sqlite3.connect(ready_index(source)) as db:
        return [dict(id=f'batch-{i}',title=f'{cell} {step}',file=f'batch-{i}.json',paged=True)
                for i,cell,step in db.execute('SELECT id,cell,step FROM batches ORDER BY start,id')]


def batch(source, identity):
    with sqlite3.connect(ready_index(source)) as db:
        row=db.execute('SELECT cell,step,offset,start,end,data FROM batches WHERE id=?',(identity,)).fetchone()
        if row is None:raise ValueError('计算批次不存在')
        cell,step,offset,start,end,blob=row
        result=unpacked(db.execute('SELECT data FROM meta WHERE id=1').fetchone()[0])
        calls=unpacked(blob)
        for call in calls:
            call['deps']=[d-offset if d>=0 else -1 for d in call['deps']]
            if any(d>=len(calls) or d < -1 for d in call['deps']):raise ValueError('算子依赖超出当前批次')
        flows={f for c in calls for phase in c['phases'] for f in phase}
        correlations=[cell+'/'+step,cell+'/'+step+'/input']
        transfer_rows=db.execute('SELECT data FROM transfers WHERE correlation IN (?,?)',correlations).fetchall()
        transfers={v['id']:v for (raw,) in transfer_rows for v in [unpacked(raw)]}
        for flow in flows:
            found=db.execute('SELECT data FROM transfers WHERE id=?',(flow,)).fetchone()
            if found:transfers[flow]=unpacked(found[0])
        hops=[unpacked(raw) for flow in transfers for (raw,) in db.execute('SELECT data FROM hops WHERE flow=?',(flow,))]
        steps=[unpacked(raw) for (raw,) in db.execute('SELECT data FROM steps WHERE cell=? AND step=?',(cell,step))]
        result.update(calls=calls,transfers=list(transfers.values()),hops=hops,steps=steps,
            events=[unpacked(raw) for (raw,) in db.execute('SELECT data FROM events WHERE time BETWEEN ? AND ? ORDER BY time',(start,end))],
            title=cell+' '+step,page_scope=dict(kind='batch',start=start,end=end),
            end=end)
        return result
