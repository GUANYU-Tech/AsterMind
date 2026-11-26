"""Indexed completed paths; bounded memory and no full event-log reads in HTTP."""
import json
import os
from pathlib import Path
from shutil import copyfile, copyfileobj
import sqlite3
import sys
import tempfile
import threading
import time
import weakref
from contextlib import closing

ACTIVE = weakref.WeakValueDictionary()
ACTIVE_LOCK = threading.RLock()


class FlowArchive:
    def __init__(self, output):
        self.path=Path(output)/'native-replay'/'flow-paths.sqlite3'
        self.pending=[]
        self.database=None
        self.lock=threading.RLock()
        self.dirty=False
        self.published_at=0
        self.temporary=None
        self.working_path=self.path
        if sys.platform=='linux':
            self._select_workspace()
        with ACTIVE_LOCK:
            ACTIVE[str(self.path.resolve())]=self

    def _select_workspace(self):
        temporary=None
        try:
            temporary=tempfile.TemporaryDirectory(prefix='orbitserve-path-index-',
                                                  dir=os.environ.get('TMPDIR'), ignore_cleanup_errors=True)
            working_path=Path(temporary.name)/'paths.sqlite3'
            if self.path.is_file():
                copyfile(self.path,working_path)
        except OSError:
            if temporary is not None:
                temporary.cleanup()
            return
        self.temporary=temporary
        self.working_path=working_path

    def __del__(self):
        if self.database is not None:
            self.database.close()
        if self.temporary is not None:
            self.temporary.cleanup()

    def append(self, row):
        self.pending.append((row['id'],row.get('request_id'),row.get('stage'),row['start_order'],row['terminal_order'],json.dumps(row,ensure_ascii=False)))
        # This is a rebuildable query index; live captures still flush at each
        # published boundary. Batch export avoids one disk transaction per 128
        # paths while retaining a bounded pending buffer.
        if len(self.pending)>=1024:self.flush()

    def flush(self):
        with self.lock:
            self._flush()

    def _flush(self):
        if not self.pending:return
        if self.database is None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            database=sqlite3.connect(self.working_path,timeout=30,check_same_thread=False)
            # Keep the writer and its page cache across batches. WAL is avoided
            # because the workspace can be on a WSL-mounted Windows drive.
            database.execute('PRAGMA cache_size=-16384')
            database.execute('CREATE TABLE IF NOT EXISTS paths (id TEXT PRIMARY KEY, request_id TEXT, stage TEXT, start_order INTEGER, terminal_order INTEGER, data TEXT)')
            database.execute('CREATE INDEX IF NOT EXISTS paths_request ON paths(request_id COLLATE NOCASE)')
            database.execute('CREATE INDEX IF NOT EXISTS paths_id_prefix ON paths(id COLLATE NOCASE)')
            self.database=database
        with self.database as database:
            # Recording export visits the same completed paths again. Keep
            # unchanged rows and their indexes instead of rewriting all pages.
            database.executemany('''INSERT INTO paths VALUES (?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET request_id=excluded.request_id,
                stage=excluded.stage,start_order=excluded.start_order,
                terminal_order=excluded.terminal_order,data=excluded.data
                WHERE paths.data<>excluded.data''',self.pending)
        self.pending.clear()
        self.dirty=True

    def publish(self, *, force=False):
        with self.lock:
            self._flush()
            if not self.dirty or self.working_path==self.path:
                return
            if not force and self.path.is_file() and time.monotonic()-self.published_at<10:
                return
            temporary=self.path.with_suffix('.sqlite3.tmp')
            # The private database uses rollback journaling and has committed
            # above. Under the writer lock its main file is a complete SQLite
            # snapshot. Copy sequentially; SQLite.backup writes small random
            # pages to the Windows target and recreates the original bottleneck.
            with self.working_path.open('rb') as source, temporary.open('wb') as destination:
                copyfileobj(source,destination,length=1024*1024)
            temporary.replace(self.path)
            self.dirty=False
            self.published_at=time.monotonic()


def query_path(output):
    path=Path(output)/'native-replay'/'flow-paths.sqlite3'
    with ACTIVE_LOCK:
        active=ACTIVE.get(str(path.resolve()))
    return active.working_path if active is not None else path


def search_paths(output, query, through):
    if len(query)>120:raise ValueError('查询内容过长')
    path=query_path(output)
    if not query or not path.is_file():return dict(paths=[],available=path.is_file())
    literal=query.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%'
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=10)) as database:
        rows=database.execute("SELECT data FROM paths WHERE start_order<? AND (request_id LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\') ORDER BY start_order LIMIT 100",(through,literal,literal)).fetchall()
    return dict(paths=[json.loads(row[0]) for row in rows],available=True)
