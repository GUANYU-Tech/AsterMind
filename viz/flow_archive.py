"""Indexed completed paths; bounded memory and no full event-log reads in HTTP."""
import json
from pathlib import Path
import sqlite3


class FlowArchive:
    def __init__(self, output):
        self.path=Path(output)/'native-replay'/'flow-paths.sqlite3'
        self.pending=[]

    def append(self, row):
        self.pending.append((row['id'],row.get('request_id'),row.get('stage'),row['start_order'],row['terminal_order'],json.dumps(row,ensure_ascii=False)))
        if len(self.pending)>=128:self.flush()

    def flush(self):
        if not self.pending:return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path,timeout=30) as database:
            database.execute('CREATE TABLE IF NOT EXISTS paths (id TEXT PRIMARY KEY, request_id TEXT, stage TEXT, start_order INTEGER, terminal_order INTEGER, data TEXT)')
            database.execute('CREATE INDEX IF NOT EXISTS paths_request ON paths(request_id COLLATE NOCASE)')
            database.execute('CREATE INDEX IF NOT EXISTS paths_id_prefix ON paths(id COLLATE NOCASE)')
            database.executemany('INSERT OR REPLACE INTO paths VALUES (?,?,?,?,?,?)',self.pending)
        self.pending.clear()


def search_paths(output, query, through):
    if len(query)>120:raise ValueError('查询内容过长')
    path=Path(output)/'native-replay'/'flow-paths.sqlite3'
    if not query or not path.is_file():return dict(paths=[],available=path.is_file())
    literal=query.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%'
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=10) as database:
        rows=database.execute("SELECT data FROM paths WHERE start_order<? AND (request_id LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\') ORDER BY start_order LIMIT 100",(through,literal,literal)).fetchall()
    return dict(paths=[json.loads(row[0]) for row in rows],available=True)
