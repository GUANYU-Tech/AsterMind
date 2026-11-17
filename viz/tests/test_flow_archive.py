from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

VIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(VIZ))
from flow_archive import FlowArchive, search_paths, query_path, ACTIVE, ACTIVE_LOCK


class FlowArchiveTests(unittest.TestCase):
    def test_live_query_and_final_snapshot_keep_identical_paths(self):
        temporary=VIZ/'.runtime/tmp'
        temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            output=Path(folder)
            self.assertTrue(output.resolve().is_relative_to(temporary.resolve()))
            archive=FlowArchive(output)
            first=dict(id='flow-1',request_id='request-1',stage='input',start_order=0,terminal_order=3)
            second=dict(id='flow-2',request_id='request-2',stage='output',start_order=4,terminal_order=7)
            archive.append(first);archive.flush()
            self.assertEqual(search_paths(output,'request-',10)['paths'],[first])
            archive.publish(force=True)
            archive.append(second);archive.flush()
            self.assertEqual(search_paths(output,'request-',10)['paths'],[first,second])
            self.assertEqual(search_paths(output,'request-',4)['paths'],[first])
            self.assertEqual(search_paths(output,'request_%',10)['paths'],[])
            before=archive.database.total_changes
            archive.append(first);archive.flush()
            self.assertEqual(archive.database.total_changes,before,'unchanged exports rewrote the index')
            archive.publish(force=True)
            with sqlite3.connect(archive.path) as saved:
                self.assertEqual(saved.execute('PRAGMA integrity_check').fetchone()[0],'ok')
                self.assertEqual(saved.execute('SELECT count(*) FROM paths').fetchone()[0],2)
            saved.close()
            with ACTIVE_LOCK:ACTIVE.pop(str(archive.path.resolve()),None)
            self.assertEqual(query_path(output),archive.path)
            self.assertEqual(search_paths(output,'request-',10)['paths'],[first,second])
            archive.database.close();archive.database=None
            if archive.temporary is not None:
                archive.temporary.cleanup();archive.temporary=None


if __name__=='__main__':unittest.main()
