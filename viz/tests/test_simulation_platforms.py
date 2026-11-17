from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import simulation_platforms as platforms


class PlatformSelectionTests(unittest.TestCase):
    def test_selector_marks_the_actual_platform_and_is_not_duplicated(self):
        html=b'<html><body><header>Brand</header><main>Scene</main></body></html>'
        for platform in ('lyz','v1'):
            value=platforms.control(html,platform)
            self.assertIn(f'value="{platform}" selected'.encode(),value)
            self.assertEqual(value.count(b'data-platform-control'),1)
            self.assertEqual(platforms.control(value,platform),value)
            self.assertIn(b'aria-label="',value)

    def test_invalid_platform_never_starts_an_engine_and_v1_stays_on_its_server(self):
        with patch.object(platforms.subprocess,'Popen') as launch:
            with self.assertRaises(ValueError):platforms.select('unknown')
            self.assertEqual(platforms.select('v1',8766),{'platform':'v1','url':'http://localhost:8766/'})
            launch.assert_not_called()

    def test_running_legacy_platform_reuses_its_process(self):
        with patch.object(platforms,'health',return_value={'simulation_platform':'lyz'}),patch.object(platforms.subprocess,'Popen') as launch:
            result=platforms.select('lyz')
            self.assertEqual(result['url'],'http://localhost:8798/')
            self.assertEqual(result['commit'],platforms.LYZ_COMMIT)
            launch.assert_not_called()

    def test_legacy_result_fragments_receive_working_selector_assets(self):
        for ending in (b'',b'</html>'):
            html=b'<header>Brand</header><main>Result</main>'+ending
            value=platforms.control(html,'lyz')
            self.assertIn(b'src="/assets/platform_selector.js"',value)
            self.assertIn(b'href="/assets/platform_selector.css"',value)
            self.assertEqual(platforms.control(value,'lyz'),value)


if __name__=='__main__':unittest.main()
