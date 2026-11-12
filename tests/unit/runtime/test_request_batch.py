"""User workload import is atomic and preserves actual native input contracts."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from simulator.request_batch import parse_rows, prepare_batch, seconds_ns


class RequestBatchTests(unittest.TestCase):
    def setUp(self):
        self.experiment = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "simulator/examples/two_cells.json"
            ).read_text()
        )

    def batch(self, content, format="csv", mode="replace", **updates):
        return prepare_batch(
            dict(
                experiment=self.experiment,
                content=content,
                format=format,
                mode=mode,
                **updates,
            )
        )

    def generated(self, **options):
        return prepare_batch(
            dict(
                experiment=self.experiment,
                kind="generate",
                mode="replace",
                options=dict(window_s=0.5, **options),
            )
        )

    def test_csv_bom_aliases_and_seconds_are_normalized(self):
        value = self.batch(
            "\ufeff请求ID,到达时间(s),地面接入,输入token,输出token,目标Cell\nreq-1,0.1,ground,16,2,cellA\n"
        )
        request = value["requests"][0]
        self.assertEqual(request["arrival_ns"], 100_000_000)
        self.assertEqual(request["user_device"], "user")
        self.assertEqual(request["input_bytes"], 64)
        self.assertEqual(request["cell"], "cellA")

    def test_native_json_and_jsonl_preserve_explicit_payload(self):
        row = dict(
            id="req-1",
            arrival_ns=1,
            user_device="user",
            input_tokens=16,
            output_tokens=2,
            input_bytes=0,
        )
        for format, text in (
            ("json", json.dumps([row])),
            ("json", json.dumps(dict(requests=[row]))),
            ("jsonl", json.dumps(row) + "\n"),
        ):
            with self.subTest(format=format):
                value = self.batch(text, format)
                self.assertEqual(value["requests"][0]["input_bytes"], 0)
                self.assertEqual(value["total_count"], 1)

    def test_append_rejects_duplicates_but_replace_can_reuse_an_id(self):
        row = deepcopy(self.experiment["requests"][0])
        with self.assertRaisesRegex(ValueError, "重复"):
            self.batch(json.dumps([row]), "json", "append")
        value = self.batch(json.dumps([row]), "json", "replace")
        self.assertEqual(value["total_count"], 1)

    def test_failed_import_never_partially_changes_the_existing_scene(self):
        original = deepcopy(self.experiment)
        with self.assertRaisesRegex(ValueError, "第 3 行"):
            self.batch(
                "id,arrival_s,user_device,input_tokens\ngood,0,user,16\nbad,0,missing,16\n"
            )
        self.assertEqual(self.experiment, original)

    def test_replacement_can_repair_invalid_old_requests(self):
        self.experiment["requests"][0]["input_tokens"] = -1
        value = self.batch("id,user_device,input_tokens\nrepaired,user,16\n")
        self.assertEqual(value["requests"][0]["id"], "repaired")

    def test_bad_csv_columns_and_ambiguous_fields_rejected(self):
        for text in (
            "id,id\nx,y\n",
            "id,user_device,input_tokens\nx,user,16,extra\n",
            'id,user_device,input_tokens\n"unclosed,user,16\n',
            "id,request_id,user_device,input_tokens\nx,y,user,16\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.batch(text)

    def test_invalid_values_and_context_limits_rejected(self):
        base = dict(id="req", user_device="user", input_tokens=16, output_tokens=2)
        for change in (
            dict(arrival_ns=-1),
            dict(arrival_s=1),
            dict(arrival_s=float("nan")),
            dict(input_tokens=0),
            dict(input_tokens=1000),
            dict(user_device="missing"),
            dict(cell="unknown"),
            dict(unrecognized="value"),
            dict(arrival_ns=0, arrival_s=0),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.batch(json.dumps([{**base, **change}]), "json")

    def test_empty_oversize_and_duplicate_json_keys_rejected(self):
        for content, format in (
            ("", "csv"),
            ("[]", "json"),
            ('[{"id":"a","id":"b"}]', "json"),
            ("x" * 8_000_001, "csv"),
        ):
            with self.subTest(format=format), self.assertRaises(ValueError):
                parse_rows(content, format)

    def test_generation_is_reproducible_and_inside_the_window(self):
        a, b = self.generated(count=1000, seed=42), self.generated(count=1000, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(a["total_count"], 1000)
        self.assertEqual(len({row["id"] for row in a["requests"]}), 1000)
        arrivals = [row["arrival_ns"] for row in a["requests"]]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertTrue(all(0 <= at < 500_000_000 for at in arrivals))
        self.assertTrue(all(row["input_bytes"] == 128 for row in a["requests"]))

    def test_spaced_and_simultaneous_are_distinct_distributions(self):
        spaced = self.generated(count=4, distribution="spaced")
        simultaneous = self.generated(count=4, distribution="simultaneous")
        self.assertEqual(
            [row["arrival_ns"] for row in spaced["requests"]],
            [0, 125_000_000, 250_000_000, 375_000_000],
        )
        self.assertEqual(
            [row["arrival_ns"] for row in simultaneous["requests"]], [0] * 4
        )

    def test_append_generation_avoids_existing_identifiers(self):
        self.experiment["requests"][0]["id"] = "request-1"
        value = prepare_batch(
            dict(
                experiment=self.experiment,
                kind="generate",
                mode="append",
                options=dict(count=10, window_s=0.5),
            )
        )
        self.assertEqual(value["total_count"], len(self.experiment["requests"]) + 10)
        self.assertEqual(
            len({row["id"] for row in value["requests"]}), value["total_count"]
        )

    def test_invalid_generation_parameters_rejected(self):
        for options in (
            dict(count=0),
            dict(count=True),
            dict(count=100001),
            dict(window_s=2),
            dict(seed=-1),
            dict(distribution="unknown"),
            dict(input_tokens=1000),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                prepare_batch(
                    dict(
                        experiment=self.experiment,
                        kind="generate",
                        options={"window_s": 0.5, **options},
                    )
                )

    def test_missing_parent_history_and_migration_trigger_are_rejected(self):
        row = dict(
            id="req",
            user_device="user",
            input_tokens=16,
            session_id="s",
            parent_context_id="missing",
        )
        with self.assertRaisesRegex(ValueError, "历史"):
            self.batch(json.dumps([row]), "json")
        self.experiment["migrations"] = [
            dict(
                id="m",
                after_request="removed",
                source_cell="cellA",
                target_cell="cellB",
                session_id="s",
                context_id="ctx",
            )
        ]
        with self.assertRaises(ValueError):
            self.batch("id,user_device,input_tokens\nnew,user,16\n")

    def test_sub_nanosecond_arrivals_round_up_once(self):
        self.assertEqual(seconds_ns("0.0000000001"), 1)
        self.assertEqual(seconds_ns("0.100000001"), 100_000_001)
        with self.assertRaises(ValueError):
            seconds_ns(True)

    def test_generated_batch_must_fit_the_editor_save_transport(self):
        with self.assertRaisesRegex(ValueError, "10 MB"):
            self.generated(count=30000)
