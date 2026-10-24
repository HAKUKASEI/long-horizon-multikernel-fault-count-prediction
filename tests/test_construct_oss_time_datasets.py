from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from construct_oss_time_datasets import Record, build_sequence, parse_utc, read_array, validate


class ConstructionTests(unittest.TestCase):
    def setUp(self):
        self.origin = datetime(2023, 1, 31, 23, 59, 50, tzinfo=timezone.utc)

    def test_sort_origin_exclusion_and_calendar_boundary(self):
        records = [Record(3, self.origin + timedelta(seconds=25), "bug"),
                   Record(1, self.origin, "bug"),
                   Record(2, self.origin + timedelta(seconds=10), "question")]
        ordered, times = build_sequence(records)
        self.assertEqual([record.issue_number for record in ordered], [1, 2, 3])
        self.assertEqual(times, [10, 25])

    def test_duplicate_issues_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_sequence([Record(1, self.origin, "bug"), Record(1, self.origin + timedelta(seconds=1), "bug")])

    def test_zero_model_time_is_not_silently_removed(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            build_sequence([Record(1, self.origin, "bug"), Record(2, self.origin, "bug")])

    def test_origin_alone_is_not_a_dataset(self):
        with self.assertRaisesRegex(ValueError, "At least"):
            build_sequence([Record(1, self.origin, "bug")])

    def test_utc_and_whole_second_requirements(self):
        self.assertEqual(parse_utc("2023-01-31T23:59:50Z"), self.origin)
        for value in ("invalid", "2023-01-31T23:59:50", "2023-01-31T23:59:50+09:00", "2023-01-31T23:59:50.1Z"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_utc(value)

    def test_array_without_dimension_metadata(self):
        sheet = SimpleNamespace(title="Array", max_column=None,
                                iter_rows=lambda **kwargs: iter([(10,), (25,)]))
        self.assertEqual(read_array(sheet), [10, 25])

    def test_non_numeric_and_extra_columns_are_rejected(self):
        for rows in ([("header",)], [(1, 2)], [(None,)], [(True,)], [(1.5,)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                read_array(SimpleNamespace(title="Array", iter_rows=lambda **kwargs: iter(rows)))

    def test_every_released_array_matches_raw_timestamps(self):
        results = validate(ROOT / "data/JSS_OSS_time_datasets_OSS1_OSS4.xlsx", ROOT / "data/OSS")
        self.assertEqual([result["model_events"] for result in results], [278, 278, 183, 290])
        self.assertEqual(sum(result["model_events"] for result in results), 1029)
        self.assertEqual(sum(result["raw_records"] for result in results), 1033)


if __name__ == "__main__":
    unittest.main()
