import unittest

import pandas as pd

from besttime.cleaning import SchemaError, clean_fares
from tests.helpers import clean, raw_rows, snapshot


class LeadTimeTests(unittest.TestCase):
    def test_lead_time_is_departure_minus_observation(self):
        # 2025-02-08 -> 2025-03-10: 20 days left in Feb (28-8) + 10 = 30
        df, _ = clean([{"observation_date": "2025-02-08", "departure_date": "2025-03-10"}])
        self.assertEqual(int(df.loc[0, "lead_days"]), 30)

    def test_same_day_is_zero_days(self):
        df, _ = clean([{"observation_date": "2025-03-10", "departure_date": "2025-03-10"}])
        self.assertEqual(int(df.loc[0, "lead_days"]), 0)

    def test_time_of_day_is_ignored(self):
        df, _ = clean([{"observation_date": "2025-03-09 23:59:00", "departure_date": "2025-03-10 00:05:00"}])
        self.assertEqual(int(df.loc[0, "lead_days"]), 1)

    def test_observation_after_departure_is_excluded(self):
        df, rep = clean([{"observation_date": "2025-03-16", "departure_date": "2025-03-15"}])
        self.assertTrue(df.empty)
        self.assertEqual(rep.excluded["observation_after_departure"], 1)


class InvalidDataTests(unittest.TestCase):
    def test_invalid_fares_excluded(self):
        rows = [{"fare": "0"}, {"fare": "-50"}, {"fare": "abc"}, {"fare": None},
                {"fare": "3000000"}, {"fare": "12,500"}, {"fare": "900"}]
        df, rep = clean(rows)
        self.assertEqual(len(df), 1)
        self.assertEqual(float(df.loc[0, "fare"]), 900.0)
        self.assertEqual(rep.excluded["invalid_fare"], 6)

    def test_missing_dates_excluded(self):
        rows = [{"observation_date": None}, {"departure_date": ""}, {"departure_date": "not-a-date"}, {}]
        df, rep = clean(rows)
        self.assertEqual(len(df), 1)
        self.assertEqual(rep.excluded["missing_or_invalid_date"], 3)

    def test_missing_and_mixed_currency_excluded_without_conversion(self):
        rows = [{"currency": None}, {"currency": ""}, {"currency": "USD"}, {"currency": " pkr "}]
        df, rep = clean(rows)
        self.assertEqual(len(df), 1)  # " pkr " is normalised to PKR
        self.assertEqual(rep.excluded["missing_currency"], 2)
        self.assertEqual(rep.excluded["other_currency"], 1)

    def test_route_problems_excluded(self):
        rows = [{"origin": None}, {"destination": " "}, {"origin": "LHE", "destination": "lhe"}, {}]
        df, rep = clean(rows)
        self.assertEqual(len(df), 1)
        self.assertEqual(rep.excluded["missing_route"], 2)
        self.assertEqual(rep.excluded["same_origin_destination"], 1)

    def test_codes_are_normalised(self):
        df, _ = clean([{"origin": " lhe ", "destination": "khi"}])
        self.assertEqual((df.loc[0, "origin"], df.loc[0, "destination"]), ("LHE", "KHI"))

    def test_each_excluded_row_counted_once(self):
        rows = [{"currency": None, "fare": "abc"}, {}]  # fails two rules; counted once
        df, rep = clean(rows)
        self.assertEqual(rep.rows_in, 2)
        self.assertEqual(rep.rows_out, 1)
        self.assertEqual(sum(rep.excluded.values()), rep.rows_in - rep.rows_out)


class DuplicateTests(unittest.TestCase):
    def test_repeated_obs_id_and_identical_content_are_removed(self):
        rows = [
            {"obs_id": "A", "fare": "1000"},
            {"obs_id": "A", "fare": "1000"},   # same obs_id
            {"obs_id": "B", "fare": "1000"},   # same itinerary, day and fare as A
            {"obs_id": "C", "fare": "1100"},   # different fare -> a genuine second observation
        ]
        df, rep = clean(rows)
        self.assertEqual(sorted(df["obs_id"]), ["A", "C"])
        self.assertEqual(rep.excluded["duplicate_record"], 2)

    def test_same_itinerary_different_days_are_not_duplicates(self):
        df, rep = clean([snapshot("C1", 2, 500), snapshot("C1", 10, 500)])
        self.assertEqual(len(df), 2)
        self.assertEqual(rep.excluded["duplicate_record"], 0)


class SchemaAndEmptyTests(unittest.TestCase):
    def test_missing_required_column_raises(self):
        raw = raw_rows([{}]).drop(columns=["currency"])
        with self.assertRaises(SchemaError) as ctx:
            clean_fares(raw, "PKR")
        self.assertIn("currency", str(ctx.exception))

    def test_empty_dataset_is_handled(self):
        raw = raw_rows([{}]).iloc[0:0]
        df, rep = clean_fares(raw, "PKR")
        self.assertTrue(df.empty)
        self.assertEqual((rep.rows_in, rep.rows_out), (0, 0))

    def test_itinerary_id_derived_when_absent(self):
        df, _ = clean([{"carrier": "C9", "departure_date": "2025-03-15"}])
        self.assertEqual(df.loc[0, "itinerary_id"], "LHE|KHI|C9|2025-03-15")

    def test_missing_carrier_becomes_unknown(self):
        df, _ = clean([{"carrier": None}])
        self.assertEqual(df.loc[0, "carrier"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
