import unittest

import numpy as np
import pandas as pd

from besttime.analysis import (INSUFFICIENT_MESSAGE, STATUS_EMPTY, STATUS_INSUFFICIENT, STATUS_OK,
                               analyze_booking_window, assign_window, check_selection, explain,
                               filter_route_month, window_for_lead, window_labels)
from besttime.cleaning import clean_fares
from besttime.config import AnalysisConfig
from besttime.formatting import format_pct
from tests.helpers import SMALL_CFG, clean, manual_dataset, raw_rows, snapshot


def run(rows, **kw):
    df, _ = clean(rows)
    return analyze_booking_window(df, kw.pop("o", "LHE"), kw.pop("d", "KHI"), 2025, 3, "PKR", kw.pop("cfg", SMALL_CFG))


class WindowBoundaryTests(unittest.TestCase):
    def test_boundaries_of_default_windows(self):
        expected = {
            -1: None, 0: "0–3 days", 3: "0–3 days", 4: "4–7 days", 7: "4–7 days",
            8: "8–14 days", 14: "8–14 days", 15: "15–30 days", 30: "15–30 days",
            31: "31–60 days", 60: "31–60 days", 61: "61–90 days", 90: "61–90 days", 91: None,
        }
        for days, label in expected.items():
            with self.subTest(days=days):
                self.assertEqual(window_for_lead(days), label)

    def test_vectorised_assignment_matches_scalar(self):
        days = pd.Series(list(range(-2, 95)))
        got = assign_window(days)
        for d, g in zip(days, got):
            exp = window_for_lead(int(d))
            self.assertTrue((exp is None and pd.isna(g)) or exp == g, (d, exp, g))

    def test_custom_windows_are_supported(self):
        cfg = AnalysisConfig(windows=((0, 7), (8, 30), (31, 90)))
        self.assertEqual(window_labels(cfg), ["0–7 days", "8–30 days", "31–90 days"])
        self.assertEqual(window_for_lead(8, cfg), "8–30 days")

    def test_non_contiguous_windows_rejected(self):
        with self.assertRaises(ValueError):
            AnalysisConfig(windows=((0, 3), (5, 7)))


class FilterTests(unittest.TestCase):
    def test_route_and_month_filtering(self):
        rows = manual_dataset() + [
            {**snapshot("X", 2, 99999), "origin": "KHI", "destination": "LHE"},  # other direction
            snapshot("Y", 2, 88888, dep="2025-04-15"),
            snapshot("Z", 2, 77777, dep="2026-03-15"),
        ]
        df, _ = clean(rows)
        sel = filter_route_month(df, "lhe", "khi", 2025, 3)
        self.assertEqual(len(sel), 15)
        self.assertTrue((sel["origin"] == "LHE").all())
        self.assertNotIn(99999, sel["fare"].tolist())

    def test_check_selection_codes(self):
        df, _ = clean(manual_dataset())
        self.assertEqual(check_selection(df, "LHE", "LHE", 2025, 3).code, "same_airport")
        self.assertEqual(check_selection(df, "LHE", "ISB", 2025, 3).code, "unknown_route")
        nd = check_selection(df, "LHE", "KHI", 2025, 7)
        self.assertEqual(nd.code, "no_data")
        self.assertEqual(nd.months_with_data, ((2025, 3),))
        self.assertEqual(check_selection(df, "LHE", "KHI", 2025, 3).code, "ok")


class StatisticsTests(unittest.TestCase):
    """Expected values below were worked out by hand (see tests/helpers.manual_dataset)."""

    @classmethod
    def setUpClass(cls):
        cls.res = run(manual_dataset())
        cls.w = cls.res.windows.set_index("window")

    def test_status_and_counts(self):
        self.assertEqual(self.res.status, STATUS_OK)
        self.assertEqual(self.res.n_itineraries, 5)
        self.assertEqual(self.res.n_rows_selected, 15)

    def test_median_mean_std_iqr(self):
        w0 = self.w.loc["0–3 days"]
        self.assertEqual((w0["median"], w0["mean"]), (540, 540))
        self.assertAlmostEqual(w0["std"], 31.6228, places=3)  # sqrt(4000/4)
        self.assertEqual((w0["p25"], w0["p75"]), (520, 560))
        self.assertEqual((self.w.loc["8–14 days", "median"], self.w.loc["8–14 days", "mean"]), (340, 340))
        self.assertEqual((self.w.loc["31–60 days", "median"], self.w.loc["31–60 days", "mean"]), (270, 270))

    def test_windows_without_data_have_no_numbers(self):
        for label in ("4–7 days", "15–30 days", "61–90 days"):
            self.assertEqual(self.w.loc[label, "n_itineraries"], 0)
            self.assertTrue(np.isnan(self.w.loc[label, "median"]))
            self.assertFalse(self.w.loc[label, "eligible"])

    def test_recommendation_and_differences(self):
        r = self.res.recommendation
        self.assertEqual(r.best_window, "31–60 days")
        self.assertEqual(r.best_median, 270)
        self.assertEqual(r.comparison_window, "0–3 days")
        self.assertEqual(r.comparison_median, 540)
        self.assertEqual(r.saving_abs, 270)
        self.assertAlmostEqual(r.saving_pct, 0.5)
        self.assertEqual((r.runner_up_window, r.runner_up_median), ("8–14 days", 340))
        self.assertAlmostEqual(self.w.loc["31–60 days", "vs_baseline_pct"], -0.5)
        self.assertAlmostEqual(self.w.loc["8–14 days", "vs_baseline_pct"], 340 / 540 - 1)
        self.assertEqual(format_pct(-0.5), "-50%")

    def test_evidence_rule(self):
        r = self.res.recommendation
        # n=5 < 100 -> no sample-size point; bootstrap CIs cannot overlap (max 290 < min 300) -> 1 point
        self.assertTrue(r.ci_separated)
        self.assertEqual(r.evidence_level, "Moderate")
        self.assertEqual(len(r.evidence_points), 1)

    def test_strong_evidence_needs_both_points(self):
        cfg = AnalysisConfig(min_samples_per_window=5, min_eligible_windows=3, strong_sample_size=5)
        res = run(manual_dataset(), cfg=cfg)
        self.assertEqual(res.recommendation.evidence_level, "Strong")

    def test_deterministic(self):
        again = run(manual_dataset())
        pd.testing.assert_frame_equal(self.res.windows, again.windows)

    def test_explanation_mentions_key_facts(self):
        text = " ".join(explain(self.res))
        for needle in ("31–60 days", "PKR 270", "5 itineraries", "0–3 days", "PKR 540", "not a guaranteed saving"):
            self.assertIn(needle, text)


class RepeatedSnapshotTests(unittest.TestCase):
    def test_repeated_snapshots_are_collapsed_not_double_counted(self):
        rows = manual_dataset() + [snapshot("C1", 1, 700)]  # 2nd snapshot of C1 in window 0-3
        res = run(rows)
        w0 = res.windows.set_index("window").loc["0–3 days"]
        self.assertEqual(w0["n_itineraries"], 5)      # still five itineraries
        self.assertEqual(w0["n_snapshots"], 6)        # six raw rows
        # C1 collapses to median(500, 700) = 600 -> [600, 520, 540, 560, 580] -> median 560
        self.assertEqual(w0["median"], 560)
        self.assertEqual(res.n_itineraries, 5)


class BaselineTests(unittest.TestCase):
    def test_no_saving_claimed_when_last_minute_is_cheapest(self):
        rows = []
        for i in range(5):
            c = f"C{i + 1}"
            rows += [snapshot(c, 2, 100 + i), snapshot(c, 10, 300 + i), snapshot(c, 40, 500 + i)]
        r = run(rows).recommendation
        self.assertEqual(r.best_window, "0–3 days")
        self.assertIsNone(r.saving_abs)
        self.assertIsNone(r.saving_pct)
        self.assertIn("no saving", " ".join(explain(run(rows))))


class InsufficientAndEmptyTests(unittest.TestCase):
    def test_insufficient_when_too_few_eligible_windows(self):
        # five itineraries, but each is only seen in two windows (0-3 and 8-14): 2 eligible < 3 required
        rows = []
        for i in range(5):
            c = f"C{i + 1}"
            rows += [snapshot(c, 2, 500 + 20 * i), snapshot(c, 10, 300 + 20 * i)]
        res = run(rows)
        self.assertEqual(int(res.windows["eligible"].sum()), 2)
        self.assertEqual(res.status, STATUS_INSUFFICIENT)
        self.assertIsNone(res.recommendation)
        self.assertTrue(res.message.startswith(INSUFFICIENT_MESSAGE))

    def test_insufficient_when_sample_below_threshold(self):
        res = run(manual_dataset(), cfg=AnalysisConfig(min_samples_per_window=6, min_eligible_windows=3))
        self.assertEqual(res.status, STATUS_INSUFFICIENT)
        self.assertEqual(int(res.windows["eligible"].sum()), 0)

    def test_empty_selection(self):
        df, _ = clean(manual_dataset())
        res = analyze_booking_window(df, "LHE", "KHI", 2025, 8, "PKR", SMALL_CFG)
        self.assertEqual(res.status, STATUS_EMPTY)
        self.assertTrue(res.windows.empty)

    def test_empty_dataset(self):
        df, _ = clean_fares(raw_rows([{}]).iloc[0:0], "PKR", SMALL_CFG)  # valid schema, zero rows
        self.assertTrue(df.empty)
        res = analyze_booking_window(df, "LHE", "KHI", 2025, 3, "PKR", SMALL_CFG)
        self.assertEqual(res.status, STATUS_EMPTY)
        self.assertEqual(check_selection(df, "LHE", "KHI", 2025, 3).code, "unknown_route")

    def test_observations_beyond_90_days_are_not_analysed(self):
        rows = manual_dataset() + [snapshot("C1", 120, 1)]  # absurdly cheap but outside all windows
        res = run(rows)
        self.assertEqual(res.n_rows_outside_windows, 1)
        self.assertEqual(res.recommendation.best_median, 270)


if __name__ == "__main__":
    unittest.main()
