"""Integration tests on the shipped DEMO dataset: labelling, independent recomputation,
edge-case cells, and table/chart consistency."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from besttime.analysis import STATUS_EMPTY, STATUS_INSUFFICIENT, STATUS_OK, analyze_booking_window, check_selection
from besttime.config import DEFAULT_CONFIG, DEFAULT_DATA_PATH
from besttime.loader import DataLoadError, load_dataset, metadata_path_for
from besttime.metadata import DEMO_LABEL, DatasetMeta
from besttime import views

PAKISTAN_ROUTES = {("LHE", "KHI"), ("KHI", "LHE"), ("LHE", "ISB"), ("ISB", "LHE"), ("KHI", "ISB"), ("ISB", "KHI")}


class SyntheticLabellingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset()

    def test_demo_dataset_is_flagged_synthetic(self):
        self.assertTrue(self.ds.meta.is_synthetic)
        self.assertEqual(self.ds.meta.data_type_label, DEMO_LABEL)
        self.assertIn(DEMO_LABEL, self.ds.meta.disclosure)
        self.assertIn("not real airline fares", self.ds.meta.disclosure)
        self.assertIn("SIMULATED", self.ds.meta.name.upper())

    def test_every_raw_row_is_marked_simulated_and_ids_are_sim(self):
        raw = pd.read_csv(DEFAULT_DATA_PATH, dtype=str)
        self.assertEqual(set(raw["source"]), {"SIMULATED"})
        self.assertTrue(raw["obs_id"].str.startswith("SIM-").all())
        self.assertTrue(raw["carrier"].str.startswith("Carrier ").all())  # generic, not real airlines

    def test_metadata_has_no_fake_source_url(self):
        self.assertIsNone(self.ds.meta.source_url)

    def test_real_dataset_meta_has_no_demo_banner(self):
        meta = DatasetMeta(name="x", is_synthetic=False, currency="PKR")
        self.assertEqual(meta.disclosure, "")
        self.assertNotEqual(meta.data_type_label, DEMO_LABEL)

    def test_only_represented_routes_exist(self):
        routes = set(zip(self.ds.df["origin"], self.ds.df["destination"]))
        self.assertEqual(routes, PAKISTAN_ROUTES)
        self.assertEqual(check_selection(self.ds.df, "LHE", "PEW", 2025, 3).code, "unknown_route")

    def test_nothing_excluded_from_clean_demo_data(self):
        self.assertEqual(self.ds.report.rows_in, self.ds.report.rows_out)


class IndependentRecomputationTests(unittest.TestCase):
    """Recompute window medians straight from the CSV with plain pandas and compare."""

    def test_window_medians_match_independent_calculation(self):
        ds = load_dataset()
        res = analyze_booking_window(ds.df, "LHE", "KHI", 2025, 3, "PKR")
        self.assertEqual(res.status, STATUS_OK)

        raw = pd.read_csv(DEFAULT_DATA_PATH)
        raw["dep"] = pd.to_datetime(raw["departure_date"])
        raw["obs"] = pd.to_datetime(raw["observation_date"])
        raw = raw[(raw.origin == "LHE") & (raw.destination == "KHI") & (raw.dep.dt.year == 2025) & (raw.dep.dt.month == 3)]
        raw["lead"] = (raw.dep - raw.obs).dt.days

        def bucket(d):
            for lo, hi in DEFAULT_CONFIG.windows:
                if lo <= d <= hi:
                    return f"{lo}–{hi} days"

        raw["win"] = raw["lead"].map(bucket)
        per_itin = raw.groupby(["carrier", "departure_date", "win"])["fare"].median().reset_index()
        expected = per_itin.groupby("win")["fare"].agg(["median", "mean", "size"])

        got = res.windows.set_index("window")
        for win, row in expected.iterrows():
            self.assertAlmostEqual(got.loc[win, "median"], row["median"], places=6, msg=win)
            self.assertAlmostEqual(got.loc[win, "mean"], row["mean"], places=6, msg=win)
            self.assertEqual(got.loc[win, "n_itineraries"], row["size"], msg=win)

        best = min(expected["median"].items(), key=lambda kv: kv[1])
        self.assertEqual(res.recommendation.best_window, best[0])


class EdgeCellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset()

    def test_all_routes_in_a_full_month_give_a_recommendation(self):
        for o, d in PAKISTAN_ROUTES:
            res = analyze_booking_window(self.ds.df, o, d, 2025, 6, "PKR")
            with self.subTest(route=f"{o}-{d}"):
                self.assertEqual(res.status, STATUS_OK)
                self.assertGreaterEqual(res.recommendation.best_n, DEFAULT_CONFIG.min_samples_per_window)

    def test_thin_cell_reports_insufficient_data(self):
        res = analyze_booking_window(self.ds.df, "ISB", "KHI", 2025, 2, "PKR")
        self.assertEqual(res.status, STATUS_INSUFFICIENT)
        self.assertIsNone(res.recommendation)
        self.assertIn("Not enough data for a reliable recommendation", res.message)

    def test_missing_cell_is_empty(self):
        self.assertEqual(check_selection(self.ds.df, "KHI", "ISB", 2025, 1).code, "no_data")
        res = analyze_booking_window(self.ds.df, "KHI", "ISB", 2025, 1, "PKR")
        self.assertEqual(res.status, STATUS_EMPTY)

    def test_changing_selection_changes_results(self):
        a = analyze_booking_window(self.ds.df, "LHE", "KHI", 2025, 3, "PKR")
        b = analyze_booking_window(self.ds.df, "LHE", "ISB", 2025, 9, "PKR")
        self.assertNotEqual(a.recommendation.best_median, b.recommendation.best_median)
        self.assertEqual(b.route_label, "LHE → ISB")


class TableChartConsistencyTests(unittest.TestCase):
    def test_displayed_table_and_chart_frame_come_from_the_same_numbers(self):
        ds = load_dataset()
        res = analyze_booking_window(ds.df, "KHI", "LHE", 2025, 10, "PKR")
        frame = views.chart_frame(res)
        table = views.display_table(res)

        self.assertEqual(len(frame), len(table))
        self.assertEqual(int(frame["is_best"].sum()), 1)
        self.assertEqual(frame.loc[frame["is_best"], "window"].iloc[0], res.recommendation.best_window)
        self.assertTrue((frame["lead_min"].diff().dropna() < 0).all())   # far -> near order
        self.assertFalse(frame[["median", "mean", "n_itineraries"]].isna().any().any())  # no blanks/zeros plotted
        self.assertTrue((frame["n_itineraries"] > 0).all())

        by_window = res.windows.set_index("window")
        for _, trow in table.sort_values("Itineraries (n)").iterrows():
            win = trow["Booking window"].replace("  ★ recommended", "")
            self.assertEqual(trow["Itineraries (n)"], by_window.loc[win, "n_itineraries"])
            self.assertEqual(trow["Median (PKR)"], f"{by_window.loc[win, 'median']:,.0f}")
        for _, frow in frame.iterrows():
            self.assertEqual(frow["median"], by_window.loc[frow["window"], "median"])
            self.assertEqual(frow["n_itineraries"], by_window.loc[frow["window"], "n_itineraries"])

    def test_chart_frame_for_empty_result_is_empty(self):
        ds = load_dataset()
        res = analyze_booking_window(ds.df, "KHI", "ISB", 2025, 1, "PKR")
        self.assertTrue(views.chart_frame(res).empty)
        self.assertTrue(views.display_table(res).empty)


class LoaderErrorTests(unittest.TestCase):
    def _write(self, tmp: Path, csv_text: str, meta: dict | None):
        csv = tmp / "d.csv"
        csv.write_text(csv_text, encoding="utf-8")
        if meta is not None:
            metadata_path_for(csv).write_text(json.dumps(meta), encoding="utf-8")
        return csv

    def test_missing_metadata_refuses_to_load(self):
        with tempfile.TemporaryDirectory() as t:
            csv = self._write(Path(t), "a,b\n1,2\n", None)
            with self.assertRaises(DataLoadError):
                load_dataset(csv)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as t:
            meta = metadata_path_for(Path(t) / "nope.csv")
            meta.write_text(json.dumps({"name": "n", "is_synthetic": False, "currency": "PKR"}))
            with self.assertRaises(DataLoadError) as ctx:
                load_dataset(Path(t) / "nope.csv")
            self.assertNotIn(t, str(ctx.exception))  # no local paths leak into messages

    def test_missing_columns(self):
        with tempfile.TemporaryDirectory() as t:
            csv = self._write(Path(t), "origin,destination\nLHE,KHI\n",
                              {"name": "n", "is_synthetic": False, "currency": "PKR"})
            with self.assertRaises(DataLoadError) as ctx:
                load_dataset(csv)
            self.assertIn("missing required column", str(ctx.exception))

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as t:
            csv = self._write(Path(t), "", {"name": "n", "is_synthetic": False, "currency": "PKR"})
            with self.assertRaises(DataLoadError):
                load_dataset(csv)

    def test_real_dataset_drop_in_works(self):
        header = "obs_id,origin,destination,observation_date,departure_date,fare,currency\n"
        with tempfile.TemporaryDirectory() as t:
            csv = self._write(Path(t), header + "r1,LHE,KHI,2025-03-01,2025-03-15,20000,PKR\n",
                              {"name": "Some real source", "is_synthetic": False, "currency": "PKR"})
            ds = load_dataset(csv)
            self.assertFalse(ds.meta.is_synthetic)
            self.assertEqual(len(ds.df), 1)
            self.assertEqual(int(ds.df.loc[0, "lead_days"]), 14)


if __name__ == "__main__":
    unittest.main()
