"""Checks the real Plotly figures against the chart frame. Runs only where plotly
is installed (e.g. after `pip install -r requirements.txt`); otherwise skipped."""
import unittest

try:
    import plotly.graph_objects as go  # noqa: F401
    HAVE_PLOTLY = True
except ImportError:  # pragma: no cover
    HAVE_PLOTLY = False

from besttime import views
from besttime.analysis import analyze_booking_window
from besttime.loader import load_dataset


@unittest.skipUnless(HAVE_PLOTLY, "plotly not installed")
class PlotlyFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from besttime import charts
        cls.charts = charts
        ds = load_dataset()
        cls.res = analyze_booking_window(ds.df, "LHE", "KHI", 2025, 3, "PKR")
        cls.frame = views.chart_frame(cls.res)

    def test_bar_chart_plots_exactly_the_table_medians(self):
        fig = self.charts.bar_chart(self.frame, "PKR")
        self.assertEqual(list(fig.data[0].x), list(self.frame["window"]))
        self.assertEqual([float(v) for v in fig.data[0].y], [float(v) for v in self.frame["median"]])
        self.assertIn("PKR", fig.layout.yaxis.title.text)

    def test_line_chart_median_trace_matches(self):
        fig = self.charts.line_chart(self.frame, "PKR")
        median = [t for t in fig.data if t.name == "Median fare"][0]
        ok = self.frame[self.frame["eligible"]]
        self.assertEqual([float(v) for v in median.y], [float(v) for v in ok["median"]])

    def test_sample_chart_counts_match_and_axes_labelled(self):
        fig = self.charts.sample_chart(self.frame, 30)
        self.assertEqual([int(v) for v in fig.data[0].y], [int(v) for v in self.frame["n_itineraries"]])
        self.assertTrue(fig.layout.xaxis.title.text)
        self.assertTrue(fig.layout.yaxis.title.text)


if __name__ == "__main__":
    unittest.main()
