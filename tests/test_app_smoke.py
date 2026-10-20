"""Executes app.py end-to-end against a FAKE Streamlit and FAKE Plotly.

What this proves: app.py, besttime/ui.py and besttime/charts.py run without
errors for every UI state, calling the real loader and analysis code, and that
the expected messages are emitted. What it does NOT prove: that the real
Streamlit/Plotly render correctly in a browser - see tests/test_charts_plotly.py
(runs only if plotly is installed) and the manual checklist in the README.
"""
import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


class _Ctx:
    """Context manager / column that forwards widget calls to the fake module."""

    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._st, name)


class FakeStreamlit(types.ModuleType):
    def __init__(self, answers=None, submit=True):
        super().__init__("streamlit")
        self.answers, self.submit = answers or {}, submit
        self.session_state = {}
        self.calls = []

    def cache_data(self, *a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

    def columns(self, spec, **k):
        return [_Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, labels):
        return [_Ctx(self) for _ in labels]

    def form(self, *a, **k):
        return _Ctx(self)

    def spinner(self, *a, **k):
        return _Ctx(self)

    def expander(self, *a, **k):
        return _Ctx(self)

    def selectbox(self, label, options, index=0, format_func=None, **k):
        options = list(options)
        if format_func:
            [format_func(o) for o in options]
        self.calls.append(("selectbox", (label,), {}))
        return self.answers.get(label, options[index])

    def form_submit_button(self, label, **k):
        self.calls.append(("form_submit_button", (label,), k))
        return self.submit

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def record(*a, **k):
            self.calls.append((name, a, k))

        return record

    # helpers for assertions
    def text(self, *names):
        return "\n".join(str(a[0]) for n, a, k in self.calls if n in names and a)

    def count(self, name):
        return sum(1 for n, *_ in self.calls if n == name)


def run_app(answers=None, submit=True, env=None):
    st = FakeStreamlit(answers, submit)
    go = mock.MagicMock(name="plotly.graph_objects")
    plotly = types.ModuleType("plotly")
    plotly.graph_objects = go
    stubs = {"streamlit": st, "plotly": plotly, "plotly.graph_objects": go}
    with mock.patch.dict(sys.modules, stubs), mock.patch.dict(os.environ, env or {}):
        import besttime
        for mod in ("ui", "charts"):
            # `from besttime import ui` would otherwise reuse a module bound to a previous fake.
            sys.modules.pop(f"besttime.{mod}", None)
            besttime.__dict__.pop(mod, None)
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    for mod in ("ui", "charts"):
        sys.modules.pop(f"besttime.{mod}", None)
        besttime.__dict__.pop(mod, None)
    return st


class AppSmokeTests(unittest.TestCase):
    def test_initial_state_before_search(self):
        st = run_app(submit=False)
        self.assertIn("Choose a route and travel month", st.text("markdown"))
        self.assertEqual(st.count("plotly_chart"), 0)
        self.assertIn("DEMO / SIMULATED DATA", st.text("markdown"))  # disclosure always visible

    def test_search_button_exists_with_required_label(self):
        st = run_app(submit=False)
        labels = [a[0] for n, a, k in st.calls if n == "form_submit_button"]
        self.assertEqual(labels, ["Find Best Time to Book"])

    def test_successful_result(self):
        st = run_app()  # defaults: LHE -> KHI, March, 2025
        md = st.text("markdown")
        self.assertIn("LHE → KHI", md)
        self.assertIn("Recommended booking window", md)
        self.assertIn("DEMO / SIMULATED DATA", md)
        self.assertEqual(st.count("plotly_chart"), 3)  # line, bar, sample-count
        self.assertEqual(st.count("error"), 0)
        self.assertIn("Data sources and methodology", st.text("header"))

    def test_invalid_route_same_airport(self):
        st = run_app({"From": "LHE", "To": "LHE"})
        self.assertIn("Invalid route", st.text("warning"))
        self.assertEqual(st.count("plotly_chart"), 0)

    def test_route_not_in_dataset(self):
        st = run_app({"From": "KHI", "To": "KHI"})  # same-airport guard
        self.assertIn("different airports", st.text("warning"))

    def test_insufficient_data(self):
        st = run_app({"From": "ISB", "To": "KHI", "Travel month": 2})
        self.assertIn("Not enough data for a reliable recommendation", st.text("warning"))
        self.assertNotIn("Recommended booking window", st.text("markdown"))

    def test_empty_filtered_dataset(self):
        st = run_app({"From": "KHI", "To": "ISB", "Travel month": 1})
        self.assertIn("No data for this month", st.text("info"))

    def test_data_loading_error_is_shown_not_raised(self):
        st = run_app(env={"TRAVELPK_FARE_DATA": str(ROOT / "does-not-exist.csv")})
        self.assertIn("could not be loaded", st.text("error"))
        self.assertEqual(st.count("form_submit_button"), 0)


if __name__ == "__main__":
    unittest.main()
