# Task-compliance checklist

Status key: ✅ implemented and covered by an automated test · 🟡 implemented, *not* verified in a real browser ·
⛔ not done (needs you)

| # | Requirement (brief §) | Where | Status |
|---|---|---|---|
| 1 | Answers "how far ahead should I book for this route/month" (§1) | `analysis.analyze_booking_window`, `app.py` | ✅ logic · 🟡 UI |
| 2 | Journey: origin, destination, month/year, **Find Best Time to Book**, results, change & re-run (§2) | `app.py` (form), `tests/test_app_smoke.py` (fake Streamlit) | 🟡 |
| 3 | Data source investigated; no fabricated "real" data; synthetic clearly labelled DEMO / SIMULATED (§3) | README "Data source investigation", `data/*.metadata.json`, `ui.render_demo_banner` | ✅ labelling · see note A |
| 4 | Data Sources & Methodology section with all required items (§3) | `ui.render_methodology`, metadata JSON | 🟡 |
| 5 | Replaceable dataset (§3) | `loader.py`, README "Replacing the demo data", `test_real_dataset_drop_in_works` | ✅ |
| 6 | Data model, normalisation, duplicates, invalid fares, currency handling (§4) | `cleaning.py`, `tests/test_cleaning.py` | ✅ |
| 7 | Lead time = departure − observation (§4) | `cleaning.clean_fares` | ✅ |
| 8 | Six booking windows, configurable (§5) | `config.AnalysisConfig`, `analysis.assign_window` | ✅ |
| 9 | n / median / mean / variability per window (§5.4) | `analysis.compute_window_stats` | ✅ (hand-checked values) |
| 10 | Explicit, configurable minimum sample size; insufficient-data message (§5.6–7) | `AnalysisConfig`, `INSUFFICIENT_MESSAGE` | ✅ |
| 11 | Differences only vs an identified comparison window (§5.9) | `analysis` (baseline = closest eligible window) | ✅ |
| 12 | Repeated snapshots not treated as independent; no look-ahead (§5) | itinerary-in-window collapse; limitations text | ✅ (collapse) · see note B |
| 13 | Pakistan routes; only routes in the data (§6) | `test_only_represented_routes_exist` | ✅ (simulated routes) |
| 14 | Results screen A–D: recommendation, comparison, 3 charts, transparency (§7) | `ui.render_result`, `charts.py`, `views.py` | 🟡 (charts' real Plotly output unverified here) |
| 15 | Charts use the same filtering as the table (§7) | `views.chart_frame` shared; `TableChartConsistencyTests`, `test_charts_plotly.py` | ✅ data · Plotly test runs once plotly is installed |
| 16 | No meaningless zeros / fake precision / guaranteed savings (§7) | empty windows omitted; whole-unit rounding; wording in `explain` | ✅ |
| 17 | Branded, responsive UI; all interface states (§8) | `ui.py` CSS, `app.py` | 🟡 |
| 18 | Streamlit + pandas + Plotly, modular code, requirements, no paid credentials (§9) | whole repo, `requirements.txt` | ✅ |
| 19 | Graceful failure on data errors (§9) | `loader.DataLoadError`, `test_data_loading_error_is_shown_not_raised` | ✅ |
| 20 | Automated tests for the listed areas (§10) | `tests/` | ✅ 63 passed, 3 skipped |
| 21 | README incl. Windows setup, tests, GitHub, deployment (§12) | `README.md` | ✅ |
| 22 | Deployment-ready package (§11) | `requirements.txt`, `.gitignore`, `.streamlit/config.toml`, README | ✅ |
| 23 | **Public live URL deployed and verified (§11, §14.11)** | — | ⛔ not done |

**Note A:** the data is simulated, so the assignment's "analyze fare data" is demonstrated on synthetic observations only.
**Note B:** look-ahead is avoided by construction (each observation is analysed only as the price seen on its own date),
but there is no separate test for it.
