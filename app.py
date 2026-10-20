"""Travel.pk - Best Time to Book Finder (Streamlit entry point).

Run locally:  streamlit run app.py
"""
import streamlit as st

from besttime import ui
from besttime.analysis import analyze_booking_window, check_selection
from besttime.config import DEFAULT_CONFIG
from besttime.formatting import month_name
from besttime.loader import DataLoadError, dataset_coverage, load_dataset

st.set_page_config(page_title="Best Time to Book Finder | Travel.pk", page_icon="✈️", layout="wide")


@st.cache_data(show_spinner="Loading fare data…")
def get_dataset():
    return load_dataset()


def _index(options, preferred, fallback=0):
    return options.index(preferred) if preferred in options else fallback


def main() -> None:
    ui.inject_css()
    ui.render_header()

    # --- data-loading / processing error state ---------------------------
    try:
        ds = get_dataset()
    except DataLoadError as exc:
        ui.render_error("The fare dataset could not be loaded.", str(exc))
        ui.render_footer()
        return
    except Exception as exc:  # unexpected - still fail gracefully
        ui.render_error("Something went wrong while preparing the data.", f"{type(exc).__name__}: {exc}")
        ui.render_footer()
        return

    ui.render_demo_banner(ds.meta)

    cov = dataset_coverage(ds.df)
    airports, years = cov["airports"], cov["years"]
    if not airports or not years:
        ui.render_error("The dataset contains no usable fare observations.",
                        "Check the data file and the exclusion counts in the methodology section below.")
        ui.render_methodology(ds)
        ui.render_footer()
        return

    # --- search form ------------------------------------------------------
    with st.form("search"):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1.4])
        origin = c1.selectbox("From", airports, index=_index(airports, "LHE"), format_func=ui.airport_label)
        destination = c2.selectbox("To", airports, index=_index(airports, "KHI", 1 if len(airports) > 1 else 0),
                                   format_func=ui.airport_label)
        month = c3.selectbox("Travel month", list(range(1, 13)), index=2, format_func=month_name)
        year = c4.selectbox("Year", years, index=0)
        submitted = st.form_submit_button("Find Best Time to Book", type="primary")

    if submitted:
        st.session_state["query"] = (origin, destination, int(year), int(month))

    query = st.session_state.get("query")
    if not query:
        ui.render_initial_state(ds)
    else:
        q_origin, q_dest, q_year, q_month = query
        check = check_selection(ds.df, q_origin, q_dest, q_year, q_month)
        if check.code != "ok":
            ui.render_selection_problem(check, ds)
        else:
            try:
                with st.spinner("Analysing fare observations…"):
                    result = analyze_booking_window(ds.df, q_origin, q_dest, q_year, q_month,
                                                    ds.meta.currency, DEFAULT_CONFIG)
            except Exception as exc:
                ui.render_error("The analysis could not be completed.", f"{type(exc).__name__}: {exc}")
            else:
                ui.render_result(result, ds)

    ui.render_methodology(ds)
    ui.render_footer()


main()
