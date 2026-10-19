import io
import math
import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="🌎", layout="wide")
st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB • Historical data, mathematical models, and interactive function analysis")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if mortality patterns at birth remained constant."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Unemployment as a percentage of the labor force, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population, in current US dollars; not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual number of live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people (net)", "Net number of migrants during the period; this is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods/services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP."),
}
CONTEXTS = {
    "Business": ("Revenue", "currency units", "Time", "months", "a business revenue trend", "a*exp(b*x)"),
    "Engineering": ("Displacement", "meters", "Time", "seconds", "motion of an engineered object", "a*x**2+b*x+c"),
    "Medical": ("Concentration", "mg/L", "Time", "hours", "a changing concentration", "a*exp(-b*x)"),
    "Sports science": ("Performance", "score units", "Training time", "weeks", "athletic performance over training", "a*sin(b*x)+c"),
    "Biology": ("Population", "organisms", "Time", "days", "population growth", "a*exp(b*x)"),
    "Bioengineering": ("Cell density", "cells/mL", "Time", "hours", "cell culture growth", "a/(1+b*exp(-c*x))"),
    "Economic forecasting": ("Indicator", "index units", "Year", "years", "an economic indicator", "a*x**2+b*x+c"),
    "Weather": ("Temperature", "°C", "Time", "hours", "temperature variation", "a*sin(b*x)+c"),
    "Custom": ("Output", "units", "Input", "units", "a custom relationship", "a*x**2+b*x+c"),
}
BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code: str, indicator_code: str, start_year: int, end_year: int):
    url = BASE_URL.format(country=country_code, indicator=indicator_code)
    params = {"format": "json", "date": f"{start_year}:{end_year}", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), f"Unexpected or empty API response from {url}"
        rows = [{"year": int(r["date"]), "value": r.get("value")} for r in payload[1]
                if r.get("date") and str(r["date"]).isdigit()]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["year", "value"]), f"No observations returned by the World Bank API ({url})"
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
        return df, None if not df.empty else f"No non-null observations were returned ({url})"
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not retrieve data: {type(exc).__name__}: {exc}"

def valid_value(value, indicator_name):
    if not np.isfinite(value):
        return False
    # Indicator-specific constraints are conservative; net migration and external balance can be negative.
    if indicator_name in ("Life expectancy at birth", "Crude birth rate", "Intentional homicide rate",
                          "Total reserves including gold"):
        return value >= 0
    if indicator_name == "Unemployment, national estimate" and not (0 <= value <= 100):
        return False
    if indicator_name == "Central government debt" and value < 0:
        return False
    return True

def clean_edited_data(frame, indicator_name):
    errors = []
    out = frame.copy()
    if not {"year", "value"}.issubset(out.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    out = out[["year", "value"]].copy()
    numeric_year = pd.to_numeric(out["year"], errors="coerce")
    numeric_value = pd.to_numeric(out["value"], errors="coerce")
    bad_year = numeric_year.isna() | (numeric_year % 1 != 0)
    if bad_year.any():
        errors.append(f"{int(bad_year.sum())} row(s) have a missing or non-integer year.")
    out["year"] = numeric_year
    out["value"] = numeric_value
    out = out.dropna(subset=["year", "value"])
    out["year"] = out["year"].astype(int)
    duplicates = out["year"].duplicated(keep=False)
    if duplicates.any():
        errors.append("Duplicate years are not allowed. Keep one row per year.")
        out = out.loc[~duplicates]
    bad = ~out["value"].apply(lambda v: valid_value(float(v), indicator_name))
    if bad.any():
        errors.append(f"{int(bad.sum())} row(s) violate indicator-specific numeric constraints and were excluded.")
        out = out.loc[~bad]
    out = out.sort_values("year").reset_index(drop=True)
    return out, errors

def fit_polynomial(df, degree):
    if len(df) < degree + 1 or df["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid years; current count is {df['year'].nunique()}.")
    base = int(df["year"].min())
    t = df["year"].to_numpy(dtype=float) - base
    y = df["value"].to_numpy(dtype=float)
    coeff = np.polynomial.polynomial.polyfit(t, y, degree)
    pred = np.polynomial.polynomial.polyval(t, coeff)
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    return {"base": base, "coeff": coeff, "degree": degree, "r2": r2, "mae": mae,
            "residuals": y - pred, "predicted": pred, "n": len(df),
            "min_year": int(df.year.min()), "max_year": int(df.year.max()),
            "mean": float(np.mean(y))}

def poly_predict(model, years):
    return np.polynomial.polynomial.polyval(np.asarray(years, dtype=float) - model["base"], model["coeff"])

def poly_equation(model, units):
    terms = []
    for power, c in enumerate(model["coeff"]):
        if abs(c) < 1e-12:
            continue
        if power == 0:
            terms.append(f"{c:.8g}")
        elif power == 1:
            terms.append(f"{c:+.8g}t")
        else:
            terms.append(f"{c:+.8g}t^{power}")
    return "y = " + (" ".join(terms) if terms else "0") + f", where t = year − {model['base']} ({units})"

def holdout_score(df, degree):
    # Chronological last 20%, at least one test point; train set must have degree+1 observations.
    n = len(df)
    test_n = max(1, int(math.ceil(n * 0.2)))
    if n - test_n < degree + 1 or df.year.nunique() < degree + 1:
        return None
    train, test = df.iloc[:-test_n], df.iloc[-test_n:]
    base = int(train.year.min())
    co = np.polynomial.polynomial.polyfit(train.year.to_numpy(float) - base, train.value.to_numpy(float), degree)
    pred = np.polynomial.polynomial.polyval(test.year.to_numpy(float) - base, co)
    return {"mae": float(mean_absolute_error(test.value, pred)), "r2": float(r2_score(test.value, pred)) if len(test)>1 else None,
            "train_n": len(train), "test_n": len(test), "first_test": int(test.year.min())}

def parse_function(text, x):
    safe = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
            "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
            "pi": sp.pi, "E": sp.E, "sinh": sp.sinh, "cosh": sp.cosh}
    # Accept familiar ^ notation as exponentiation; sympy parses only an explicit allowlist.
    cleaned = text.replace("^", "**")
    try:
        expr = sp.sympify(cleaned, locals=safe, evaluate=True)
        if not expr.free_symbols.issubset({x}):
            raise ValueError("Use x as the only variable.")
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError("Function contains an undefined or infinite constant.")
        return expr
    except Exception as exc:
        raise ValueError(f"Could not parse function. Use x, numbers, + − * /, powers, and sin/cos/tan/exp/log/sqrt. Details: {exc}")

def safe_numeric(expr, xvals, x):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        try:
            vals = np.asarray(fn(xvals))
            if vals.ndim == 0:
                vals = np.full(np.asarray(xvals).shape, vals.item())
            vals = np.broadcast_to(vals, np.asarray(xvals).shape).astype(float)
            vals[~np.isfinite(vals)] = np.nan
            return vals
        except Exception:
            return np.full(np.asarray(xvals).shape, np.nan)

def fmt(v, digits=6):
    if v is None or not np.isfinite(float(v)):
        return "not available"
    return f"{float(v):,.{digits}g}"

def derivative_analysis(expr, x, lo, hi):
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    results = {"d1": d1, "d2": d2, "critical": [], "inflections": [], "notes": []}
    # Numeric root candidates from solveset are used only when finite and explicit.
    for target, kind in [(d1, "critical"), (d2, "inflection candidate")]:
        try:
            roots = sp.solveset(target, x, domain=sp.Interval(lo, hi))
            if isinstance(roots, sp.FiniteSet):
                root_values = sorted([float(sp.N(r)) for r in roots if abs(float(sp.im(sp.N(r)))) < 1e-8])
            else:
                root_values = []
                results["notes"].append(f"{kind.title()} roots are not a finite explicit set on this interval; inspect graph and derivative.")
        except Exception:
            root_values = []
        for r in root_values:
            if not lo <= r <= hi:
                continue
            if kind == "critical":
                left = max(lo, r - max(1e-5, (hi-lo)*1e-4))
                right = min(hi, r + max(1e-5, (hi-lo)*1e-4))
                fl, fr = safe_numeric(d1, np.array([left, right]), x)
                f2 = safe_numeric(d2, np.array([r]), x)[0]
                if np.isfinite(fl) and np.isfinite(fr) and fl*fr < 0:
                    typ = "local minimum" if fl < 0 < fr else "local maximum"
                elif np.isfinite(f2) and f2 > 1e-7:
                    typ = "local minimum (second-derivative test)"
                elif np.isfinite(f2) and f2 < -1e-7:
                    typ = "local maximum (second-derivative test)"
                else:
                    typ = "stationary/critical point; not classified as an extremum"
                results["critical"].append((r, typ))
            else:
                left = max(lo, r - max(1e-5, (hi-lo)*1e-4))
                right = min(hi, r + max(1e-5, (hi-lo)*1e-4))
                vals = safe_numeric(d2, np.array([left, right]), x)
                if np.all(np.isfinite(vals)) and vals[0]*vals[1] < 0:
                    results["inflections"].append((r, "concavity changes"))
                else:
                    results["notes"].append(f"x ≈ {r:.6g}: second derivative is zero, but a concavity change was not confirmed.")
    # Sign chart by numerical probes, explicitly avoiding singular points.
    probes = np.linspace(lo, hi, 201)
    v1, v2 = safe_numeric(d1, probes, x), safe_numeric(d2, probes, x)
    def sign_summary(vals, name):
        finite = vals[np.isfinite(vals)]
        if not len(finite): return f"{name}: unavailable on this interval."
        return f"{name}: increasing/positive at {np.mean(finite > 1e-7)*100:.0f}% of finite probes; decreasing/negative at {np.mean(finite < -1e-7)*100:.0f}%."
    results["notes"].extend([sign_summary(v1, "First derivative"), sign_summary(v2, "Second derivative")])
    return results

def make_chart_png(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
    b.seek(0)
    return b

now_year = datetime.now().year
start_default, end_default = now_year - 70, now_year - 1
st.sidebar.header("Data & model controls")
country_label = st.sidebar.selectbox("Featured country", list(COUNTRIES), index=0)
indicator_name = st.sidebar.selectbox("World Bank indicator", list(INDICATORS), index=0)
degree = st.sidebar.slider("Polynomial degree (3–8)", 3, 8, 3)
sample_step = st.sidebar.slider("Graph sample increment (years)", 1, 10, 3)
horizon = st.sidebar.slider("Forward display horizon (years)", 0, 50, 10)
comparison_labels = [x for x in COUNTRIES if x != country_label]
comparison_selected = st.sidebar.multiselect("Comparison countries (same indicator)", comparison_labels, default=[])
printer_mode = st.sidebar.checkbox("Printer-friendly report view", value=False)
st.sidebar.caption(f"Requested World Bank year window: {start_default}–{end_default} (current year minus 70 through minus 1).")

country_code = COUNTRIES[country_label]
indicator_code, units, definition = INDICATORS[indicator_name]
raw_df, api_error = fetch_world_bank(country_code, indicator_code, start_default, end_default)
if api_error:
    st.error(api_error)
    st.info("The app does not invent substitute data. Check the URL/network and try again later.")
raw_df = raw_df[["year", "value"]].copy()
if "working_df" not in st.session_state or st.session_state.get("working_key") != (country_code, indicator_code):
    st.session_state.working_df = raw_df.copy()
    st.session_state.working_key = (country_code, indicator_code)
    st.session_state.edited_scenario = False
source_df = raw_df.copy()
st.subheader("1. Source data and validation")
st.markdown(f"**Indicator:** {indicator_name} (`{indicator_code}`)  \n**Definition:** {definition}  \n**Units:** {units}  \n**Source:** [World Bank API](https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&date={start_default}:{end_default})  \n**Retrieved:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  \n**Country note:** Brazil is included as a Latin American comparison country; it is not a Spanish-speaking country.")
if raw_df.empty:
    st.warning("No source observations are currently available for this selection. Polynomial fitting and prediction are disabled until valid data are available.")
else:
    actual_min, actual_max = int(raw_df.year.min()), int(raw_df.year.max())
    st.write(f"Actual non-null coverage: **{actual_min}–{actual_max}**, **{len(raw_df)} observations** (not necessarily 70).")
    st.dataframe(raw_df, use_container_width=True, hide_index=True)
st.markdown("Edit the working table below to explore a scenario. The original API data above stays separate.")
edited_input = st.data_editor(st.session_state.working_df, num_rows="dynamic", use_container_width=True, key="source_editor")
if st.button("Reset to source data"):
    st.session_state.working_df = source_df.copy()
    st.session_state.edited_scenario = False
    st.rerun()
validated_df, validation_errors = clean_edited_data(edited_input, indicator_name)
if validation_errors:
    for err in validation_errors: st.warning(err)
edited_changed = not validated_df.reset_index(drop=True).equals(source_df.reset_index(drop=True))
st.session_state.edited_scenario = bool(edited_changed)
if st.session_state.edited_scenario:
    st.warning("USER-MODIFIED SCENARIO: calculations and exports use the validated editable table, not untouched source observations.")
if len(validated_df):
    st.write(f"Validated working observations: **{len(validated_df)}**; year range **{validated_df.year.min()}–{validated_df.year.max()}**.")
else:
    st.warning("The working table has no valid rows.")

model = None
fit_error = None
if len(validated_df) >= degree + 1 and validated_df.year.nunique() >= degree + 1:
    try: model = fit_polynomial(validated_df, degree)
    except Exception as exc: fit_error = str(exc)
else:
    fit_error = f"Degree {degree} requires at least {degree+1} distinct valid years; there are {validated_df.year.nunique()}."
if fit_error: st.warning(f"Model not fitted: {fit_error}")

st.subheader("2. Fit and evaluate the historical trend")
if model:
    equation = poly_equation(model, units)
    a,b,c,d = st.columns(4)
    a.metric("Observations", model["n"])
    b.metric("In-sample R²", fmt(model["r2"], 5))
    c.metric("In-sample MAE", f"{fmt(model['mae'])} {units}")
    d.metric("Mean observed value", f"{fmt(model['mean'])} {units}")
    st.code(equation)
    st.caption(f"Centered time: t = year − {model['base']}. The fitted polynomial is a descriptive model, not proof of causation.")
    holdout = holdout_score(validated_df, degree)
    if holdout:
        st.write(f"**Chronological holdout (separate from in-sample metrics):** trained on {holdout['train_n']} earlier observations, tested on {holdout['test_n']} later observation(s) starting {holdout['first_test']}; MAE = {fmt(holdout['mae'])} {units}; holdout R² = {fmt(holdout['r2']) if holdout['r2'] is not None else 'not meaningful for one test point'}.")
    else:
        st.caption("Chronological holdout not shown: not enough training observations remain for this degree after reserving a later test segment.")
    residual_df = validated_df.copy()
    residual_df["fitted_value"] = model["predicted"]
    residual_df["residual"] = model["residuals"]
    with st.expander("Residuals and fitted values"):
        st.dataframe(residual_df, use_container_width=True, hide_index=True)
    st.markdown(f"**Uncertainty and limitations:** in-sample MAE is the average absolute residual ({fmt(model['mae'])} {units}); individual residuals show how far the fitted curve differs from observations. These are not validated 95% prediction intervals. Polynomial extrapolation can become implausible rapidly, especially at higher degrees; inspect values and context before interpreting them.")
else:
    equation, holdout = "Not fitted", None

st.subheader("3. Calculate single-year and batch estimates")
single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(end_default, 1950), 2100), step=1)
if st.button("Calculate Value", type="primary", disabled=model is None):
    st.session_state.single_result = {"year": int(single_year), "country": country_label, "indicator": indicator_name, "degree": degree,
        "data_hash": (tuple(validated_df.year), tuple(np.round(validated_df.value, 12))), "estimate": float(poly_predict(model, [single_year])[0])}
if model and "single_result" in st.session_state:
    sr = st.session_state.single_result
    current_hash = (tuple(validated_df.year), tuple(np.round(validated_df.value, 12)))
    if (sr["country"], sr["indicator"], sr["degree"], sr["data_hash"]) != (country_label, indicator_name, degree, current_hash):
        st.info("Previous single-year result invalidated because country, indicator, degree, or working data changed. Press Calculate Value again.")
        st.session_state.pop("single_result", None)
    else:
        in_range = model["min_year"] <= sr["year"] <= model["max_year"]
        st.success(f"Year {sr['year']}: estimated {fmt(sr['estimate'])} {units}. This is a fitted value at an observed year only if {sr['year']} exists in the working data; otherwise it is a model estimate. The year is {'within' if in_range else 'outside'} the fitted historical range ({model['min_year']}–{model['max_year']}).")
        if not in_range:
            st.warning("Extrapolation: this estimate is outside the observed fitting range and may be implausible; it is not a forecast guarantee.")

batch_mode = st.radio("Batch year input", ["Year range", "Custom years"], horizontal=True)
batch_years, batch_errors = [], []
if batch_mode == "Year range":
    q1, q2, q3 = st.columns(3)
    batch_start = q1.number_input("Start year", 1950, 2100, max(1950, model["max_year"] if model else 2000), key="batch_start")
    batch_end = q2.number_input("End year", 1950, 2100, min(2100, (model["max_year"] if model else 2000)+10), key="batch_end")
    batch_step = q3.slider("Range step (years)", 1, 10, 1)
    if batch_start > batch_end: batch_errors.append("Start year must be less than or equal to end year.")
    else: batch_years = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
else:
    custom_text = st.text_input("Custom years (comma-separated)", value="2000, 2010, 2020")
    for token in custom_text.split(","):
        token = token.strip()
        if not token: continue
        try:
            num = float(token)
            if not num.is_integer() or not 1950 <= num <= 2100:
                batch_errors.append(f"'{token}' must be an integer from 1950 to 2100.")
            else: batch_years.append(int(num))
        except ValueError: batch_errors.append(f"'{token}' is not a valid year.")
    if len(set(batch_years)) != len(batch_years): batch_errors.append("Duplicate custom years were removed.")
    batch_years = sorted(set(batch_years))
for err in batch_errors: st.warning(err)
batch_results = pd.DataFrame()
if model and batch_years and not batch_errors:
    estimates = poly_predict(model, batch_years)
    batch_results = pd.DataFrame({"year": batch_years, "estimate": estimates, "units": units,
        "classification": ["within fitted range" if model["min_year"] <= yr <= model["max_year"] else "outside fitted range (extrapolation)" for yr in batch_years]})
    st.dataframe(batch_results, use_container_width=True, hide_index=True)
    st.caption("Batch estimates use the same centered-time fitted transformation as the single-year calculator.")
elif not batch_errors:
    st.info("Enter valid years and fit a model to see batch estimates.")

st.subheader("4. Compare countries")
comparison_data = {}
for label in comparison_selected:
    code = COUNTRIES[label]
    cdf, err = fetch_world_bank(code, indicator_code, start_default, end_default)
    if err:
        st.warning(f"{label}: {err}")
    else:
        comparison_data[label] = cdf
if comparison_data:
    comp_rows = []
    for label, cdf in comparison_data.items():
        comp_rows.append({"Country": label, "Coverage": f"{int(cdf.year.min())}–{int(cdf.year.max())}" if len(cdf) else "none",
                          "Observations": len(cdf), "Latest available year": int(cdf.year.max()) if len(cdf) else None})
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
    st.caption("Country coverage can differ because World Bank series may begin later or contain missing observations. Comparisons show the same indicator and units; they do not fill gaps.")

st.subheader("5. Context and function lab")
course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], index=2)
context = st.selectbox("Context", list(CONTEXTS))
suggest_y, suggest_y_units, suggest_x, suggest_x_units, default_description, suggested_fn = CONTEXTS[context]
use_fit = st.checkbox("Use generated function from the World Bank curve of best fit", value=bool(model))
function_source = st.radio("Function source", ["Suggested function", "Change parameters", "Enter my own function"], horizontal=True)
description = st.text_input("Context description", default_description)
y_name = st.text_input("Dependent variable name", suggest_y)
y_units = st.text_input("Dependent variable units", suggest_y_units)
x_name = st.text_input("Independent variable name", suggest_x)
x_units = st.text_input("Independent variable units", suggest_x_units)
p1,p2,p3 = st.columns(3)
pa = p1.number_input("Parameter a", value=2.0, step=0.5)
pb = p2.number_input("Parameter b", value=0.1, step=0.05)
pc = p3.number_input("Parameter c", value=5.0, step=0.5)
if use_fit and model:
    x = sp.Symbol("x", real=True)
    # model's centered variable t maps to x as a year value
    custom_expr = sum(sp.Float(float(co)) * (x - model["base"])**i for i, co in enumerate(model["coeff"]))
    active_fn_text = equation
elif function_source == "Suggested function":
    x = sp.Symbol("x", real=True)
    active_fn_text = suggested_fn
    try: custom_expr = parse_function(active_fn_text, x)
    except ValueError: custom_expr = None
elif function_source == "Change parameters":
    x = sp.Symbol("x", real=True)
    active_fn_text = suggested_fn.replace("a", f"({pa})").replace("b", f"({pb})").replace("c", f"({pc})")
    try: custom_expr = parse_function(active_fn_text, x)
    except ValueError: custom_expr = None
else:
    active_fn_text = st.text_input("Function f(x) (examples: x^2-3*x+2, 2*exp(-0.4*x), sin(x), 1/(x-2))", value="x^2-3*x+2")
    x = sp.Symbol("x", real=True)
    try: custom_expr = parse_function(active_fn_text, x)
    except ValueError as exc:
        custom_expr = None
        st.error(str(exc))
if function_source != "Enter my own function":
    st.code(active_fn_text)
else:
    st.caption("Use x as the input variable. Supported: polynomial, rational, exponential, logarithmic, and trigonometric expressions. Use * explicitly for multiplication and ^ or ** for powers.")
interval_default_lo = float(model["min_year"] if model else 0)
interval_default_hi = float(model["max_year"] if model else 10)
if not use_fit:
    interval_default_lo, interval_default_hi = 0.0, 10.0
interval_col1, interval_col2 = st.columns(2)
x_lo = interval_col1.number_input("Analysis interval start", value=interval_default_lo, key="analysis_lo")
x_hi = interval_col2.number_input("Analysis interval end", value=max(interval_default_lo+1, interval_default_hi), key="analysis_hi")
if x_lo >= x_hi:
    st.error("Analysis interval start must be less than the end.")
    analysis_ok = False
else: analysis_ok = True
if custom_expr is not None and analysis_ok:
    d1_expr, d2_expr = sp.diff(custom_expr, x), sp.diff(custom_expr, x, 2)
    st.markdown(f"**Function:** `f(x) = {sp.sstr(custom_expr)}`")
    st.markdown(f"**First derivative:** `f′(x) = {sp.sstr(d1_expr)}`")
    st.markdown(f"**Second derivative:** `f″(x) = {sp.sstr(d2_expr)}`")
    if course != "Calculus":
        st.info("Algebra 2 focus: domain/range, graphical trends, end behavior, and average rates of change. Precalculus adds degree/coefficient effects, turning points, and concavity through changing average rates. Calculus mode includes derivative-based analysis below.")
    # Animation-like synchronized input: Streamlit reruns on slider changes. Play uses a short, bounded stepping sequence via rerun-safe session state.
    x_min_slider = float(x_lo); x_max_slider = float(x_hi)
    if "sim_x" not in st.session_state or not x_min_slider <= st.session_state.sim_x <= x_max_slider:
        st.session_state.sim_x = x_min_slider
    sx1, sx2, sx3 = st.columns([2,1,1])
    sim_x = sx1.slider(f"Current {x_name} ({x_units})", min_value=float(x_min_slider), max_value=float(x_max_slider), value=float(st.session_state.sim_x), key="sim_x_slider")
    st.session_state.sim_x = sim_x
    if sx2.button("▶ Play one step"):
        st.session_state.sim_x = min(x_max_slider, sim_x + max((x_max_slider-x_min_slider)/20, 0.01))
        st.rerun()
    if sx3.button("↺ Reset point"):
        st.session_state.sim_x = x_min_slider
        st.rerun()
    step_x = max((x_max_slider-x_min_slider)/20, 0.01)
    av1, av2 = st.columns(2)
    sec_a = av1.slider("Secant interval start", float(x_min_slider), float(x_max_slider), float(x_min_slider), key="sec_a")
    sec_b = av2.slider("Secant interval end", float(x_min_slider), float(x_max_slider), float(x_max_slider), key="sec_b")
    y0 = safe_numeric(custom_expr, np.array([sim_x]), x)[0]
    dydx = safe_numeric(d1_expr, np.array([sim_x]), x)[0]
    ya, yb = safe_numeric(custom_expr, np.array([sec_a, sec_b]), x)
    avg_rate = (yb-ya)/(sec_b-sec_a) if sec_b != sec_a and np.isfinite(ya) and np.isfinite(yb) else np.nan
    st.write(f"At {x_name} = **{sim_x:.5g} {x_units}**, {y_name} ≈ **{fmt(y0)} {y_units}**; instantaneous rate f′(x) ≈ **{fmt(dydx)} {y_units}/{x_units}**.")
    st.write(f"Average rate from {sec_a:.5g} to {sec_b:.5g}: **{fmt(avg_rate)} {y_units}/{x_units}**.")
    if context in ("Engineering",):
        st.caption(f"Context interpretation: function value is displacement; first derivative is velocity ({y_units}/{x_units}); second derivative is acceleration ({y_units}/{x_units}²).")
    elif context in ("Business", "Economic forecasting"):
        st.caption(f"Context interpretation: function value is {y_name}; first derivative is its rate of growth/decline ({y_units}/{x_units}); second derivative describes how that rate itself changes ({y_units}/{x_units}²).")
    else:
        st.caption(f"Context interpretation: function value measures {y_name}; the first derivative is its rate of change ({y_units}/{x_units}); the second derivative describes how that rate is changing ({y_units}/{x_units}²). Units depend on the variables and the assumptions of the selected context.")
    show_fn = st.checkbox("Show function curve", True)
    show_d1 = st.checkbox("Show first derivative", course == "Calculus")
    show_d2 = st.checkbox("Show second derivative", False)
    xx = np.linspace(x_min_slider, x_max_slider, 800)
    yy = safe_numeric(custom_expr, xx, x)
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_fn: ax.plot(xx, yy, label=f"f(x) [{y_units}]")
    if show_d1: ax.plot(xx, safe_numeric(d1_expr, xx, x), label="f′(x)")
    if show_d2: ax.plot(xx, safe_numeric(d2_expr, xx, x), label="f″(x)")
    if np.isfinite(y0):
        ax.scatter([sim_x], [y0], label="Current point", zorder=5)
        if np.isfinite(dydx):
            tangent_x = np.linspace(max(x_min_slider, sim_x-(x_max_slider-x_min_slider)/8), min(x_max_slider, sim_x+(x_max_slider-x_min_slider)/8), 50)
            ax.plot(tangent_x, y0+dydx*(tangent_x-sim_x), linestyle="--", label="Tangent line")
    if sec_a != sec_b and np.isfinite(avg_rate) and np.isfinite(ya):
        sec_x = np.array([sec_a, sec_b]); sec_y = np.array([ya, yb])
        ax.plot(sec_x, sec_y, marker="o", linestyle=":", label="Secant line")
    ax.axvline(model["max_year"] if use_fit and model else x_max_slider, linestyle="--", alpha=.5, label="Historical boundary" if use_fit and model else "Interval boundary")
    ax.set_xlabel(f"{x_name} ({x_units})"); ax.set_ylabel(f"{y_name} / derivative units")
    ax.set_title(f"{description}: function and rates of change")
    ax.grid(True, alpha=.25); ax.legend()
    st.pyplot(fig, use_container_width=True)
    if use_fit and model:
        sample_years = np.arange(model["min_year"], model["max_year"]+1, sample_step)
        st.caption(f"Graph sample increment = {sample_step} year(s). Smooth curve is plotted separately; the sample table below uses this increment.")
        st.dataframe(pd.DataFrame({"year": sample_years, "model estimate": poly_predict(model, sample_years)}), use_container_width=True, hide_index=True)
    if course == "Calculus":
        st.markdown("#### Calculus analysis")
        analysis = derivative_analysis(custom_expr, x, float(x_lo), float(x_hi))
        st.write(f"**Critical points from f′(x) = 0:** {', '.join([f'x ≈ {r:.6g}: {typ}' for r,typ in analysis['critical']]) or 'No finite explicit roots identified.'}")
        st.write(f"**Inflection points:** {', '.join([f'x ≈ {r:.6g} ({typ})' for r,typ in analysis['inflections']]) or 'No confirmed inflection points identified.'}")
        for note in analysis["notes"]: st.caption(note)
        # Endpoint-aware extrema from a finite numerical grid; clearly labeled approximations.
        grid = np.linspace(float(x_lo), float(x_hi), 4001)
        vals = safe_numeric(custom_expr, grid, x)
        finite = np.isfinite(vals)
        if finite.any():
            indices = np.where(finite)[0]
            imin, imax = indices[np.argmin(vals[finite])], indices[np.argmax(vals[finite])]
            st.write(f"**Numerical absolute extrema on closed interval (approximate, finite samples only):** min ≈ ({grid[imin]:.6g}, {vals[imin]:.6g}); max ≈ ({grid[imax]:.6g}, {vals[imax]:.6g}). Verify endpoints and discontinuities before treating these as global interval extrema.")
        st.write(f"**Domain/discontinuities:** function is evaluated numerically across the chosen interval; undefined and non-finite values are omitted rather than joined across gaps. Expression: `{sp.sstr(custom_expr)}`. For rational/log functions, check denominator zeros and logarithm arguments explicitly.")
    elif course == "Precalculus":
        st.write(f"**Degree/coefficient effects:** `{sp.sstr(custom_expr)}`. Changing coefficients can change steepness, turning points, and end behavior. The second derivative is `{sp.sstr(d2_expr)}`; its sign indicates concavity where defined.")
    else:
        st.write(f"**Domain/range and end behavior:** inspect the plotted interval and expression `{sp.sstr(custom_expr)}`. Average rate of change is the secant slope; increasing/decreasing trends are visible in the graph. The derivative expressions are available above for extension into calculus.")
else:
    st.info("Enter a valid function and an increasing analysis interval to enable function analysis.")

st.subheader("6. Contextual simulation")
visual = st.selectbox("Visual simulation", ["Walking person with prosthetic leg", "Moving object / projectile", "Tank or drug concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"])
if custom_expr is not None and analysis_ok:
    v_x = st.session_state.get("sim_x", x_lo)
    v_y = safe_numeric(custom_expr, np.array([v_x]), x)[0]
    # Normalize value to a simple visual gauge; never claim physical scale unless units match the selected context.
    norm = 0.5 if not np.isfinite(v_y) else float(np.clip(0.5 + 0.1*np.tanh(v_y/10), 0, 1))
    st.progress(norm, text=f"{visual} • normalized illustrative indicator ({norm*100:.0f}%). Scaling: 0–100% gauge uses 0.5 + 0.1·tanh(f(x)/10); it is illustrative, not a measured physical scale.")
    if visual == "Walking person with prosthetic leg":
        st.write("Illustrative gait phase follows the selected input: " + ("stance phase" if (v_x % 1) < .6 else "swing phase") + f"; the function value still controls the normalized indicator ({fmt(v_y)} {y_units}). This is not a biomechanical or medical assessment.")
    elif visual == "Moving object / projectile":
        st.write(f"Object indicator at the selected input is {fmt(v_y)} {y_units}. Use a physically appropriate displacement function and units before interpreting it as a real trajectory.")
    elif visual == "Tank or drug concentration":
        st.write(f"Tank/concentration indicator is {fmt(v_y)} {y_units}; nonphysical negative concentrations should be flagged by the student rather than silently clipped.")
    elif visual == "Population / cell colony":
        st.write(f"Population/cell indicator is {fmt(v_y)} {y_units}; exponential growth is only a simplified model and may not remain valid indefinitely.")
    elif visual == "Thermometer / weather":
        st.write(f"Weather indicator is {fmt(v_y)} {y_units}; temperature models need location, time scale, and seasonal assumptions.")
    else:
        st.write(f"Business/economic indicator is {fmt(v_y)} {y_units}; the derivative represents modeled growth or decline, not a causal explanation.")
else:
    st.caption("A valid function and interval are needed for the simulation.")

st.subheader("7. Exports")
metadata = pd.DataFrame([{
    "country": country_label, "country_code": country_code, "indicator": indicator_name, "indicator_code": indicator_code,
    "definition": definition, "units": units, "requested_start_year": start_default, "requested_end_year": end_default,
    "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "actual_coverage": f"{int(validated_df.year.min())}–{int(validated_df.year.max())}" if len(validated_df) else "none",
    "observations": len(validated_df), "edited_scenario": st.session_state.edited_scenario,
    "degree": degree if model else None, "equation": equation if model else "not fitted",
    "in_sample_r2": model["r2"] if model else None, "in_sample_mae": model["mae"] if model else None,
}])
csv_bytes = validated_df.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", csv_bytes, file_name="working_data.csv", mime="text/csv")
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    metadata.to_excel(writer, sheet_name="Summary", index=False)
    validated_df.to_excel(writer, sheet_name="Working Data", index=False)
    batch_results.to_excel(writer, sheet_name="Calculated Years", index=False)
xlsx_buf.seek(0)
st.download_button("Download Excel workbook (.xlsx)", xlsx_buf.getvalue(), file_name="historical_data_explorer.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
pdf_buf = io.BytesIO()
styles = getSampleStyleSheet()
story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]), Spacer(1, 8)]
for _, row in metadata.iterrows():
    for key, val in row.items():
        story.append(Paragraph(f"<b>{key.replace('_',' ').title()}:</b> {str(val)}", styles["BodyText"]))
story.append(Spacer(1, 10))
story.append(Paragraph("Findings and limitations", styles["Heading2"]))
story.append(Paragraph("Polynomial fit is descriptive. In-sample R-squared and MAE do not establish causation or guarantee future performance. Extrapolations can be implausible; uncertainty is summarized with residuals and MAE, not a validated 95% prediction interval.", styles["BodyText"]))
if model:
    story.append(Spacer(1, 8)); story.append(Paragraph("Equation and metrics", styles["Heading2"]))
    story.append(Paragraph(equation.replace("&","&amp;"), styles["BodyText"]))
    story.append(Paragraph(f"In-sample R²: {fmt(model['r2'])}; in-sample MAE: {fmt(model['mae'])} {units}; n={model['n']}; mean={fmt(model['mean'])} {units}.", styles["BodyText"]))
    fig_pdf, ax_pdf = plt.subplots(figsize=(8, 3.5))
    ax_pdf.scatter(validated_df.year, validated_df.value, label="Observed data")
    xx_pdf = np.linspace(model["min_year"], model["max_year"]+horizon, 500)
    yy_pdf = poly_predict(model, xx_pdf)
    ax_pdf.plot(xx_pdf, yy_pdf, label="Polynomial fit / extrapolation")
    ax_pdf.axvline(model["max_year"], linestyle="--", label="Historical boundary")
    ax_pdf.set_xlabel("Year"); ax_pdf.set_ylabel(units); ax_pdf.legend(); ax_pdf.grid(True, alpha=.2)
    chart_bytes = make_chart_png(fig_pdf)
    story.append(Spacer(1, 8)); story.append(RLImage(chart_bytes, width=650, height=285))
if len(validated_df):
    story.append(Spacer(1, 8)); story.append(Paragraph("Data preview", styles["Heading2"]))
    preview = [["Year", "Value"]] + [[str(int(r.year)), fmt(r.value)] for r in validated_df.head(25).itertuples()]
    tab = Table(preview, repeatRows=1)
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]))
    story.append(tab)
if not batch_results.empty:
    story.append(Spacer(1, 8)); story.append(Paragraph("Calculated-year results", styles["Heading2"]))
    tab = Table([list(batch_results.columns)] + batch_results.astype(str).values.tolist()[:50], repeatRows=1)
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
    story.append(tab)
doc = SimpleDocTemplate(pdf_buf, pagesize=landscape(letter), title="Hispanic Heritage Historical Data Explorer")
doc.build(story)
pdf_buf.seek(0)
st.download_button("Download PDF report", pdf_buf.getvalue(), file_name="historical_data_report.pdf", mime="application/pdf")
if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(f"{country_label} • {indicator_name} • {units}")
    st.write(f"Coverage: {metadata.iloc[0]['actual_coverage']} | Observations: {len(validated_df)} | Scenario edited: {st.session_state.edited_scenario}")
    if model: st.write(f"Model: degree {degree}; {equation}; in-sample R²={fmt(model['r2'])}; in-sample MAE={fmt(model['mae'])} {units}.")
    st.write("Interpretation: historical association only; no causal claims. Extrapolation is uncertain and not a validated prediction interval.")

with st.expander("Sources, definitions, and classroom notes"):
    st.markdown("Official indicator catalog: [World Bank Indicators](https://data.worldbank.org/indicator). Country and indicator observations are retrieved through the [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation).")
    st.markdown("Brazil is a Latin American comparison country but Portuguese is its official language. The nine country options are selected for this project, not a claim that they are all Spanish-speaking.")
    st.markdown("Mathematical caution: polynomial regression minimizes squared residuals on the supplied observations. High-degree polynomials may oscillate and extrapolate poorly. In-sample R²/MAE are not out-of-sample guarantees.")
