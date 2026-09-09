# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import gamma as gamma_dist, lognorm, pearsonr

st.set_page_config(page_title="Universal Epidemic Analyzer", page_icon="🦠", layout="wide")

# ------------------ Helpers ------------------
def guess_columns(df: pd.DataFrame):
    cols = [str(c).lower() for c in df.columns]
    def pick(keys):
        for i,c in enumerate(cols):
            if any(k in c for k in keys): return df.columns[i]
        return None
    date_col = pick(["date","дата","day","день","time","время"])
    inc_col  = pick(["new","cases","случ","нов","incid","забол"])
    cum_col  = pick(["cum","total","накоп","всего","итог","sum"])
    return date_col, inc_col, cum_col

def guess_death_column(df: pd.DataFrame):
    cols = [str(c).lower() for c in df.columns]
    keys = ["death","deaths","умер","летал","погиб","смерт"]
    for i,c in enumerate(cols):
        if any(k in c for k in keys): return df.columns[i]
    return None

def to_incidence(df: pd.DataFrame, date_col, inc_col, cum_col) -> pd.DataFrame:
    w = df.copy()
    if date_col is not None:
        w["date"] = pd.to_datetime(w[date_col], errors="coerce")
    else:
        w["date"] = pd.to_datetime(range(len(w)), unit="D", origin="2020-01-01")

    if inc_col is not None:
        inc = pd.to_numeric(w[inc_col], errors="coerce")
    elif cum_col is not None:
        cum = pd.to_numeric(w[cum_col], errors="coerce")
        inc = cum.diff().fillna(cum.iloc[0])
    else:
        num_cols = [c for c in w.columns if c!="date" and pd.api.types.is_numeric_dtype(w[c])]
        if not num_cols: raise ValueError("No numeric column with new or cumulative cases found.")
        inc = pd.to_numeric(w[num_cols[0]], errors="coerce")

    out = pd.DataFrame({"date": w["date"], "incidence": inc})
    out["incidence"] = out["incidence"].clip(lower=0).fillna(0).astype(int)
    out = (out.dropna(subset=["date"])
              .groupby("date", as_index=False)["incidence"].sum()
              .sort_values("date"))
    out = out.set_index("date").asfreq("D", fill_value=0).reset_index()
    return out

def to_cumulative(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if not (s.diff().fillna(0) >= 0).all():
        s = s.clip(lower=0).cumsum()
    return s

def discretize_generation_time(mean_si=4.8, sd_si=2.3, max_days=30, kind="gamma"):
    days = np.arange(1, max_days+1)
    if kind == "gamma":
        k = (mean_si / sd_si) ** 2
        theta = (sd_si ** 2) / mean_si
        cdf = gamma_dist.cdf(days, a=k, scale=theta)
        cdf_prev = gamma_dist.cdf(days-1, a=k, scale=theta)
    else:
        sigma2 = np.log(1 + (sd_si**2)/(mean_si**2))
        sigma = np.sqrt(sigma2)
        mu = np.log(mean_si) - sigma2/2
        cdf = lognorm.cdf(days, s=sigma, scale=np.exp(mu))
        cdf_prev = lognorm.cdf(days-1, s=sigma, scale=np.exp(mu))
    pmf = (cdf - cdf_prev)
    return pmf / pmf.sum()

def cori_rt(incidence: np.ndarray, w: np.ndarray, tau:int=7, a:float=1.0, b:float=5.0):
    T, L = len(incidence), len(w)
    lam = np.zeros(T)
    for t in range(T):
        smax = min(t, L)
        lam[t] = (incidence[t - np.arange(1, smax+1)] * w[:smax]).sum() if smax>0 else 0.0
    Rt_mean = np.full(T, np.nan); Rt_low = np.full(T, np.nan); Rt_high = np.full(T, np.nan)
    for t in range(T):
        t0 = max(0, t - tau + 1)
        I_sum = incidence[t0:t+1].sum()
        L_sum = lam[t0:t+1].sum()
        shape = a + I_sum
        rate  = 1.0/b + L_sum
        if rate > 0:
            Rt_mean[t] = shape / rate
            Rt_low[t]  = gamma_dist.ppf(0.025, a=shape, scale=1.0/rate)
            Rt_high[t] = gamma_dist.ppf(0.975, a=shape, scale=1.0/rate)
    return lam, Rt_mean, Rt_low, Rt_high

def r_to_R0(r: float, mean_si: float, sd_si: float, kind: str = "gamma"):
    t = np.arange(1, 60)
    w = discretize_generation_time(mean_si, sd_si, max_days=len(t), kind=kind)
    lt = (w * np.exp(-r * t)).sum()
    return (1.0 / lt) if lt > 0 else np.nan

VIRUS_PRESETS = {
    "COVID-19 (early)":  (4.8, 2.3),
    "COVID-19 (Omicron)":(3.2, 1.8),
    "Influenza":         (2.6, 1.3),
    "Measles":           (11.7, 2.0),
    "RSV":               (6.0, 2.5),
    "Norovirus":         (2.0, 1.0),
    "Custom":            (5.0, 2.0),
}

# ------------------ Sidebar ------------------
st.sidebar.title("Parameters")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel (date + cases; deaths optional)", type=["csv","xlsx","xls"])

virus = st.sidebar.selectbox("Virus preset (sets serial interval)", list(VIRUS_PRESETS.keys()), index=0)
si_kind = st.sidebar.selectbox("Serial-interval shape", ["gamma","lognormal"], index=0)
mean_si, sd_si = st.sidebar.number_input("Mean SI (days)", value=VIRUS_PRESETS[virus][0], step=0.1), \
                 st.sidebar.number_input("SD SI (days)", value=VIRUS_PRESETS[virus][1], step=0.1)

tau = st.sidebar.slider("Rt window (days)", 3, 14, 7, 1)
min_cases_for_rt = st.sidebar.slider("Min cases for valid Rt", 0, 50, 10, 1)
smooth = st.sidebar.checkbox("7-day centered smoothing", True)

N = st.sidebar.number_input("Population N", min_value=1000, value=1_000_000, step=1000)
infectious_days = st.sidebar.number_input("Infectious period (days)", min_value=0.1, value=5.0, step=0.1)
ascertain = st.sidebar.slider("Ascertainment (share detected)", 0.1, 1.0, 0.5, 0.05)
sero_conv = st.sidebar.slider("Seroconversion after infection", 0.5, 1.0, 0.9, 0.05)
vax_cov = st.sidebar.slider("Vaccination coverage", 0.0, 1.0, 0.4, 0.05)
vax_eff = st.sidebar.slider("Vaccine efficacy (against infection)", 0.0, 1.0, 0.7, 0.05)

beta_manual = st.sidebar.number_input("β (/day) for model R0 (optional)", value=0.35, step=0.01)
gamma_manual = st.sidebar.number_input("γ (/day) for model R0 (optional)", value=0.20, step=0.01)

export = st.sidebar.checkbox("Enable CSV export", False)

st.title("🦠 Universal Epidemic Analyzer")

# ------------------ Load data ------------------
if uploaded is not None:
    content = uploaded.read()
    bio = io.BytesIO(content)
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(bio)
    else:
        df = pd.read_excel(bio)
else:
    # synthetic fallback
    days = np.arange(160)
    inc  = np.r_[np.zeros(5), np.exp(0.18*(days[5:25]-5)), np.linspace(60,8,135)].astype(int)
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(inc)), "new_cases": inc})
    st.info("No file uploaded — using a synthetic example.")

date_col, inc_col, cum_col = guess_columns(df)
data = to_incidence(df, date_col, inc_col, cum_col)
if smooth:
    data["incidence_smoothed"] = data["incidence"].rolling(7, center=True, min_periods=1).mean().round().astype(int)
else:
    data["incidence_smoothed"] = data["incidence"]

# deaths (optional)
death_col = guess_death_column(df)
CFR_overall = np.nan
if death_col is not None:
    dd = df.copy()
    if "date" not in dd.columns:
        dd["date"] = pd.to_datetime(range(len(dd)), unit="D", origin="2020-01-01")
    dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
    dd = dd.dropna(subset=["date"])
    deaths_cum = to_cumulative(dd.set_index("date")[death_col]).asfreq("D", fill_value=0)
    cases_cum  = data.set_index("date")["incidence"].cumsum()
    join = pd.concat([cases_cum.rename("cum_cases"),
                      deaths_cum.rename("cum_deaths")], axis=1).fillna(method="ffill").fillna(0)
    if join["cum_cases"].iloc[-1] > 0:
        CFR_overall = 100.0 * join["cum_deaths"].iloc[-1] / join["cum_cases"].iloc[-1]

# ------------------ Rt ------------------
w = discretize_generation_time(mean_si, sd_si, max_days=30, kind=si_kind)
lam, Rt_mean, Rt_low, Rt_high = cori_rt(data["incidence_smoothed"].to_numpy(), w, tau=tau)
data["Rt_mean"], data["Rt_low"], data["Rt_high"] = Rt_mean, Rt_low, Rt_high

# ------------------ gamma, S_t, beta(t) ------------------
gamma_rate = 1.0 / infectious_days
cum_cases_obs = data["incidence"].cumsum().to_numpy()
cum_inf_true  = cum_cases_obs / max(ascertain, 1e-6)
immune_inf    = cum_inf_true * sero_conv
immune_vax    = N * vax_cov * vax_eff
immune_total  = immune_inf + immune_vax - (immune_inf * immune_vax) / max(N,1)
immune_total  = np.clip(immune_total, 0, N)
immune_share  = immune_total / max(N,1)
S_t = np.clip(1.0 - immune_share, 0.01, 1.0)

beta_t = Rt_mean * gamma_rate / S_t
beta_t = np.where(np.isfinite(beta_t), beta_t, np.nan)

# ------------------ Valid mask ------------------
valid_mask = (
    np.isfinite(Rt_mean) &
    (data["incidence_smoothed"].to_numpy() >= min_cases_for_rt) &
    (lam > 0)
)
Rt_valid   = np.where(valid_mask, Rt_mean, np.nan)
RtL_valid  = np.where(valid_mask, Rt_low,  np.nan)
RtH_valid  = np.where(valid_mask, Rt_high, np.nan)
beta_valid = np.where(valid_mask, beta_t,  np.nan)

# ------------------ R0 ------------------
R0_model = (beta_manual / gamma_manual) if (gamma_manual and gamma_manual>0) else np.nan

early = data.loc[data["incidence_smoothed"]>=min_cases_for_rt].head(14)
if len(early) >= 5:
    x = (early["date"] - early["date"].iloc[0]).dt.days.to_numpy()
    y = np.log(early["incidence_smoothed"].replace(0, np.nan)).dropna()
    x = x[-len(y):]
    A = np.vstack([x, np.ones_like(x)]).T
    r, _ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)[0]
    R0_from_r = r_to_R0(r, mean_si, sd_si, kind=si_kind)
else:
    R0_from_r = np.nan

Rt0_valid_series = pd.Series(Rt_valid).replace([np.inf,-np.inf], np.nan).dropna()
R0_initialRt = Rt0_valid_series.iloc[0] if len(Rt0_valid_series) else np.nan
R0_pick = next((v for v in [R0_model, R0_from_r, R0_initialRt] if np.isfinite(v)), np.nan)
HIT = (1.0 - 1.0/R0_pick) if (np.isfinite(R0_pick) and R0_pick>1) else np.nan

# ------------------ Vaccination analysis ------------------
Rt_vacc_valid = Rt_valid * (1.0 - vax_cov*vax_eff)
prevent_frac = np.clip(1.0 - (Rt_vacc_valid / Rt_valid), 0, 1)
prevent_frac = np.where(np.isfinite(prevent_frac), prevent_frac, np.nan)

# ------------------ Pearson correlations ------------------
vm = valid_mask & np.isfinite(beta_valid)
r2 = p2 = np.nan
if vm.any():
    r2, p2 = pearsonr(Rt_valid[vm], beta_valid[vm])
vm2 = valid_mask
r1 = p1 = np.nan
if vm2.any():
    r1, p1 = pearsonr(Rt_valid[vm2], data["incidence_smoothed"].to_numpy()[vm2])

# ------------------ Charts ------------------
col1, col2 = st.columns([2,1])

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data["date"], y=data["incidence"], name="Incidence (raw)", mode="lines"))
    fig1.add_trace(go.Scatter(x=data["date"], y=data["incidence_smoothed"], name="Smoothed", mode="lines"))
    fig1.update_layout(title="Daily incidence", legend=dict(orientation="h"))
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data["date"], y=Rt_valid, name="Rₜ (valid)", mode="lines"))
    grey = np.where(~valid_mask, Rt_mean, np.nan)
    fig2.add_trace(go.Scatter(x=data["date"], y=grey, name="Rₜ (discarded)", mode="lines", line=dict(color="lightgrey")))
    m = np.isfinite(RtL_valid) & np.isfinite(RtH_valid)
    if m.any():
        fig2.add_trace(go.Scatter(
            x=np.r_[data["date"][m], data["date"][m][::-1]],
            y=np.r_[RtH_valid[m], RtL_valid[m][::-1]],
            fill="toself", name="95% CrI", line=dict(color="rgba(0,0,0,0)"), fillcolor="rgba(0,0,255,0.2)"
        ))
    fig2.add_hline(y=1.0, line=dict(dash="dash"))
    fig2.update_layout(title=f"Time-varying Rₜ (min cases = {min_cases_for_rt})", legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data["date"], y=beta_valid, name="β(t) valid (/day)", mode="lines"))
    fig3.add_trace(go.Scatter(x=data["date"], y=np.where(~valid_mask, beta_t, np.nan),
                              name="β(t) discarded", mode="lines", line=dict(color="lightgrey")))
    fig3.add_hline(y=gamma_rate, line=dict(dash="dash"), annotation_text=f"γ ≈ {gamma_rate:.3f}/day")
    fig3.update_layout(title="Transmission β(t) and recovery γ", legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=data["date"], y=immune_share*100, name="Immune share (%)", mode="lines"))
    if np.isfinite(HIT):
        fig4.add_hline(y=100*HIT, line=dict(dash="dash"), annotation_text=f"HIT ≈ {100*HIT:.1f}% (R₀≈{R0_pick:.2f})")
    fig4.update_layout(title="Estimated immune share vs. HIT", legend=dict(orientation="h"))
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    st.subheader("Summary (valid-only)")
    Rt_mean_valid_mean = np.nanmean(Rt_valid)
    Rt_valid_peak = np.nanmax(Rt_valid)
    st.metric("Mean Rt (valid)", f"{Rt_mean_valid_mean:.2f}")
    st.metric("Peak Rt (valid)", f"{Rt_valid_peak:.2f}")
    st.metric("γ (/day)", f"{gamma_rate:.3f}")
    st.metric("CFR (overall)", f"{CFR_overall:.2f}%" if np.isfinite(CFR_overall) else "n/a")
    st.metric("Immune (final day, est.)", f"{100*immune_share[-1]:.1f}%")
    if np.isfinite(HIT):
        st.metric("Herd immunity threshold", f"{100*HIT:.1f}%")
    st.markdown("---")
    st.write(f"**R₀ (model β,γ):** {R0_model:.2f}" if np.isfinite(R0_model) else "**R₀ (model β,γ):** n/a")
    st.write(f"**R₀ (early growth):** {R0_from_r:.2f}" if np.isfinite(R0_from_r) else "**R₀ (early growth):** n/a")
    st.write(f"**Initial Rt (proxy R₀):** {R0_initialRt:.2f}" if np.isfinite(R0_initialRt) else "**Initial Rt:** n/a")
    st.markdown("---")
    st.write(f"**Pearson r(Rt, incidence)** = {r1:.3f} (p={p1:.3g})" if np.isfinite(r1) else "**Pearson r(Rt, incidence):** n/a")
    st.write(f"**Pearson r(Rt, β(t))** = {r2:.3f} (p={p2:.3g})" if np.isfinite(r2) else "**Pearson r(Rt, β(t))**: n/a")

# ------------------ Export ------------------
if export:
    out_df = data.copy()
    out_df["beta_t"] = beta_t
    out_df["beta_valid"] = beta_valid
    out_df["Rt_valid"] = Rt_valid
    out_df["gamma"] = gamma_rate
    out_df["immune_share"] = immune_share
    out_df["valid_mask"] = valid_mask
    csv = out_df.to_csv(index=False).encode()
    st.download_button("Download results CSV", csv, "epidemic_analysis_output.csv", "text/csv")