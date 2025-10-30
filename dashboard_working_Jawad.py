# dashboard_scaled_separate_fit_fixed_v2_x0.py
import os
import streamlit as st
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from Full_Model_Definition import MA3_PM

# === CONFIG ===
days_in_year = 365
spinup_repeats = 2
spinup_days = spinup_repeats * days_in_year

# === USER CONFIG ===
scale_end_index = 1826      # rows to FIT scaler
end_index = 3655            # rows to APPLY scaler to
use_spinup = True




# === FILE PATHS ===
import os

# Base directory of the app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for data and model
forcing_file = os.path.join(BASE_DIR, "9223000_IRC.csv")
model_save_path = os.path.join(BASE_DIR, "9223000_HY_90_2.pth")

# === FILE PATHS ===
#forcing_file = r"E:/U_Arizona/Research_Data/Salt_project/Data/Up_Stream_Hoseshoe/python/Mode_Architectures_MCP_Type/IRC_Work/9223000_IRC.csv"
#model_save_path = r"E:/U_Arizona/Research_Data/Salt_project/Data/Up_Stream_Hoseshoe/python/Mode_Architectures_MCP_Type/IRC_Work/9223000_HY_90_2.pth"

# === PAGE CONFIG ===
st.set_page_config(page_title="Hydro Dashboard", layout="wide", page_icon="💧")
st.title("💧 IRC Model Dashboard")

# === LOAD RAW DATA ===
df = pd.read_csv(forcing_file)
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
else:
    df["Date"] = pd.date_range("2000-01-01", periods=len(df))

if df.shape[0] < end_index:
    st.error(f"Data has only {df.shape[0]} rows but end_index={end_index}. Reduce end_index.")
    st.stop()

# === SPLIT FOR SCALER AND MODEL ===
df_for_scaler = df.iloc[:scale_end_index].reset_index(drop=True)
df_for_model = df.iloc[:end_index].reset_index(drop=True)

# Detect columns
forcing_cols = [c for c in df.columns if c != "Date"]
precip_col = "APCP" if "APCP" in df.columns else forcing_cols[0]
temp_col = "Air_Temp" if "Air_Temp" in df.columns else (forcing_cols[1] if len(forcing_cols) > 1 else forcing_cols[0])

# === SCALE (fit on limited, apply on full) ===
scaler = MinMaxScaler()
scaler.fit(df_for_scaler.iloc[:, 1:].values.astype(float))
scaled_full_forcing_values = scaler.transform(df_for_model.iloc[:, 1:].values.astype(float))
#st.sidebar.success(f"Scaler fitted on 0–{scale_end_index-1} rows; applied to 0–{end_index-1}.")

# === SPINUP ===
if use_spinup:
    spinup_raw = np.vstack([df_for_model.iloc[:days_in_year, 1:].values] * spinup_repeats)
    spinup_scaled = np.vstack([scaled_full_forcing_values[:days_in_year]] * spinup_repeats)
    full_forcing_with_spinup = np.vstack([spinup_raw, df_for_model.iloc[:, 1:].values])
    scaled_full_forcing_with_spinup = np.vstack([spinup_scaled, scaled_full_forcing_values])
else:
    full_forcing_with_spinup = df_for_model.iloc[:, 1:].values
    scaled_full_forcing_with_spinup = scaled_full_forcing_values

# === Observed runoff ===
observed = df_for_model.iloc[:, 15].values.astype(np.float32)
catchment_area = df.iloc[0, 14]
scaling_factor = 2.4466 / catchment_area
observed *= scaling_factor

# === Convert to tensors ===
forcing_tensor = torch.tensor(full_forcing_with_spinup, dtype=torch.float32)
scaled_forcing_tensor = torch.tensor(scaled_full_forcing_with_spinup, dtype=torch.float32)

# === Load Model ===
device = torch.device("cpu")
model = MA3_PM().to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()
st.sidebar.success("Model loaded successfully.")

# === Sidebar controls ===
st.sidebar.header("Date Range")
# Convert pandas Timestamps to datetime.date

min_date = df_for_model["Date"].min().date()
max_date = df_for_model["Date"].max().date()

# Start date in sidebar
start_date = st.sidebar.date_input(
    "Start Date",
    min_date,
    min_value=min_date,
    max_value=max_date
)

# Default end date
default_end_date = pd.to_datetime("2007-09-30").date()

# Ensure default_end_date is within valid range
if default_end_date < min_date:
    default_end_date = min_date
elif default_end_date > max_date:
    default_end_date = max_date

# End date in sidebar
end_date = st.sidebar.date_input(
    "End Date",
    value=default_end_date,
    min_value=min_date,
    max_value=max_date
)

#st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

st.sidebar.header("Model Perturbations")
precip_scale = st.sidebar.slider("Precipitation Scale", 0.1, 5.0, 1.0, 0.05)
temp_scale = st.sidebar.slider("Temperature Scale", 0.1, 5.0, 1.0, 0.05)
k_factor = st.sidebar.slider("Soil Water Holdig Capacity Factor", 0.1, 3.0, 1.0, 0.01)

st.sidebar.header("Streamflow Curve Display")
show_obs = st.sidebar.checkbox("Observed", True)
show_base = st.sidebar.checkbox("Baseline", True)
show_pert = st.sidebar.checkbox("Perturbed", True)

# === Run Model ===
with torch.no_grad():
    base_out = model(forcing_tensor.to(device), scaled_forcing_tensor.to(device), 1.0, 1.0, 1.0)
    baseline_runoff = base_out.squeeze(-1).cpu().numpy()[spinup_days:]

    pert_out = model(forcing_tensor.to(device), scaled_forcing_tensor.to(device),
                     float(precip_scale), float(temp_scale), float(k_factor))
    perturbed_runoff = pert_out.squeeze(-1).cpu().numpy()[spinup_days:]

# === Trim lengths ===
min_len = min(len(observed), len(baseline_runoff), len(perturbed_runoff))
dates_use = df_for_model["Date"].iloc[:min_len]

plot_df = pd.DataFrame({
    "Date": dates_use,
    "Obs": observed[:min_len],
    "Base": baseline_runoff[:min_len],
    "Pert": perturbed_runoff[:min_len],
    "P_orig": df_for_model[precip_col].iloc[:min_len],
    "T_orig": df_for_model[temp_col].iloc[:min_len],
})
plot_df["P_disp"] = plot_df["P_orig"] * precip_scale
plot_df["T_disp"] = plot_df["T_orig"] * temp_scale

mask = (plot_df["Date"] >= pd.to_datetime(start_date)) & (plot_df["Date"] <= pd.to_datetime(end_date))
plot_df = plot_df.loc[mask]

# === Metrics ===
def kge(sim, obs):
    sim, obs = np.asarray(sim), np.asarray(obs)
    if obs.std() == 0 or sim.std() == 0: return np.nan
    r = np.corrcoef(sim, obs)[0, 1]; alpha = sim.std()/obs.std(); beta = sim.mean()/obs.mean()
    kge=1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    kges_ss=1-((1-kge)/np.sqrt(2))
    return kges_ss

def rmse(sim, obs): return np.sqrt(np.mean((np.asarray(sim)-np.asarray(obs))**2))
def rcoef(sim, obs): return np.corrcoef(sim, obs)[0, 1] if np.std(sim)>0 and np.std(obs)>0 else np.nan

KGE = kge(plot_df["Base"], plot_df["Obs"])
R = rcoef(plot_df["Base"], plot_df["Obs"])
RMSE = rmse(plot_df["Base"], plot_df["Obs"])
mean_obs = np.mean(plot_df["Obs"])

st.markdown(
    f"""
    <div style='background:#f8f9fa;padding:10px 16px;border-radius:8px;text-align:center;'>
    <b>Performance (Selected Window)</b><br>
    KGE: <b>{KGE:.3f}</b> | R: <b>{R:.3f}</b> | RMSE: <b>{RMSE:.3f}</b> | Mean(Obs): <b>{mean_obs:.3f}</b>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("---")

# === Streamflow Plot ===
fig_sf = go.Figure()
if show_obs:
    fig_sf.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Obs"],
                                mode="markers", name="Observed",
                                marker=dict(color="red", symbol="circle-open", size=7)))
if show_base:
    fig_sf.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Base"],
                                mode="lines", name="Baseline", line=dict(color="black", width=2)))
if show_pert:
    fig_sf.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Pert"],
                                mode="lines", name="Perturbed", line=dict(color="blue", width=2)))

fig_sf.update_layout(
    title="Streamflow Time Series",
    xaxis_title="Date", yaxis_title="Runoff (mm/day)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    height=520, margin=dict(l=40, r=40, t=80, b=40)
)
st.plotly_chart(fig_sf, use_container_width=True)

# === Scatters (force x-axis from 0) ===
# === Scatters (force x-axis from 0) ===
col1, col2 = st.columns(2)

scatter_comb = pd.DataFrame({
    "Precip": np.concatenate([plot_df["P_orig"], plot_df["P_disp"]]),
    "Streamflow": np.concatenate([plot_df["Base"], plot_df["Pert"]]),
    "Type": ["Baseline"] * len(plot_df) + ["Perturbed"] * len(plot_df)
})

# compute explicit positive ranges (use 1e-6 to avoid zero-size range)
xmax_sc1 = max(scatter_comb["Streamflow"].max(), 0) * 1.05 + 1e-6
ymax_sc1 = max(scatter_comb["Precip"].max(), 0) * 1.05 + 1e-6

fig_sc1 = px.scatter(scatter_comb, x="Streamflow", y="Precip", color="Type",
                     title="Precipitation vs Streamflow (Baseline & Perturbed)")
fig_sc1.update_layout(height=520)
# remove scaleanchor and explicitly set ranges starting at 0
fig_sc1.update_xaxes(range=[0, xmax_sc1], autorange=False)
fig_sc1.update_yaxes(range=[0, ymax_sc1], autorange=False)

col1.plotly_chart(fig_sc1, use_container_width=True)


# === Right Scatter: Observed vs Baseline ===
fig_sc2 = px.scatter(plot_df, x="Obs", y="Base", title="Observed vs Simulated Streamflow (mm)")
fig_sc2.add_trace(go.Scatter(
    x=[0, max(plot_df["Obs"].max(), plot_df["Base"].max())],
    y=[0, max(plot_df["Obs"].max(), plot_df["Base"].max())],
    mode="lines", line=dict(color="gray", dash="dash"), showlegend=False
))

# Compute common axis limit (same for x and y)
common_max = max(plot_df["Obs"].max(), plot_df["Base"].max()) * 1.05
common_max = max(common_max, 1e-6)  # ensure > 0

fig_sc2.update_xaxes(range=[0, common_max], autorange=False)
fig_sc2.update_yaxes(range=[0, common_max], autorange=False)

fig_sc2.update_layout(
    height=520,
    width=None,
    yaxis_scaleanchor=None,  # remove anchor so manual limits hold
    xaxis_scaleanchor=None,
)

col2.plotly_chart(fig_sc2, use_container_width=True)


# === Precipitation and Temperature ===
fig_p = go.Figure()
fig_p.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["P_orig"], mode="lines", name="Precip (orig)"))
fig_p.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["P_disp"], mode="lines", name=f"Precip ×{precip_scale}", line=dict(dash="dash")))
fig_p.update_layout(title="Precipitation", height=360)
st.plotly_chart(fig_p, use_container_width=True)

fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["T_orig"], mode="lines", name="Temp (orig)", line=dict(color="orange")))
fig_t.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["T_disp"], mode="lines", name=f"Temp ×{temp_scale}", line=dict(color="orange", dash="dash")))
fig_t.update_layout(title="Temperature", height=360)
st.plotly_chart(fig_t, use_container_width=True)

# === Data Table and Download ===
st.markdown("### Filtered Time Series Data")
st.dataframe(plot_df, use_container_width=True)
st.download_button("Download CSV", plot_df.to_csv(index=False).encode("utf-8"),
                   "hydro_filtered_truncated.csv", "text/csv")
