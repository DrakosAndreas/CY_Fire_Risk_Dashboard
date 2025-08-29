#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

import folium
from streamlit_folium import st_folium
from branca.colormap import linear  # για τους θεματικούς χάρτες

# -----------------------------------
# ΡΥΘΜΙΣΕΙΣ
# -----------------------------------
st.set_page_config(page_title="CY Fire Dashboard", layout="wide")

DATA_DIR   = os.path.abspath(os.getcwd())
POINTS_CSV = os.path.join(DATA_DIR, "cy_points_grid_2km.csv")
WEATHER_CSV= os.path.join(DATA_DIR, "cy_weather_sample.csv")      # ανανεώνεται από cron
ENRICH_CSV = os.path.join(DATA_DIR, "dom_enrich.csv")             # rain_7d_mm, wind_gust_mps
FWI_STATE  = os.path.join(DATA_DIR, "fwi_state.csv")              # FFMC/DMC/DC ανά σταθμό (state)

# -----------------------------------
# ΓΕΝΙΚΑ HELPERS
# -----------------------------------
def nz(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default

def clip01(x): return max(0.0, min(1.0, float(x)))

def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlmb = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def expo_score(km, max_km):
    if km is None or (isinstance(km, float) and math.isnan(km)):
        return 0.0
    d = max(0.0, float(km))
    return max(0.0, 1.0 - min(d, max_km)/max_km)

def aspect_wind_alignment(aspect_deg, wind_dir_deg):
    try:
        if aspect_deg is None or pd.isna(aspect_deg) or wind_dir_deg is None or pd.isna(wind_dir_deg):
            return 0.8
        diff = abs((float(aspect_deg) - float(wind_dir_deg) + 180) % 360 - 180)
        if diff <= 90:
            return 1.0 - (diff/90.0)*0.2
        else:
            return 0.8 - ((diff-90)/90.0)*0.2
    except Exception:
        return 0.8

def slope_factor(slope_deg, a=0.8):
    if slope_deg is None or pd.isna(slope_deg):
        return 1.0
    try:
        t = math.tan(math.radians(float(slope_deg)))
        return 1.0 + (t ** a)
    except Exception:
        return 1.0

def compute_ffwi_simple(temp_c, rh, wind_mps):
    # απλός “δικός μας” δείκτης (όχι FWI) – παραμένει για IL/SIP
    T = nz(temp_c, 25.0)
    H = nz(rh, 40.0)
    W = nz(wind_mps, 2.0)
    score = (T/40.0)*0.45 + ((100.0 - H)/100.0)*0.35 + (min(W,15)/15.0)*0.20
    return clip01(score)*100.0

def adjust_dryness(ffwi, rain7d_mm):
    r = nz(rain7d_mm, 0.0)
    red = min(r/50.0, 0.6)  # έως 60% μείωση
    return max(0.0, ffwi * (1.0 - red))

def veg_knn_majority(dom_df, points_df, k=5, radius_km=2.0):
    P = points_df[["lat","lon"]].to_numpy(float)
    veg = points_df["veg_class"].reset_index(drop=True)
    out = []
    for _, r in dom_df.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        if np.isnan(lat) or np.isnan(lon) or len(P)==0:
            out.append("unknown")
            continue
        d2 = (P[:,0]-lat)**2 + (P[:,1]-lon)**2
        k_eff = min(k, len(P))
        idx = np.argpartition(d2, k_eff-1)[:k_eff]
        cand = []
        for i in idx:
            if haversine_km(lat, lon, P[i,0], P[i,1]) <= radius_km:
                cand.append(veg.iloc[i])
        if not cand:
            j = int(np.argmin(d2))
            out.append(veg.iloc[j])
        else:
            out.append(pd.Series(cand).mode().iloc[0])
    return pd.Series(out)

VEG_TO_FUEL_FACTOR = {
    "grassland": 1.10, "pastures": 1.10,
    "sclerophyllous_shrubland": 1.20, "transitional_woodland_shrub": 1.15,
    "pine_forest": 1.30, "coniferous_forest": 1.30, "mixed_forest": 1.20,
    "olive_groves": 1.05, "vineyards": 1.05,
    "agriculture": 1.00, "urban": 0.95, "bare": 0.90, "water": 0.80,
}
def fuel_factor_from_veg(veg_class):
    veg = str(veg_class).strip().lower()
    return VEG_TO_FUEL_FACTOR.get(veg, 1.0)

def ensure_numeric(df, cols, fallback_gust=True):
    for c, d in cols.items():
        if c not in df.columns:
            df[c] = d
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if d is not None:
            df[c] = df[c].fillna(d)
    if fallback_gust:
        m = pd.to_numeric(df.get("wind_mps", 0.0), errors="coerce").fillna(0.0)
        if "wind_gust_mps" in df.columns:
            g = pd.to_numeric(df["wind_gust_mps"], errors="coerce").fillna(0.0)
            df.loc[(g<=0) | df["wind_gust_mps"].isna(), "wind_gust_mps"] = (m*1.3).clip(upper=20)
        else:
            df["wind_gust_mps"] = (m*1.3).clip(upper=20)
    return df

# -----------------------------------
# FWI (Canadian System) – ενσωματωμένο
# -----------------------------------
LE_DMC = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
LF_DC  = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]

def ffmc_calc(prev_ffmc, T, H, U_kmh, P):
    m = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)
    if P > 0.5:
        rf = P - 0.5
        if m <= 150.0:
            m = m + 42.5 * rf * math.exp(-100.0 / (251.0 - m)) * (1 - math.exp(-6.93 / rf))
        else:
            m = m + 42.5 * rf * math.exp(-100.0 / (251.0 - m)) * (1 - math.exp(-6.93 / rf)) \
                + 0.0015 * (m - 150.0) ** 2 * math.sqrt(rf)
        m = min(m, 250.0)
    Ed = 0.942 * (H ** 0.679) + (11.0 * math.exp((H - 100.0) / 10.0)) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))
    Ew = 0.618 * (H ** 0.753) + (10.0 * math.exp((H - 100.0) / 10.0)) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))
    if m < Ed:
        ko = 0.424 * (1.0 - ((H/100.0) ** 1.7)) + (0.0694 * math.sqrt(U_kmh)) * (1.0 - ((H/100.0) ** 8))
        kd = ko * 0.581 * math.exp(0.0365 * T)
        m = Ew + (m - Ew) * math.exp(-kd)
    else:
        kw = 0.424 * (1.0 - ((H/100.0) ** 1.7)) + 0.0694 * math.sqrt(U_kmh) * (1.0 - ((H/100.0) ** 8))
        kw = kw * 0.581 * math.exp(0.0365 * T)
        m = Ed + (m - Ed) * math.exp(-kw)
    ffmc = (59.5 * (250.0 - m)) / (147.2 + m)
    return max(0.0, min(101.0, ffmc))

def dmc_calc(prev_dmc, T, H, P, month):
    dmc = prev_dmc
    if P > 1.5:
        re = 0.92 * P - 1.27
        mo = 20.0 + math.exp(5.6348 - prev_dmc / 43.43)
        if prev_dmc <= 33.0:
            b = 100.0 / (0.5 + 0.3 * prev_dmc)
        elif prev_dmc <= 65.0:
            b = 14.0 - 1.3 * math.log(prev_dmc)
        else:
            b = 6.2 * math.log(prev_dmc) - 17.2
        mr = mo + (1000.0 * re) / (48.77 + b * re)
        dmc = 244.72 - 43.43 * math.log(mr - 20.0)
    T_eff = max(-1.1, T)
    K = 1.894 * (T_eff + 1.1) * (100.0 - H) * LE_DMC[month - 1] * 1e-6
    return max(0.0, dmc + K)

def dc_calc(prev_dc, T, P, month):
    dc = prev_dc
    if P > 2.8:
        rd = 0.83 * P - 1.27
        Qo = 800.0 * math.exp(-prev_dc / 400.0)
        Qr = Qo + 3.937 * rd
        dc = max(0.0, 400.0 * math.log(800.0 / Qr))
    T_eff = max(-2.8, T)
    V = 0.36 * (T_eff + 2.8) + LF_DC[month - 1]
    V = max(0.0, V)
    return dc + V

def isi_calc(ffmc, U_kmh):
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fU = math.exp(0.05039 * U_kmh)
    fF = 91.9 * math.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / (4.93e7))
    return 0.208 * fU * fF

def bui_calc(dmc, dc):
    if dmc <= 0.4 * dc:
        return (0.8 * dmc * dc) / (dmc + 0.4 * dc + 1e-9)
    else:
        return dmc - (1.0 - (0.8 * dc) / (dmc + 0.4 * dc + 1e-9)) * (0.92 + (0.0114 * dmc) ** 1.7)

def fwi_calc(isi, bui):
    if bui <= 0.0:
        F_D = 0.0
    else:
        F_D = (0.626 * (bui ** 0.809) + 2.0) / (1.0 + 0.0186 * bui)
    B = 0.1 * isi * F_D
    if B <= 1.0:
        return B
    else:
        return math.exp(2.72 * ((math.log(B) * 0.434) ** 0.647))

def fwi_level(fwi):
    if pd.isna(fwi): return ""
    if fwi < 5:  return "Low"
    if fwi < 12: return "Moderate"
    if fwi < 21: return "High"
    if fwi < 38: return "Very High"
    return "Extreme"

# ---------- STATE HELPERS ----------
DEFAULT_FFMC, DEFAULT_DMC, DEFAULT_DC = 85.0, 6.0, 15.0

def make_key(lat, lon, prec=5):
    try:
        return f"{float(lat):.{prec}f},{float(lon):.{prec}f}"
    except Exception:
        return None

# Φόρτωση/δημιουργία state
def load_fwi_state(state_path=FWI_STATE):
    if os.path.exists(state_path):
        try:
            df = pd.read_csv(state_path)
            need = {"key","FFMC","DMC","DC","last_yyyymmdd"}
            if not need.issubset(df.columns):
                raise ValueError("Invalid fwi_state.csv structure")
            for c in ["FFMC","DMC","DC"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["last_yyyymmdd"] = df.get("last_yyyymmdd","").astype(str)
            df["key"] = df.get("key","").astype(str)
            return df
        except Exception as e:
            print(f"[WARN] fwi_state load failed: {e}")
            return pd.DataFrame(columns=["key","FFMC","DMC","DC","last_yyyymmdd"])
    else:
        return pd.DataFrame(columns=["key","FFMC","DMC","DC","last_yyyymmdd"])

# αυτή είναι η stateful έκδοση με ασφαλή λήψη "month" (ZoneInfo)
def add_fwi_columns_stateful(df, fwi_state_df, persist_updates=False, state_path=FWI_STATE):
    """
    Προσθέτει FFMC,DMC,DC,ISI,BUI,FWI,fwi_level σε df με στήλες temp_c,rh,wind_mps (m/s).
    Χρησιμοποιεί fwi_state_df για "χθεσινούς" κώδικες ανά σταθμό (key=lat,lon).
    Αν persist_updates=True, γράφει πίσω τα νέα state στο state_path.
    """
    if df.empty:
        for c in ["FFMC","DMC","DC","ISI","BUI","FWI","fwi_level"]:
            df[c] = np.nan if c!="fwi_level" else ""
        return df

    # ασφαλής λήψη τοπικού μήνα
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        month = datetime.now(ZoneInfo("Asia/Nicosia")).month
    except Exception:
        month = datetime.now().month

    # mapping από key -> index στο state df
    idx = {k: i for i, k in enumerate(fwi_state_df.get("key", []))} if not fwi_state_df.empty else {}

    rows = []
    today_str = datetime.now().strftime("%Y%m%d")

    for _, r in df.iterrows():
        try:
            k = f"{float(r['lat']):.5f},{float(r['lon']):.5f}"
            if k in idx:
                srow = fwi_state_df.loc[idx[k]]
                prev_ffmc = nz(srow.get("FFMC"), DEFAULT_FFMC)
                prev_dmc  = nz(srow.get("DMC"),  DEFAULT_DMC)
                prev_dc   = nz(srow.get("DC"),   DEFAULT_DC)
            else:
                prev_ffmc, prev_dmc, prev_dc = DEFAULT_FFMC, DEFAULT_DMC, DEFAULT_DC

            T = float(pd.to_numeric(r.get("temp_c"), errors="coerce"))
            H = float(pd.to_numeric(r.get("rh"), errors="coerce"))
            U_kmh = float(pd.to_numeric(r.get("wind_mps"), errors="coerce")) * 3.6
            P = float(pd.to_numeric(r.get("precip_mm"), errors="coerce")) if "precip_mm" in df.columns else 0.0

            if np.isnan(T): T = 25.0
            if np.isnan(H): H = 40.0
            if np.isnan(U_kmh): U_kmh = 10.0
            if np.isnan(P): P = 0.0

            ffmc = ffmc_calc(prev_ffmc, T, H, U_kmh, P)
            dmc  = dmc_calc(prev_dmc,  T, H, P, month)
            dc   = dc_calc(prev_dc,    T, P, month)
            isi  = isi_calc(ffmc, U_kmh)
            bui  = bui_calc(dmc, dc)
            fwi  = fwi_calc(isi, bui)
            lvl  = fwi_level(fwi)

            rows.append({
                "FFMC": ffmc, "DMC": dmc, "DC": dc,
                "ISI": isi, "BUI": bui, "FWI": fwi, "fwi_level": lvl
            })

            # ενημέρωσε το state DF
            if persist_updates:
                if k in idx:
                    fwi_state_df.loc[idx[k], ["FFMC","DMC","DC","last_yyyymmdd"]] = [ffmc, dmc, dc, today_str]
                else:
                    fwi_state_df = pd.concat([fwi_state_df, pd.DataFrame([{
                        "key": k, "FFMC": ffmc, "DMC": dmc, "DC": dc, "last_yyyymmdd": today_str
                    }])], ignore_index=True)

        except Exception:
            rows.append({"FFMC": np.nan,"DMC": np.nan,"DC": np.nan,"ISI": np.nan,"BUI": np.nan,"FWI": np.nan,"fwi_level":""})

    fwi_df = pd.DataFrame(rows).reset_index(drop=True)
    df = df.reset_index(drop=True).join(fwi_df)

    # αν θέλουμε να αποθηκεύουμε το state
    if persist_updates and not fwi_state_df.empty:
        try:
            fwi_state_df.to_csv(state_path, index=False)
            st.sidebar.success(f"Αποθηκεύτηκε state: {state_path}")
        except Exception as e:
            st.sidebar.error(f"Αποτυχία αποθήκευσης state: {e}")

    return df

# -----------------------------------
# ΥΠΟΔΕΙΚΤΕΣ IL / SIP / EI / RISK
# -----------------------------------
def compute_subindices(df, w_IL, w_SIP, w_EI, total_w):
    if "ffwi" not in df.columns:
        df["ffwi"] = [compute_ffwi_simple(t, h, w) for t, h, w in zip(df["temp_c"], df["rh"], df["wind_mps"])]
    df["dryness_adj"] = [adjust_dryness(ff, r) for ff, r in zip(df["ffwi"], df["rain_7d_mm"])]

    gust_boost = (df["wind_gust_mps"] / 20.0).clip(0,1)
    human_near = (
        df["dist_wui_km"].apply(lambda x: expo_score(x, 15.0))*0.5 +
        df["dist_roads_km"].apply(lambda x: expo_score(x, 3.0))*0.3 +
        df["dist_power_km"].apply(lambda x: expo_score(x, 10.0))*0.2
    )
    hist = (pd.to_numeric(df.get("fires5y_per_km2", 0.0), errors="coerce") / 3.0).clip(0,1)
    thunder = (pd.to_numeric(df.get("lightning_24h", 0.0), errors="coerce") / 10.0).clip(0,1)
    wknd = pd.to_numeric(df.get("is_weekend_or_holiday", 0), errors="coerce").fillna(0)*0.2
    ban_reduce = pd.to_numeric(df.get("burn_ban", 0), errors="coerce").fillna(0)*0.3
    dry01 = (df["dryness_adj"]/100.0).clip(0,1)
    wind01 = (df["wind_mps"]/15.0).clip(0,1)
    IL01 = (0.45*dry01 + 0.20*wind01 + 0.10*gust_boost + 0.15*human_near + 0.07*hist + 0.08*thunder + wknd) - ban_reduce
    df["IL"] = (IL01.clip(0,1)*100).round(1)

    if "slope_deg" not in df.columns: df["slope_deg"] = np.nan
    if "aspect_deg" not in df.columns: df["aspect_deg"] = np.nan
    slope_fac = df["slope_deg"].apply(slope_factor)
    align = [aspect_wind_alignment(a, None) for a in df["aspect_deg"]]
    wind_s = (df["wind_mps"]/15.0).clip(0,1)
    dry_s = (df["dryness_adj"]/100.0).clip(0,1)
    if "veg_class" not in df.columns: df["veg_class"] = "unknown"
    fuel01 = [((fuel_factor_from_veg(v)-1.0)/0.3) for v in df["veg_class"]]
    fuel01 = pd.Series(fuel01).clip(0,1)
    SIP01 = (0.35*wind_s + 0.20*dry_s + 0.25*(slope_fac-1.0).clip(0,1) + 0.10*pd.Series(align) + 0.10*fuel01)
    df["SIP"] = (SIP01.clip(0,1)*100).round(1)

    for miss in ["dist_wui_km","dist_roads_km","dist_power_km","dist_critical_km","in_protected"]:
        if miss not in df.columns:
            df[miss] = 0 if miss=="in_protected" else np.nan
    wui = df["dist_wui_km"].apply(lambda x: expo_score(x, 15.0))
    roads = df["dist_roads_km"].apply(lambda x: expo_score(x, 3.0))
    power = df["dist_power_km"].apply(lambda x: expo_score(x, 10.0))
    critical = df["dist_critical_km"].apply(lambda x: expo_score(x, 20.0))
    prot = pd.to_numeric(df.get("in_protected", 0), errors="coerce").fillna(0).clip(0,1)

    w_wui, w_roads, w_power, w_critical, w_prot = 0.40, 0.20, 0.15, 0.20, 0.10
    den = w_wui + w_roads + w_power + w_critical + w_prot
    EI01 = (w_wui*wui + w_roads*roads + w_power*power + w_critical*critical + w_prot*prot) / den
    df["EI"] = (EI01.clip(0,1)*100).round(1)

    df["risk"] = ((w_IL*df["IL"] + w_SIP*df["SIP"] + w_EI*df["EI"]) / total_w).round(1)
    return df

def risk_level(r):
    if pd.isna(r): return "low"
    r = float(r)
    if r < 25: return "low"
    if r < 50: return "medium"
    if r < 75: return "high"
    return "extreme"

def risk_color_by_level(level):
    level = (level or "").lower()
    if level == "low": return "#2DC937"
    if level == "medium": return "#E7B416"
    if level == "high": return "#DB7B2B"
    return "#CC3232"

def risk_color(r):
    return risk_color_by_level(risk_level(r))

# θεματικοί μικρο-χάρτες
def make_station_map(df, value_col, title, unit="", palette="YlOrRd_09", height=320):
    if value_col not in df.columns or df[value_col].dropna().empty:
        return None
    series = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if series.empty:
        return None
    vmin = float(series.quantile(0.05)); vmax = float(series.quantile(0.95))
    if vmin == vmax:
        vmin = float(series.min()); vmax = float(series.max() + 1e-6)
    try:
        cmap = getattr(linear, palette).scale(vmin, vmax)
    except Exception:
        cmap = linear.YlOrRd_09.scale(vmin, vmax)
    m = folium.Map(location=[35.1, 33.3], zoom_start=8, tiles="cartodbpositron")
    for _, r in df.iterrows():
        lat = pd.to_numeric(r.get("lat"), errors="coerce")
        lon = pd.to_numeric(r.get("lon"), errors="coerce")
        val = pd.to_numeric(r.get(value_col), errors="coerce")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(val): continue
        name = r.get("location") or r.get("display_name") or "Station"
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=cmap(val),
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(html=f"<b>{name}</b><br>{title}: <b>{val:.2f}</b> {unit}", max_width=250),
        ).add_to(m)
    cmap.caption = title + (f" ({unit})" if unit else "")
    cmap.add_to(m)
    return m

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("⚙️ Ρυθμίσεις")
autorefresh_min = st.sidebar.number_input("Auto-refresh (λεπτά)", 1, 30, 5, 1)
st.sidebar.write(f"Τελευταίο refresh: **{time.strftime('%Y-%m-%d %H:%M:%S')}**")

with st.sidebar.expander("Βάρη υποδεικτών"):
    w_IL  = st.slider("Βάρος IL (Ignition)", 0.0, 1.0, 0.35, 0.05)
    w_SIP = st.slider("Βάρος SIP (Spread/Intensity)", 0.0, 1.0, 0.45, 0.05)
    w_EI  = st.slider("Βάρος EI (Exposure/Impact)", 0.0, 1.0, 0.20, 0.05)
total_w = w_IL + w_SIP + w_EI
if total_w == 0: total_w = 1.0

dataset_choice = st.sidebar.selectbox("Προβολή", ["DOM stations (weather.csv)", "Grid hotspots (points.csv)"])
top_n = st.sidebar.slider("Top N σημεία", 5, 200, 50, 5)

with st.sidebar.expander("🧠 FWI state", expanded=True):
    persist_fwi = st.checkbox("Αποθήκευση νέων FFMC/DMC/DC στο fwi_state.csv", value=False)
    st.caption("Αν το τικάρεις, το app θα ενημερώσει το fwi_state.csv με τους σημερινούς κώδικες.")
    st.code(FWI_STATE, language="bash")

with st.sidebar.expander("🩺 Διαγνωστικά EI", expanded=False):
    st.caption("Έκδοση EI: WUI=15km, Roads=3km, Power=10km, Critical=20km + Protected.")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data(ttl=60)
def load_points(path):
    if not os.path.exists(path):
        st.warning(f"Λείπει το αρχείο grid: {path}")
        return pd.DataFrame(columns=["location","lat","lon","veg_class"])
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for miss in ["dist_roads_km","dist_wui_km","dist_power_km","dist_critical_km","in_protected"]:
        if miss not in df.columns:
            df[miss] = 0 if miss=="in_protected" else np.nan
    return df

@st.cache_data(ttl=60)
def load_weather(path):
    if not os.path.exists(path):
        st.warning(f"Λείπει το αρχείο καιρού: {path}")
        return pd.DataFrame(columns=["location","lat","lon","temp_c","rh","wind_mps"])
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

points  = load_points(POINTS_CSV)
weather = load_weather(WEATHER_CSV)
fwi_state_df = load_fwi_state(FWI_STATE)

st.title("🔥 CY Fire Risk Dashboard (State-aware FWI)")

# -----------------------------------
# MAIN BRANCHES
# -----------------------------------
if dataset_choice == "DOM stations (weather.csv)":
    need = {
        "temp_c": 25.0, "rh": 40.0, "wind_mps": 2.0,
        "wind_gust_mps": 0.0, "rain_7d_mm": 0.0,
        "slope_deg": np.nan, "aspect_deg": np.nan,
        "dist_wui_km": np.nan, "dist_roads_km": np.nan, "dist_power_km": np.nan, "dist_critical_km": np.nan,
        "in_protected": 0, "fires5y_per_km2": 0.0, "lightning_24h": 0.0,
        "is_weekend_or_holiday": 0, "burn_ban": 0
    }
    weather = ensure_numeric(weather, need)

    # veg για DOM από κοντινό grid
    if {"lat","lon"}.issubset(weather.columns) and len(points)>0 and "veg_class" in points.columns:
        try:
            veg_dom = veg_knn_majority(weather, points, k=5, radius_km=2.0).reset_index(drop=True)
        except Exception:
            P = points[["lat","lon"]].to_numpy(float); VG = points["veg_class"].reset_index(drop=True)
            vals=[]
            for _, r in weather.iterrows():
                lat, lon = nz(r["lat"], np.nan), nz(r["lon"], np.nan)
                if np.isnan(lat) or np.isnan(lon) or len(P)==0: vals.append("unknown")
                else:
                    d2 = (P[:,0]-lat)**2 + (P[:,1]-lon)**2
                    vals.append(VG.iloc[int(np.argmin(d2))])
            veg_dom = pd.Series(vals).reset_index(drop=True)
    else:
        veg_dom = pd.Series(["unknown"]*len(weather))

    dom_df = weather.copy()
    dom_df["veg_class"]   = veg_dom.values
    dom_df["display_name"]= weather["location"] if "location" in weather.columns else pd.Series([f"DOM_{i}" for i in weather.index])

    # Enrichment (gusts & 7d rain)
    if os.path.exists(ENRICH_CSV):
        enr = pd.read_csv(ENRICH_CSV)
        for c in ("lat","lon"):
            if c in enr.columns: enr[c] = pd.to_numeric(enr[c], errors="coerce")
        ren = {}
        if "wind_gust_mps" in enr.columns: ren["wind_gust_mps"] = "wind_gust_mps_enr"
        if "rain_7d_mm"   in enr.columns: ren["rain_7d_mm"]    = "rain_7d_mm_enr"
        if ren: enr = enr.rename(columns=ren)
        dom_df = dom_df.merge(enr, on=["lat","lon"], how="left")
        if "wind_gust_mps_enr" in dom_df.columns:
            base = pd.to_numeric(dom_df.get("wind_gust_mps", np.nan), errors="coerce")
            enr_ = pd.to_numeric(dom_df["wind_gust_mps_enr"], errors="coerce")
            dom_df["wind_gust_mps"] = np.where(base.notna() & (base>0), base, enr_.fillna(0.0))
        if "rain_7d_mm_enr" in dom_df.columns:
            base = pd.to_numeric(dom_df.get("rain_7d_mm", np.nan), errors="coerce")
            enr_ = pd.to_numeric(dom_df["rain_7d_mm_enr"], errors="coerce")
            dom_df["rain_7d_mm"] = np.where(base.notna() & (base>0), base, enr_.fillna(0.0))
    dom_df["rain_7d_mm"]   = pd.to_numeric(dom_df.get("rain_7d_mm", 0.0), errors="coerce").fillna(0.0)
    dom_df["wind_gust_mps"]= pd.to_numeric(dom_df.get("wind_gust_mps", 0.0), errors="coerce").fillna(0.0)

    # Κληρονόμηση exposure από κοντινό grid
    exposure_cols = ["dist_wui_km","dist_roads_km","dist_power_km","dist_critical_km","in_protected"]
    for miss in exposure_cols:
        if miss not in points.columns:
            points[miss] = 0 if miss=="in_protected" else np.nan
    if len(points) > 0 and {"lat","lon"}.issubset(dom_df.columns):
        P = points[["lat","lon"]].to_numpy(float)
        nearest_idx = []
        for _, r in dom_df.iterrows():
            lat = pd.to_numeric(r.get("lat"), errors="coerce")
            lon = pd.to_numeric(r.get("lon"), errors="coerce")
            if np.isnan(lat) or np.isnan(lon) or len(P)==0: nearest_idx.append(None)
            else:
                d2 = (P[:,0]-lat)**2 + (P[:,1]-lon)**2
                nearest_idx.append(int(np.argmin(d2)))
        for col in exposure_cols:
            vals = []
            for j in nearest_idx:
                if j is None: vals.append(0 if col=="in_protected" else np.nan)
                else:         vals.append(points.iloc[j].get(col, (0 if col=="in_protected" else np.nan)))
            if col == "in_protected":
                dom_df[col] = pd.Series(vals).fillna(0).astype(int)
            else:
                dom_df[col] = pd.to_numeric(pd.Series(vals), errors="coerce")
    else:
        for col in exposure_cols:
            dom_df[col] = 0 if col=="in_protected" else np.nan
    for c in ["dist_wui_km","dist_roads_km","dist_power_km","dist_critical_km"]:
        dom_df[c] = pd.to_numeric(dom_df.get(c, np.nan), errors="coerce")
    dom_df["in_protected"] = pd.to_numeric(dom_df.get("in_protected", 0), errors="coerce").fillna(0).clip(0,1).astype(int)

    # ---- STATE-AWARE FWI εδώ ----
    dom_df = add_fwi_columns_stateful(dom_df, fwi_state_df, persist_updates=persist_fwi, state_path=FWI_STATE)

    # Υπολογισμοί Risk (IL/SIP/EI)
    dom_df = compute_subindices(dom_df, w_IL, w_SIP, w_EI, total_w)
    dom_df["risk_level"] = dom_df["risk"].apply(risk_level)
    df_view = dom_df

else:
    # GRID mode
    need = {
        "temp_c": 25.0, "rh": 40.0, "wind_mps": 2.0,
        "wind_gust_mps": 0.0, "rain_7d_mm": 0.0,
        "slope_deg": np.nan, "aspect_deg": np.nan,
        "dist_wui_km": np.nan, "dist_roads_km": np.nan, "dist_power_km": np.nan, "dist_critical_km": np.nan,
        "in_protected": 0, "fires5y_per_km2": 0.0, "lightning_24h": 0.0,
        "is_weekend_or_holiday": 0, "burn_ban": 0
    }
    points = ensure_numeric(points, need)
    for miss in ["dist_roads_km","dist_wui_km","dist_power_km","dist_critical_km","in_protected"]:
        if miss not in points.columns:
            points[miss] = 0 if miss=="in_protected" else np.nan
    grid_df = points.rename(columns={"location":"display_name"}).copy()

    # optional: κληρονόμησε gusts/rain από enrichment κοντινού DOM
    if os.path.exists(ENRICH_CSV) and len(grid_df)>0:
        enr = pd.read_csv(ENRICH_CSV)
        if {"lat","lon"}.issubset(enr.columns):
            P = enr[["lat","lon"]].to_numpy(float)
            g_gust, g_rain = [], []
            for _, r in grid_df.iterrows():
                lat, lon = float(r["lat"]), float(r["lon"])
                if np.isnan(lat) or np.isnan(lon) or len(P)==0:
                    g_gust.append(np.nan); g_rain.append(np.nan)
                else:
                    d2 = (P[:,0]-lat)**2 + (P[:,1]-lon)**2
                    j = int(np.argmin(d2))
                    g_gust.append(enr.iloc[j].get("wind_gust_mps", np.nan))
                    g_rain.append(enr.iloc[j].get("rain_7d_mm", np.nan))
            grid_df["wind_gust_mps"] = pd.Series(g_gust).fillna(grid_df.get("wind_gust_mps", 0.0)).astype(float)
            grid_df["rain_7d_mm"]   = pd.Series(g_rain).fillna(grid_df.get("rain_7d_mm", 0.0)).astype(float)

    # STATE-AWARE FWI και στο grid (προαιρετικά χρήσιμο)
    grid_df = add_fwi_columns_stateful(grid_df, fwi_state_df, persist_updates=False, state_path=FWI_STATE)

    grid_df = compute_subindices(grid_df, w_IL, w_SIP, w_EI, total_w)
    grid_df["risk_level"] = grid_df["risk"].apply(risk_level)
    df_view = grid_df

# -----------------------------------
# ΠΙΝΑΚΑΣ & KPIs
# -----------------------------------
df_sorted = df_view.sort_values("risk", ascending=False).reset_index(drop=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Σημεία", f"{len(df_sorted)}")
c2.metric("Μέγιστος κίνδυνος", f"{df_sorted['risk'].iloc[0]:.1f}" if len(df_sorted)>0 else "—")
c3.metric("Μέσος κίνδυνος", f"{df_sorted['risk'].mean():.1f}" if len(df_sorted)>0 else "—")
c4.metric("Διάμεσος κίνδυνος", f"{df_sorted['risk'].median():.1f}" if len(df_sorted)>0 else "—")

display_cols = [
    "display_name","veg_class",
    "IL","SIP","EI","risk_level","risk",
    "FWI","fwi_level","FFMC","DMC","DC","ISI","BUI",
    "temp_c","rh","wind_mps","wind_gust_mps","rain_7d_mm"
]
display_cols = [c for c in display_cols if c in df_sorted.columns]
st.dataframe(df_sorted[display_cols].head(top_n), use_container_width=True)

# -----------------------------------
# ΚΥΡΙΟΣ ΧΑΡΤΗΣ (RISK)
# -----------------------------------
map_center = [35.1, 33.3]
m = folium.Map(location=map_center, zoom_start=8, tiles="cartodbpositron")

for _, row in df_sorted.head(top_n).iterrows():
    lat, lon = nz(row.get("lat"), np.nan), nz(row.get("lon"), np.nan)
    if np.isnan(lat) or np.isnan(lon):
        continue
    name  = str(row.get("display_name","—"))
    level = row.get("risk_level")
    popup = folium.Popup(html=(
        f"<b>{name}</b><br>"
        f"veg: {row.get('veg_class','?')}<br>"
        f"IL: {row.get('IL','?')} | SIP: {row.get('SIP','?')} | EI: {row.get('EI','?')}<br>"
        f"Risk: <b>{row.get('risk','?')}</b> ({level})<br>"
        f"FWI: <b>{row.get('FWI','?')}</b> ({row.get('fwi_level','')})<br>"
        f"T/RH/W: {row.get('temp_c','?')}/{row.get('rh','?')}/{row.get('wind_mps','?')}<br>"
        f"Gust/7dRain: {row.get('wind_gust_mps','?')}/{row.get('rain_7d_mm','?')} mm<br>"
    ), max_width=300)
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=risk_color(row.get("risk")),
        fill=True, fill_opacity=0.9,
        popup=popup
    ).add_to(m)

st_folium(m, height=520, use_container_width=True)

# -----------------------------------
# Θεματικοί χάρτες DOM (4 mini-maps)
# -----------------------------------
st.markdown("### 🌡️/💧/🌬️/☔ Ζωντανά πεδία από σταθμούς Τμ. Μετεωρολογίας")
dom_map_df = weather.copy()
if not dom_map_df.empty:
    for c in ["temp_c","rh","wind_mps","wind_gust_mps","precip_mm","rain_1h_mm","rain_24h_mm","rain_7d_mm","lat","lon"]:
        if c in dom_map_df.columns:
            dom_map_df[c] = pd.to_numeric(dom_map_df[c], errors="coerce")

    if os.path.exists(ENRICH_CSV):
        enr = pd.read_csv(ENRICH_CSV)
        for c in ("lat","lon"):
            if c in enr.columns: enr[c] = pd.to_numeric(enr[c], errors="coerce")
        rename_map = {}
        if "wind_gust_mps" in enr.columns: rename_map["wind_gust_mps"] = "wind_gust_mps_enr"
        if "rain_7d_mm" in enr.columns:    rename_map["rain_7d_mm"]    = "rain_7d_mm_enr"
        if rename_map: enr = enr.rename(columns=rename_map)
        dom_map_df = dom_map_df.merge(enr, on=["lat","lon"], how="left")
        if "wind_gust_mps_enr" in dom_map_df.columns:
            base = pd.to_numeric(dom_map_df.get("wind_gust_mps", np.nan), errors="coerce")
            enr_ = pd.to_numeric(dom_map_df["wind_gust_mps_enr"], errors="coerce")
            dom_map_df["wind_gust_mps"] = np.where(base.notna() & (base>0), base, enr_)
        if "rain_7d_mm_enr" in dom_map_df.columns:
            base = pd.to_numeric(dom_map_df.get("rain_7d_mm", np.nan), errors="coerce")
            enr_ = pd.to_numeric(dom_map_df["rain_7d_mm_enr"], errors="coerce")
            dom_map_df["rain_7d_mm"] = np.where(base.notna() & (base>0), base, enr_)

    precip_col = None
    for candidate in ["precip_mm", "rain_1h_mm", "rain_24h_mm", "rain_7d_mm"]:
        if candidate in dom_map_df.columns and dom_map_df[candidate].notna().any():
            precip_col = candidate
            break

    cA, cB = st.columns(2)
    cC, cD = st.columns(2)

    tmap = make_station_map(dom_map_df, "temp_c", "Θερμοκρασία", "°C", palette="YlOrRd_09")
    rmap = make_station_map(dom_map_df, "rh", "Σχετική Υγρασία", "%", palette="Blues_09")
    wmap = make_station_map(dom_map_df, "wind_mps", "Άνεμος (ΜΟ)", "m/s", palette="PuBu_09")

    if tmap is not None:
        with cA:
            st_folium(tmap, height=320, use_container_width=True)
    else:
        with cA:
            st.info("Δεν υπάρχουν διαθέσιμες θερμοκρασίες.")

    if rmap is not None:
        with cB:
            st_folium(rmap, height=320, use_container_width=True)
    else:
        with cB:
            st.info("Δεν υπάρχουν διαθέσιμες τιμές υγρασίας.")

    if wmap is not None:
        with cC:
            st_folium(wmap, height=320, use_container_width=True)
    else:
        with cC:
            st.info("Δεν υπάρχουν διαθέσιμες τιμές ανέμου.")

    if precip_col:
        pmap = make_station_map(dom_map_df, precip_col, "Βροχόπτωση", "mm", palette="GnBu_09")
        if pmap is not None:
            with cD:
                st_folium(pmap, height=320, use_container_width=True)
        else:
            with cD:
                st.info("Δεν υπάρχουν διαθέσιμες τιμές βροχής.")
    else:
        with cD:
            st.info("Δεν βρέθηκε στήλη βροχόπτωσης στους σταθμούς.")
else:
    st.info("Δεν βρέθηκαν δεδομένα σταθμών (weather.csv) για θεματικούς χάρτες.")

# -----------------------------------
# AUTO REFRESH
# -----------------------------------
st.sidebar.markdown("---")
st.sidebar.info("Η σελίδα θα ανανεώνεται αυτόματα σύμφωνα με το διάστημα που επέλεξες.")
st_autorefresh_ms = int(autorefresh_min * 60_000)
if st.button("♻️ Χειροκίνητο refresh"):
    st.rerun()
st.markdown(
    f"""
    <script>
      setTimeout(function() {{
        window.location.reload();
      }}, {st_autorefresh_ms});
    </script>
    """,
    unsafe_allow_html=True
)
