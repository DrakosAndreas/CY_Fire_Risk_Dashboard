
# cy_fire_risk.py
# Minimal prototype for Cyprus wildfire danger hot-spotting and staffing suggestions.
# Inputs: two CSV files
#   1) cy_points_sample.csv: location,lat,lon,veg_class
#   2) cy_weather_sample.csv: location,temp_c,rh,wind_mps
# Output: cy_fire_risk_today.csv with risk scores and resource suggestions.
import math, sys
import pandas as pd
from pathlib import Path

VEG_FACTORS = {
    "pine_forest": 1.30, "coniferous_forest": 1.30, "mixed_forest": 1.20,
    "sclerophyllous_shrubland": 1.25, "shrubland": 1.20, "transitional_woodland_shrub": 1.20,
    "grassland": 1.10, "pastures": 1.10, "agriculture": 1.00, "olive_groves": 1.05, "vineyards": 1.05,
    "urban": 0.60, "bare": 0.70, "water": 0.20,
}

def c_to_f(tc): return tc * 9/5 + 32
def mps_to_mph(u): return u * 2.2369362920544

def equilibrium_moisture_content(rh, temp_c):
    tf = c_to_f(temp_c); H = rh
    if H < 10:
        emc = 0.03229 + 0.281073 * H - 0.000578 * H * tf
    elif H < 50:
        emc = 2.22749 + 0.160107 * H - 0.01478 * tf
    else:
        emc = 21.0606 + 0.005565 * (H ** 2) - 0.00035 * H * tf - 0.483199 * H
    return max(emc, 0.0)

def fosberg_ffwi(temp_c, rh, wind_mps):
    emc = equilibrium_moisture_content(rh, temp_c)
    x = emc / 30.0
    eta = 1 - 2 * x + 1.5 * (x ** 2) - 0.5 * (x ** 3)
    eta = max(0.0, min(eta, 1.0))
    U_mph = mps_to_mph(wind_mps)
    ffwi = (eta * (1 + U_mph ** 2) ** 0.5) / 0.3002
    return max(0.0, min(ffwi, 100.0))

def vegetation_factor(veg): return VEG_FACTORS.get(veg, 1.0)

def combined_risk_score(temp_c, rh, wind_mps, veg):
    return max(0.0, min(fosberg_ffwi(temp_c, rh, wind_mps) * vegetation_factor(veg), 100.0))

def resource_recommendations(risk):
    if risk >= 85: return {"engines": 3, "personnel": 14}
    if risk >= 70: return {"engines": 2, "personnel": 10}
    if risk >= 55: return {"engines": 1, "personnel": 6}
    if risk >= 40: return {"engines": 1, "personnel": 4}
    return {"engines": 0, "personnel": 0}

def main(points_csv, weather_csv, out_csv):
    points = pd.read_csv(points_csv)
    weather = pd.read_csv(weather_csv)
    df = points.merge(weather, on="location", how="inner")
    df["ffwi"] = [fosberg_ffwi(t, h, w) for t, h, w in zip(df["temp_c"], df["rh"], df["wind_mps"])]
    df["veg_factor"] = [vegetation_factor(v) for v in df["veg_class"]]
    df["risk"] = [combined_risk_score(t, h, w, v) for t, h, w, v in zip(df["temp_c"], df["rh"], df["wind_mps"], df["veg_class"])]
    recs = [resource_recommendations(r) for r in df["risk"]]
    df["engines"] = [r["engines"] for r in recs]
    df["personnel"] = [r["personnel"] for r in recs]
    df.sort_values("risk", ascending=False).to_csv(out_csv, index=False)
    print("Wrote {out_csv}")

if __name__ == "__main__":
    base = Path(".")
    pts = sys.argv[1] if len(sys.argv) > 1 else base / "cy_points_sample.csv"
    wx = sys.argv[2] if len(sys.argv) > 2 else base / "cy_weather_sample.csv"
    out = sys.argv[3] if len(sys.argv) > 3 else base / "cy_fire_risk_today.csv"
    main(pts, wx, out)
