#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Διαβάζει cy_weather_sample.csv (location,lat,lon,...) και φτιάχνει dom_enrich.csv
με: lat, lon, rain_7d_mm, wind_gust_mps

- rain_7d_mm: άθροισμα βροχής τελευταίων 7 ημερών (Open-Meteo daily.precipitation_sum).
- wind_gust_mps: μέγιστη ριπή 10m των τελευταίων 6 ωρών (Open-Meteo hourly.wind_gusts_10m), σε m/s.

Απαιτήσεις: requests, pandas
"""

import argparse
import sys
import time
import math
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API_BASE = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Europe/Nicosia"  # αυτόματο local timezone
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"cy-fire-dashboard/1.0"})

def fetch_station_enrich(lat, lon):
    """Επιστρέφει (rain_7d_mm, wind_gust_mps) ή (None, None) αν κάτι αποτύχει."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "hourly": "wind_gusts_10m",
        "past_days": 7,
        "timezone": TIMEZONE,
        "wind_speed_unit": "ms",  # m/s
    }
    try:
        resp = SESSION.get(API_BASE, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.warning(f"Open-Meteo error for {lat},{lon}: {e}")
        return (None, None)

    # 7-day rain sum
    rain_7d_mm = None
    try:
        daily = data.get("daily", {})
        vals = daily.get("precipitation_sum") or []
        if vals:
            # τα τελευταία 7 στοιχεία είναι ήδη των τελευταίων 7 ημερών
            rain_7d_mm = float(pd.Series(vals).sum())
    except Exception:
        pass

    # max gust of last 6 hours
    wind_gust_mps = None
    try:
        hourly = data.get("hourly", {})
        times = hourly.get("time") or []
        gusts = hourly.get("wind_gusts_10m") or []
        if times and gusts and len(times) == len(gusts):
            # κράτα τις τελευταίες 6 ώρες
            # τα times είναι ISO strings στην TIMEZONE
            now = datetime.now(timezone.utc).astimezone().replace(minute=0, second=0, microsecond=0)
            # βρες τα τελευταία 6 timestamps
            keep_from = now - timedelta(hours=6)
            rows = []
            for t_str, val in zip(times, gusts):
                try:
                    t = datetime.fromisoformat(t_str)
                except Exception:
                    # Open-Meteo timestamps είναι π.χ. "2025-08-28T12:00"
                    # χωρίς offset -> θεωρούμε local TZ της απόκρισης
                    try:
                        t = datetime.strptime(t_str, "%Y-%m-%dT%H:%M")
                    except Exception:
                        continue
                if isinstance(val, (int, float)) and not math.isnan(val):
                    rows.append((t, float(val)))
            if rows:
                rows = [v for (t,v) in rows if t >= keep_from]
                if rows:
                    wind_gust_mps = max(rows)
    except Exception:
        pass

    return (rain_7d_mm, wind_gust_mps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_csv", required=True, help="π.χ. cy_weather_sample.csv")
    parser.add_argument("--out", dest="out_csv", required=True, help="π.χ. dom_enrich.csv")
    parser.add_argument("--sleep", type=float, default=0.2, help="καθυστ. μεταξύ requests (sec)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    df = pd.read_csv(args.in_csv)
    if not {"lat","lon"}.issubset(df.columns):
        sys.exit("[ERROR] Το input CSV πρέπει να έχει στήλες lat, lon")

    # Προετοιμασία εξόδου
    out_rows = []
    for i, r in df.iterrows():
        lat = pd.to_numeric(r.get("lat"), errors="coerce")
        lon = pd.to_numeric(r.get("lon"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        rain7, gust = fetch_station_enrich(lat, lon)
        out_rows.append({
            "lat": float(lat),
            "lon": float(lon),
            "rain_7d_mm": float(rain7) if rain7 is not None else 0.0,
            "wind_gust_mps": float(gust) if gust is not None else 0.0,
        })
        time.sleep(args.sleep)  # φιλικό στο API

    if not out_rows:
        sys.exit("[ERROR] Δεν παρήχθησαν γραμμές — ίσως πρόβλημα στο API ή στο input.")

    out_df = pd.DataFrame(out_rows).drop_duplicates(subset=["lat","lon"])
    out_df.to_csv(args.out_csv, index=False)
    logging.info(f"Έγραψα {len(out_df)} γραμμές -> {args.out_csv}")

if __name__ == "__main__":
    main()
