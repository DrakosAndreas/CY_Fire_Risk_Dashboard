#!/usr/bin/env python3
"""
fetch_weather.py
----------------
Αντλεί καιρικά δεδομένα για 50 σημεία (σταθμούς) στην Κύπρο από το Open-Meteo
και παράγει CSV με στήλες:
  location,lat,lon,temp_c,rh,wind_mps

• Δεν απαιτεί API key.
• Χρησιμοποιεί current weather (temperature, windspeed) + hourly RH.
• Αν θες δικούς σου σταθμούς, δώσε --stations_csv με στήλες: name,lat,lon

Χρήση:
  python fetch_weather.py --out cy_weather_sample.csv
  # ή με δικούς σου σταθμούς:
  python fetch_weather.py --out cy_weather_sample.csv --stations_csv my_stations.csv
"""

import argparse
import csv
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import requests

# ----------------------- ΡΥΘΜΙΣΕΙΣ -----------------------

TIMEZONE = "Europe/Nicosia"   # Χρονική ζώνη για Open-Meteo
HTTP_TIMEOUT = 15             # Timeout σε sec
SLEEP_BETWEEN_CALLS = 0.15    # Μικρό διάλειμμα για ευγένεια

def kmh_to_mps(v: float) -> float:
    return float(v) / 3.6

def default_50_stations() -> List[Tuple[str, float, float]]:
    """
    Δημιουργεί 50 σημεία περίπου ομοιόμορφα σε όλη την Κύπρο.
    Bounding box ~ (lat 34.45–35.75, lon 32.25–34.85).
    10 x 5 grid = 50 σημεία.
    """
    stations: List[Tuple[str, float, float]] = []
    lat_min, lat_max = 34.45, 35.75
    lon_min, lon_max = 32.25, 34.85

    rows, cols = 5, 10  # 5*10 = 50
    for i in range(rows):
        lat = lat_min + (i + 0.5) * (lat_max - lat_min) / rows
        for j in range(cols):
            lon = lon_min + (j + 0.5) * (lon_max - lon_min) / cols
            name = f"Station_{i*cols + j + 1:02d}"
            stations.append((name, round(lat, 5), round(lon, 5)))
    return stations

def load_stations_from_csv(path: Path) -> List[Tuple[str, float, float]]:
    """Διαβάζει custom σταθμούς από CSV με στήλες: name,lat,lon"""
    out: List[Tuple[str, float, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("name") or row.get("Name")
            lat = float(row.get("lat") or row.get("Lat"))
            lon = float(row.get("lon") or row.get("Lon"))
            out.append((str(name), lat, lon))
    if not out:
        raise ValueError("Δεν βρέθηκαν σταθμοί στο CSV.")
    return out

def fetch_one_station(name: str, lat: float, lon: float) -> Optional[dict]:
    """Κάνει call στο Open-Meteo και επιστρέφει dict με μετρήσεις"""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current_weather=true"
        "&hourly=relative_humidity_2m"
        f"&timezone={TIMEZONE.replace('/', '%2F')}"
    )
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        cw = data.get("current_weather") or {}
        temp_c = cw.get("temperature")
        wind_kmh = cw.get("windspeed")
        wind_mps = kmh_to_mps(wind_kmh) if wind_kmh is not None else None

        rh = None
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        rhs = hourly.get("relative_humidity_2m") or []
        cur_t = cw.get("time")
        if cur_t and times and rhs and len(times) == len(rhs):
            try:
                idx = times.index(cur_t)
                rh = rhs[idx]
            except ValueError:
                pass
        if rh is None and rhs:
            rh = rhs[-1]

        if temp_c is None or rh is None or wind_mps is None:
            return None

        return {
            "location": name,
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "temp_c": round(float(temp_c), 1),
            "rh": round(float(rh), 1),
            "wind_mps": round(float(wind_mps), 2),
        }
    except Exception as e:
        sys.stderr.write(f"[WARN] {name} ({lat},{lon}): {e}\n")
        return None

def atomic_write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8") as tmp:
        w = csv.DictWriter(tmp, fieldnames=["location", "lat", "lon", "temp_c", "rh", "wind_mps"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        temp_name = tmp.name
    shutil.move(temp_name, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cy_weather_sample.csv", help="Διαδρομή εξόδου CSV")
    ap.add_argument("--stations_csv", default=None, help="Προαιρετικό CSV με στήλες name,lat,lon")
    args = ap.parse_args()

    if args.stations_csv:
        stations = load_stations_from_csv(Path(args.stations_csv))
    else:
        stations = default_50_stations()

    rows: List[dict] = []
    for (name, lat, lon) in stations:
        rec = fetch_one_station(name, lat, lon)
        if rec:
            rows.append(rec)
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not rows:
        raise SystemExit("Δεν συγκεντρώθηκαν δεδομένα από κανέναν σταθμό.")

    atomic_write_csv(rows, Path(args.out))
    print(f"[OK] Γράφτηκαν {len(rows)} εγγραφές στο {args.out}")

if __name__ == "__main__":
    main()

