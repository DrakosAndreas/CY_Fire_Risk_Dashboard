#!/usr/bin/env python3
import sys, datetime as dt, pandas as pd, requests, time

IN_CSV  = sys.argv[1] if len(sys.argv)>1 else "cy_weather_sample.csv"
OUT_CSV = sys.argv[2] if len(sys.argv)>2 else "rain7d.csv"

def fetch_sum(lat, lon, start, end):
    url = ("https://archive-api.open-meteo.com/v1/era5"
           f"?latitude={lat}&longitude={lon}"
           f"&start_date={start}&end_date={end}"
           "&daily=precipitation_sum&timezone=auto")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    vals = js.get("daily", {}).get("precipitation_sum", [])
    return round(float(sum([v for v in vals if v is not None])), 2)

def main():
    today = dt.date.today()
    start = (today - dt.timedelta(days=6)).isoformat()  # inclusive 7 days window
    end   = today.isoformat()

    w = pd.read_csv(IN_CSV)
    cols = [c.strip() for c in w.columns]
    w.columns = cols
    if not {"lat","lon"}.issubset(w.columns):
        print("[ERROR] weather CSV needs lat/lon", file=sys.stderr); sys.exit(1)

    rows=[]
    seen=set()
    for _, r in w.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        key = (round(lat,4), round(lon,4))
        if key in seen: continue
        seen.add(key)
        try:
            s = fetch_sum(lat, lon, start, end)
        except Exception as e:
            print("warn:", e, "retry in 2s", file=sys.stderr)
            time.sleep(2)
            try:
                s = fetch_sum(lat, lon, start, end)
            except Exception as e2:
                print("err:", e2, "set 0", file=sys.stderr)
                s = 0.0
        rows.append({"lat": key[0], "lon": key[1], "rain_7d_mm": s})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("[OK] wrote", OUT_CSV)

if __name__ == "__main__":
    main()

