#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import argparse
import pandas as pd
from datetime import datetime
from fwi import Weather, Codes, step_fwi

DEFAULTS = Codes(FFMC=85.0, DMC=6.0, DC=15.0)

def key_from_row(r):
    return f"{float(r['lat']):.5f},{float(r['lon']):.5f}"

def today_triplet(row):
    # Weather expects wind in km/h. Έχεις wind_mps -> km/h:
    U_kmh = float(row.get("wind_mps", 0.0)) * 3.6
    T = float(row.get("temp_c", 25.0))
    H = float(row.get("rh", 40.0))
    P = float(row.get("precip_mm", 0.0) or 0.0)  # αν δεν έχεις 24h βροχή, βάλε 0
    month = datetime.now().month
    return Weather(T=T, H=H, U=U_kmh, P=P, month=month)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="cy_weather_sample.csv")
    ap.add_argument("--state", dest="state_csv", default="fwi_state.csv")
    ap.add_argument("--out", dest="out_csv", default="fwi_today.csv", help="FWI ανά σταθμό (προαιρετικό)")
    args = ap.parse_args()

    dom = pd.read_csv(args.in_csv)
    if not {"lat","lon","temp_c","rh","wind_mps"}.issubset(dom.columns):
        sys.exit("[ERROR] Input πρέπει να έχει lat,lon,temp_c,rh,wind_mps")

    # φόρτωσε state
    if os.path.exists(args.state_csv):
        state = pd.read_csv(args.state_csv)
    else:
        state = pd.DataFrame(columns=["key","FFMC","DMC","DC","last_yyyymmdd"])

    idx = {k:i for i,k in enumerate(state.get("key", []))}
    out_rows = []
    today_str = datetime.now().strftime("%Y%m%d")

    for _, r in dom.iterrows():
        try:
            k = key_from_row(r)
            if k in idx:
                srow = state.loc[idx[k]]
                prev = Codes(
                    FFMC=float(srow["FFMC"]),
                    DMC=float(srow["DMC"]),
                    DC=float(srow["DC"]),
                )
            else:
                prev = DEFAULTS

            w = today_triplet(r)
            new_codes, indices = step_fwi(prev, w)

            # γράψε στο output
            out_rows.append({
                "key": k,
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "FFMC": new_codes.FFMC,
                "DMC": new_codes.DMC,
                "DC": new_codes.DC,
                "ISI": indices["ISI"],
                "BUI": indices["BUI"],
                "FWI": indices["FWI"],
            })

            # ενημέρωσε state (πάντα με τα νέα codes) + ημερομηνία
            if k in idx:
                state.loc[idx[k], ["FFMC","DMC","DC","last_yyyymmdd"]] = [new_codes.FFMC, new_codes.DMC, new_codes.DC, today_str]
            else:
                state = pd.concat([state, pd.DataFrame([{
                    "key": k, "FFMC": new_codes.FFMC, "DMC": new_codes.DMC, "DC": new_codes.DC, "last_yyyymmdd": today_str
                }])], ignore_index=True)
        except Exception:
            continue

    if out_rows:
        pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
    state.to_csv(args.state_csv, index=False)
    print(f"[OK] Updated {len(out_rows)} stations → {args.out_csv} & {args.state_csv}")

if __name__ == "__main__":
    main()
