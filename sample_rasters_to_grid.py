#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import rasterio

def sample_raster(raster_path: Path, lats: pd.Series, lons: pd.Series) -> pd.Series:
    with rasterio.open(raster_path) as src:
        coords = list(zip(lons.to_list(), lats.to_list()))  # (x=lon, y=lat) σε EPSG:4326
        vals = []
        for v in src.sample(coords):
            # v είναι array για τα bands, παίρνουμε το πρώτο
            x = v[0]
            if x is None:
                vals.append(None)
            else:
                # κάποιοι οδηγοί γυρίζουν masked arrays
                try:
                    vals.append(float(x))
                except Exception:
                    vals.append(None)
        return pd.Series(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="cy_points_grid_2km.csv", help="Grid CSV (location,lat,lon,veg_class)")
    ap.add_argument("--slope_tif", required=True, help="GeoTIFF με slope (deg)")
    ap.add_argument("--aspect_tif", required=True, help="GeoTIFF με aspect (deg)")
    ap.add_argument("--out_csv", default="cy_points_grid_2km.csv", help="Που θα σωθεί το ενημερωμένο CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    for c in ["lat","lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)

    df["slope_deg"]  = sample_raster(Path(args.slope_tif),  df["lat"], df["lon"]).round(2)
    df["aspect_deg"] = sample_raster(Path(args.aspect_tif), df["lat"], df["lon"]).round(1)

    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Έγραψα {args.out_csv} με πεδία slope_deg & aspect_deg")

if __name__ == "__main__":
    main()

