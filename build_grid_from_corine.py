#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_grid_from_corine.py
Φτιάχνει 2km grid για Κύπρο και αποδίδει veg_class από CORINE Land Cover (CLC).

Υποστηρίζει:
- Raster CLC (GeoTIFF): δώσε --clc_raster
- Vector CLC (GPKG/SHAPE): δώσε --clc_vector και --code_field (π.χ. code_18)

Έξοδος:
- cy_points_grid_2km.csv με στήλες: location,lat,lon,veg_class
- (προαιρετικό) cy_points_grid_2km.geojson για έλεγχο (με --geojson)

Χρήση (Raster):
  python build_grid_from_corine.py --clc_raster data/CLC2018_CY.tif

Χρήση (Vector):
  python build_grid_from_corine.py --clc_vector data/CLC2018_CY.gpkg --code_field code_18
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

# Raster / Vector deps
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from rasterio.sample import sample_gen
import geopandas as gpd
from shapely.geometry import Point, box

# ----- Cyprus bbox in WGS84 -----
LAT_MIN, LAT_MAX = 34.45, 35.75
LON_MIN, LON_MAX = 32.25, 34.85

# Grid spacing ~2 km in degrees (προσέγγιση: 0.02° ~ 2.2 km σε πλάτος Κύπρου)
# Για πιο “ακριβές” 2 km σε μέτρα, θα χρειαζόταν προβολή σε UTM. Για απλότητα κρατάμε degrees.
DLAT = 0.02
DLON = 0.0244  # ~2.2 km στην Κύπρο (cos φ συντελεστής)

# ----- Mapping CLC Level-3 code -> veg_class -----
# Προσαρμοσμένο στις κατηγορίες του app
def clc_to_veg(clc_code: int) -> str:
    if clc_code is None:
        return "agriculture"

    # Τεχνητές επιφάνειες 111–142 → urban
    if 111 <= clc_code <= 142:
        return "urban"

    # Καλλιέργειες
    if clc_code in (221,):                    # Vineyards
        return "vineyards"
    if clc_code in (223,):                    # Olive groves
        return "olive_groves"
    if clc_code in (231,):                    # Pastures
        return "pastures"
    if clc_code in (321,):                    # Natural grasslands
        return "grassland"
    if clc_code in (324,):                    # Transitional woodland-shrub
        return "transitional_woodland_shrub"
    if clc_code in (323,):                    # Sclerophyllous vegetation
        return "sclerophyllous_shrubland"

    # Δάση
    if clc_code in (312,):                    # Coniferous forest
        return "coniferous_forest"
    if clc_code in (313,):                    # Mixed forest
        return "mixed_forest"
    if clc_code in (311,):                    # Broad-leaved forest -> mixed_forest (πλησιέστερη συμπεριφορά)
        return "mixed_forest"

    # Υπόλοιπες γεωργικές/μεικτές
    if clc_code in (211,212,213,241,242,244):
        return "agriculture"
    if clc_code in (243,):                    # Land principally occupied by agriculture, with natural vegetation
        # συχνά μωσαϊκό -> shrubland “συντηρητικά”
        return "shrubland"

    # Γυμνές επιφάνειες
    if clc_code in (331,332,333,334,422):
        return "bare"

    # Νερά
    if 511 <= clc_code <= 523:
        return "water"
    if clc_code in (421,423):
        return "water"

    # Προεπιλογή
    return "agriculture"

def build_grid():
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, DLAT)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, DLON)
    cells = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            cells.append((f"Cell_{i}_{j}", float(lat), float(lon)))
    return pd.DataFrame(cells, columns=["location","lat","lon"])

# ----- Raster sampling -----
def sample_raster(raster_path: Path, pts_df: pd.DataFrame) -> pd.Series:
    with rasterio.open(raster_path) as src:
        # Αν δεν είναι σε WGS84, κάνουμε on-the-fly reprojection με VRT
        if src.crs is None:
            raise RuntimeError("Το raster CLC δεν έχει CRS. Χρειάζεται ορισμένο projection.")
        if src.crs.to_string() != "EPSG:4326":
            vrt = WarpedVRT(src, crs="EPSG:4326", resampling=Resampling.nearest)
            reader = vrt
        else:
            reader = src

        coords = list(zip(pts_df["lon"].tolist(), pts_df["lat"].tolist()))
        values = []
        for val in reader.sample(coords):
            # παίρνουμε το πρώτο band
            v = val[0]
            if v is None or (hasattr(v, "mask") and getattr(v, "mask", False)):
                values.append(None)
            else:
                values.append(int(v))
    return pd.Series(values, index=pts_df.index, name="clc_code")

# ----- Vector spatial join -----
def sample_vector(vector_path: Path, code_field: str, pts_df: pd.DataFrame) -> pd.Series:
    gdf_pts = gpd.GeoDataFrame(
        pts_df.copy(),
        geometry=[Point(xy) for xy in zip(pts_df["lon"], pts_df["lat"])],
        crs="EPSG:4326"
    )
    gdf_poly = gpd.read_file(vector_path)
    if gdf_poly.crs is None:
        raise RuntimeError("Το vector CLC δεν έχει CRS.")
    if gdf_poly.crs.to_string() != "EPSG:4326":
        gdf_poly = gdf_poly.to_crs("EPSG:4326")

    # γρήγορη χωρική σύζευξη (requires rtree or pygeos)
    joined = gpd.sjoin(gdf_pts, gdf_poly[[code_field, "geometry"]], how="left", predicate="within")
    codes = joined[code_field]
    # Μετά το join, επανέλαβε τη σειρά στα αρχικά indices
    codes = codes.reindex(pts_df.index)
    # άδεια -> None
    codes = codes.apply(lambda x: int(x) if pd.notna(x) else None)
    return codes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clc_raster", default=None, help="Διαδρομή σε CLC GeoTIFF (100m)")
    ap.add_argument("--clc_vector", default=None, help="Διαδρομή σε CLC vector (GPKG/SHAPE)")
    ap.add_argument("--code_field", default="code_18", help="Όνομα πεδίου κωδικού CLC στο vector (π.χ. code_18)")
    ap.add_argument("--out_csv", default="cy_points_grid_2km.csv")
    ap.add_argument("--geojson", action="store_true", help="Επίσης γράψε cy_points_grid_2km.geojson για έλεγχο")
    args = ap.parse_args()

    if not args.clc_raster and not args.clc_vector:
        raise SystemExit("Δώσε είτε --clc_raster είτε --clc_vector.")

    print("[1/4] Δημιουργία grid 2 km…")
    df = build_grid()

    print("[2/4] Δειγματοληψία CORINE…")
    if args.clc_raster:
        codes = sample_raster(Path(args.clc_raster), df)
    else:
        codes = sample_vector(Path(args.clc_vector), args.code_field, df)

    print("[3/4] Χαρτογράφηση CLC -> veg_class…")
    df["veg_class"] = [clc_to_veg(c) for c in codes]

    # Κόψε σημεία με None (εκτός κάλυψης)
    df = df[df["veg_class"].notna()].reset_index(drop=True)

    print("[4/4] Εγγραφή CSV…")
    df[["location","lat","lon","veg_class"]].to_csv(args.out_csv, index=False)
    print(f"[OK] Γράφτηκε: {args.out_csv} ({len(df)} γραμμές)")

    if args.geojson:
        print("[+] Εγγραφή GeoJSON για έλεγχο…")
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
            crs="EPSG:4326"
        )
        gdf.to_file("cy_points_grid_2km.geojson", driver="GeoJSON")
        print("[OK] cy_points_grid_2km.geojson")
if __name__ == "__main__":
    main()

