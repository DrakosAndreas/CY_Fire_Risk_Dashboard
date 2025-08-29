#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import osmnx as ox
import geopandas as gpd
from shapely.ops import unary_union

PLACE = "Cyprus"

def fetch_union(tags):
    g = ox.features_from_place(PLACE, tags)
    if g.empty:
        return None
    g = g.to_crs("EPSG:3857")
    geom = g.geometry.dropna()
    if geom.empty:
        return None
    return unary_union(geom)

def main():
    print("[1/3] Λήψη protected areas από OSM…")
    # δύο βασικές κατηγορίες στο OSM: boundary=protected_area και leisure=nature_reserve
    u1 = fetch_union({"boundary": "protected_area"})
    u2 = fetch_union({"leisure": "nature_reserve"})

    if u1 is None and u2 is None:
        raise SystemExit("[WARN] Δεν βρέθηκαν προστατευόμενες περιοχές στο OSM για την περιοχή.")

    print("[2/3] Ενοποίηση…")
    if u1 and u2:
        u = unary_union([u1, u2])
    else:
        u = u1 or u2

    print("[3/3] Αποθήκευση σε data/protected.shp (EPSG:4326)…")
    gdf = gpd.GeoDataFrame(geometry=[u], crs="EPSG:3857").explode(index_parts=False).reset_index(drop=True)
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file("data/protected.shp")
    print("[OK] data/protected.shp")

if __name__ == "__main__":
    main()

