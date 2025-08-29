#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Φτιάχνει WUI πολύγωνα από OSM κτίρια για την Κύπρο.
Απαιτεί: osmnx>=2.0, geopandas, shapely

Έξοδος: data/wui.shp (EPSG:4326)
"""
import os
import sys
import warnings
import osmnx as ox
import geopandas as gpd
from shapely.ops import unary_union

# Ρύθμιση περιοχής ενδιαφέροντος
PLACE = "Cyprus"   # μπορείς να βάλεις "Republic of Cyprus" ή μικρότερη περιοχή
BUFFER_M = 50      # buffer γύρω από κτίρια (μέτρα) για WUI όγκο

# Προαιρετικά: περιορισμός με BBOX (lon_min, lat_min, lon_max, lat_max)
BBOX = (32.25, 34.45, 34.85, 35.75)  # Κύπρος σε WGS84

def fetch_buildings_by_polygon(place_name: str) -> gpd.GeoDataFrame:
    """Προσπαθεί να πάρει τα κτίρια με polygon (administrative boundary)."""
    print("[i] Αναζήτηση διοικητικού πολυγώνου…")
    try:
        area = ox.geocode_to_gdf(place_name)  # polygon
        if area.empty:
            raise ValueError("Empty polygon")
        poly = area.to_crs("EPSG:4326").geometry.unary_union
        print("[i] Κατέβασμα buildings με features_from_polygon…")
        g = ox.features_from_polygon(poly, tags={"building": True})
        return g
    except Exception as e:
        warnings.warn(f"[warn] polygon fetch failed ({e}). Θα δοκιμάσω bbox.")
        return gpd.GeoDataFrame()

def fetch_buildings_by_bbox(bounds) -> gpd.GeoDataFrame:
    """Fallback: λήψη κτιρίων από bbox."""
    south, north = bounds[1], bounds[3]
    west, east = bounds[0], bounds[2]
    print("[i] Κατέβασμα buildings με features_from_bbox…")
    try:
        g = ox.features_from_bbox(north=north, south=south, east=east, west=west, tags={"building": True})
        return g
    except Exception as e:
        raise SystemExit(f"[ERROR] Βλάβη στο OSM fetch (bbox): {e}")

def main():
    os.makedirs("data", exist_ok=True)

    print("[1/4] Λήψη κτιρίων από OSM…")
    bld = fetch_buildings_by_polygon(PLACE)
    if bld.empty:
        bld = fetch_buildings_by_bbox(BBOX)

    if bld.empty:
        raise SystemExit("[ERROR] Δεν βρέθηκαν buildings από OSM")

    # Κράτα μόνο έγκυρες γεωμετρίες
    bld = bld[~bld.geometry.is_empty & bld.geometry.notna()].copy()
    if bld.empty:
        raise SystemExit("[ERROR] Τα buildings δεν έχουν χρήσιμες γεωμετρίες")

    print(f"[i] Συνολικά features: {len(bld)}")

    # Σε μετρικό CRS για buffer
    bld = bld.to_crs("EPSG:3857")

    print("[2/4] Υπολογισμός buffer ({} m) γύρω από κτίρια…".format(BUFFER_M))
    # buffer ανά γεωμετρία (μπορεί να πάρει 1-2′)
    buf = bld.geometry.buffer(BUFFER_M)

    print("[3/4] Dissolve (unary_union) σε ένα ή περισσότερα πολύγωνα…")
    # Ενώνουμε τα buffers σε “όγκο” WUI
    u = unary_union(buf)
    wui = gpd.GeoDataFrame(geometry=[u], crs="EPSG:3857").explode(index_parts=False).reset_index(drop=True)

    print(f"[i] Παρήχθησαν {len(wui)} WUI polygons")

    print("[4/4] Αποθήκευση σε data/wui.shp (EPSG:4326)…")
    wui = wui.to_crs("EPSG:4326")
    out_path = "data/wui.shp"
    # overwrite αν υπάρχει
    if os.path.exists(out_path):
        try:
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                p = out_path.replace(".shp", ext)
                if os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass
    wui.to_file(out_path)
    print("[OK] Γράφτηκε:", out_path)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
