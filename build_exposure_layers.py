#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_geoms(path, to_crs="EPSG:3857"):
    if not path:
        return None
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    if to_crs:
        g = g.to_crs(to_crs)
    return g

def sjoin_nearest_dist_km(points_gdf, target_gdf, col_name):
    """Γεμίζει στήλη με ελάχιστη απόσταση (km) από target_gdf. Αν target κενό -> NaN."""
    if target_gdf is None or target_gdf.empty:
        return pd.Series([pd.NA]*len(points_gdf))
    joined = gpd.sjoin_nearest(
        points_gdf, target_gdf[["geometry"]],
        how="left", distance_col="__dist_m"
    )
    dist_by_pid = joined[["_pid","__dist_m"]].groupby("_pid", as_index=True)["__dist_m"].min()
    out = pd.Series([pd.NA]*len(points_gdf), index=points_gdf.index)
    out.loc[dist_by_pid.index] = (dist_by_pid / 1000.0).round(3)
    out.name = col_name
    return out

def sjoin_within_flag(points_gdf, polys_gdf, col_name="in_protected"):
    """0/1 flag αν σημείο είναι εντός πολυγώνου(ων)."""
    if polys_gdf is None or polys_gdf.empty:
        return pd.Series([0]*len(points_gdf), dtype=int, index=points_gdf.index, name=col_name)
    j = gpd.sjoin(points_gdf, polys_gdf[["geometry"]], how="left", predicate="within")
    flag = j.index_right.notna().astype(int)
    flag.name = col_name
    return flag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="cy_points_grid_2km.csv")
    ap.add_argument("--roads", default=None)
    ap.add_argument("--wui", default=None)
    ap.add_argument("--protected", default=None)
    ap.add_argument("--out_csv", default="cy_points_grid_2km.csv")
    args = ap.parse_args()

    # Load grid points
    df = pd.read_csv(args.in_csv)
    df = df.reset_index(drop=True).copy()
    df["_pid"] = df.index

    pts = gpd.GeoDataFrame(
        df[["_pid","lat","lon"]].copy(),
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    # Load layers
    g_roads = load_geoms(args.roads)
    g_wui   = load_geoms(args.wui)
    g_prot  = load_geoms(args.protected)

    # Distances / flags
    if args.roads:
        df["dist_roads_km"] = sjoin_nearest_dist_km(pts, g_roads, "dist_roads_km")
    if args.wui:
        df["dist_wui_km"]   = sjoin_nearest_dist_km(pts, g_wui,   "dist_wui_km")
    if args.protected:
        df["in_protected"]  = sjoin_within_flag(pts, g_prot, "in_protected")

    # Cleanup
    df.drop(columns=["_pid"], inplace=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} with",
          (" dist_roads_km" if args.roads else ""),
          (" dist_wui_km" if args.wui else ""),
          (" in_protected" if args.protected else ""))
if __name__ == "__main__":
    main()
