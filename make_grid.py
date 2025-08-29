#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Rough bounding box για Κύπρο
lat_min, lat_max = 34.4, 35.8
lon_min, lon_max = 32.0, 34.9

# Βήμα ~2km (0.02 μοίρες σε lat ≈ 2.2 km)
step_lat = 0.02
step_lon = 0.02 / np.cos(np.deg2rad(35.0))  # διόρθωση για longitude

lats = np.arange(lat_min, lat_max + 1e-9, step_lat)
lons = np.arange(lon_min, lon_max + 1e-9, step_lon)

rows = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        rows.append({
            "location": f"Cell_{i}_{j}",
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "veg_class": "shrubland"  # προσωρινά, μπορείς να βάλεις άλλο dataset
        })

df = pd.DataFrame(rows)
df.to_csv("cy_points_grid_2km.csv", index=False)
print(f"Έγραψα {len(df)} σημεία στο cy_points_grid_2km.csv")
