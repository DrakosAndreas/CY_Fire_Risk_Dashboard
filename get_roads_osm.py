#!/usr/bin/env python3
import osmnx as ox

place = "Cyprus"  # Administrative boundary
# κατέβασμα γράφου δρόμων
G = ox.graph_from_place(place, network_type="drive", simplify=True)

# edges ως GeoDataFrame
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True, fill_edge_geometry=True)
edges = edges[edges.geometry.type.isin(["LineString","MultiLineString"])][["geometry","highway"]]

edges.to_file("data/roads.shp")
print("[OK] Saved roads to data/roads.shp")
