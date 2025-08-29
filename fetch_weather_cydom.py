#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_weather_cydom.py — Cyprus DoM AWS XML → weather CSV (ταιριασμένο στη δομή με <stations> + πολλαπλά <observations>)
Παράγει: location,lat,lon,temp_c,rh,wind_mps

Δουλεύει με XML όπου:
  - <stations>/<station> περιέχουν station_code, station_latitude, station_longitude (και προαιρετικά station_name)
  - Υπάρχουν πολλά <observations> blocks. Κάθε block έχει:
      station_code, date_time
      και πολλά <observation> με observation_name / observation_value / observation_unit

Χρήση:
  python fetch_weather_cydom.py --out cy_weather_sample.csv
  python fetch_weather_cydom.py --out cy_weather_sample.csv --xml cydom_raw.xml
"""

import argparse
import csv
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import requests

DOM_XML_URL = "https://www.dom.org.cy/AWS/OpenData/CyDoM.xml"
HTTP_TIMEOUT = 25

# ---------------- Helpers ----------------
def knots_to_mps(k: float) -> float:
    return float(k) * 0.514444

def to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", ".")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def atomic_write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8") as tmp:
        w = csv.DictWriter(tmp, fieldnames=["location", "lat", "lon", "temp_c", "rh", "wind_mps"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        tmpname = tmp.name
    shutil.move(tmpname, out_path)

def strip_namespaces(elem: ET.Element) -> None:
    for el in elem.iter():
        if isinstance(el.tag, str) and el.tag.startswith("{"):
            el.tag = el.tag.split("}", 1)[1]

def fetch_root(xml_path: Optional[Path]) -> ET.Element:
    if xml_path:
        data = Path(xml_path).read_bytes()
    else:
        resp = requests.get(DOM_XML_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.content
    root = ET.fromstring(data)
    strip_namespaces(root)
    return root

# ---------------- Parse stations ----------------
def build_station_index(root: ET.Element) -> Dict[str, dict]:
    """
    Φτιάχνει dict: code -> {name, lat, lon}
    Περιμένει tags:
      station_code, station_name?, station_latitude, station_longitude
    """
    idx: Dict[str, dict] = {}
    stations_parent = None
    for el in root.iter():
        t = getattr(el, "tag", "")
        if isinstance(t, str) and t.lower().endswith("stations"):
            stations_parent = el
            break
    if stations_parent is None:
        return idx

    for st in stations_parent.findall(".//station"):
        code = name = None
        lat = lon = None
        for c in st:
            tag = (c.tag or "").lower()
            txt = (c.text or "").strip() if c.text else ""
            if tag in ("station_code", "code", "stationcode"):
                code = txt
            elif tag in ("station_name", "name", "stationname"):
                name = txt
            elif tag in ("station_latitude", "latitude", "lat"):
                lat = to_float(txt)
            elif tag in ("station_longitude", "longitude", "lon"):
                lon = to_float(txt)
        if code:
            idx[code] = {"name": name or code, "lat": lat, "lon": lon}
    return idx

# ---------------- Parse observations blocks ----------------
def collect_observations_blocks(root: ET.Element) -> List[ET.Element]:
    """Βρίσκει ΟΛΑ τα elements που το tag τους περιέχει 'observations' (case-insensitive)."""
    blocks: List[ET.Element] = []
    for el in root.iter():
        t = getattr(el, "tag", "")
        if isinstance(t, str) and "observations" in t.lower():
            blocks.append(el)
    return blocks

def parse_observations_block(block: ET.Element) -> Tuple[Optional[str], dict]:
    """
    Από ένα <observations> block παίρνει:
      - station_code (στον parent)
      - params: dict observation_name.lower() -> (value, unit)
    """
    code = None
    params: Dict[str, Tuple[float, Optional[str]]] = {}

    # διαβάζουμε station_code/date_time από τα direct children του block
    for child in block:
        tag = (child.tag or "").lower()
        txt = (child.text or "").strip() if child.text else ""
        if tag in ("station_code", "stationcode", "code"):
            code = txt

    # διαβάζουμε όλα τα <observation> μέσα στο block
    for ob in block.findall(".//observation"):
        name = unit = None
        val = None
        for c in ob:
            t = (c.tag or "").lower()
            text = (c.text or "").strip() if c.text else ""
            if t in ("observation_name", "parameter", "name"):
                name = text
            elif t in ("observation_value", "value"):
                val = to_float(text)
            elif t in ("observation_unit", "unit"):
                unit = text
        if name and (val is not None):
            params[name.lower()] = (float(val), unit)

    return code, params

def extract_triplet(params: Dict[str, Tuple[float, Optional[str]]]) -> Optional[Tuple[float, float, float]]:
    """
    Επιστρέφει (temp_c, rh, wind_mps) από τα params.
    Κλειδιά που ψάχνουμε (case-insensitive, substring match):
      - temp: 'air temperature (1.2m)'
      - rh:   'relative humidity (1.2m)'
      - wind: first 'wind speed (2m)' [m/s] else 'wind speed (10m)' [knots → m/s]
    """
    # βοηθητικό για substring-match
    def find_first(sub: str) -> Optional[Tuple[float, Optional[str]]]:
        for k, vu in params.items():
            if sub in k:
                return vu
        return None

    t = find_first("air temperature (1.2m)")
    h = find_first("relative humidity (1.2m)")
    w2 = find_first("wind speed (2m)")
    w10 = find_first("wind speed (10m)")

    if not (t and h and (w2 or w10)):
        return None

    temp_c = float(t[0])
    rh = float(h[0])
    if not (0.0 <= rh <= 100.0):
        return None

    if w2:
        wind_mps = float(w2[0])  # ήδη m/s
    else:
        wind_mps = knots_to_mps(float(w10[0]))

    return (round(temp_c, 1), round(rh, 1), round(wind_mps, 2))

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cy_weather_sample.csv", help="CSV έξοδος για dashboard")
    ap.add_argument("--xml", default=None, help="(προαιρετικό) τοπικό XML αντί για live URL")
    args = ap.parse_args()

    try:
        root = fetch_root(Path(args.xml) if args.xml else None)
    except Exception as e:
        print(f"[ERROR] Αποτυχία φόρτωσης XML: {e}")
        sys.exit(2)

    # 1) Σταθερά στοιχεία σταθμών
    st_idx = build_station_index(root)
    if not st_idx:
        print("[ERROR] Δεν βρέθηκαν σταθμοί μέσα σε <stations>.")
        sys.exit(3)

    # 2) Όλα τα observation blocks
    blocks = collect_observations_blocks(root)
    if not blocks:
        print("[ERROR] Δεν βρέθηκαν καθόλου blocks 'observations' στο XML.")
        sys.exit(4)

    # 3) Συνένωση ανά block
    out_rows: List[dict] = []
    for bl in blocks:
        code, params = parse_observations_block(bl)
        if not code or not params:
            continue
        trip = extract_triplet(params)
        if not trip:
            continue
        temp_c, rh, wind_mps = trip

        meta = st_idx.get(code, {})
        name = meta.get("name") or code
        lat = meta.get("lat")
        lon = meta.get("lon")
        if lat is None or lon is None:
            continue  # δεν μπορούμε να το τοποθετήσουμε στο χάρτη

        out_rows.append({
            "location": str(name),
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "temp_c": temp_c,
            "rh": rh,
            "wind_mps": wind_mps,
        })

    if not out_rows:
        print("[ERROR] Δεν παρήχθησαν γραμμές: λείπουν οι απαραίτητες παρατηρήσεις (Temp/RH/Wind).")
        sys.exit(5)

    atomic_write_csv(out_rows, Path(args.out))
    print(f"[OK] Γράφτηκαν {len(out_rows)} εγγραφές στο {args.out}")

if __name__ == "__main__":
    main()
