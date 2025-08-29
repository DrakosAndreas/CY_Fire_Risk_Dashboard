
Cyprus Wildfire Hotspot & Staffing Prototype
===========================================
Generated: 2025-08-22 19:36

Files:
  - cy_fire_risk.py               : Main script
  - cy_points_sample.csv          : Sample locations & vegetation class
  - cy_weather_sample.csv         : Sample weather (replace with live)
  - cy_fire_risk_today.csv        : Output (after running)

Usage:
  python cy_fire_risk.py [points.csv] [weather.csv] [out.csv]

Notes:
  * Weather-only danger is computed with the Fosberg Fire Weather Index (FFWI).
  * Vegetation effect is applied as a multiplier (see VEG_FACTORS).
  * Results are heuristics for staging and must be calibrated with local doctrine.

