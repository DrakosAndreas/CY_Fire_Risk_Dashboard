# fwi.py
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass

# --- Canadian FWI monthly factors (standard, Canada). ---
# Για ακριβέστερα εκτός Καναδά, μπορείς να βάλεις latitude-adjusted πίνακες.
LE_DMC = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]   # DMC day-length
LF_DC  = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]  # DC seasonal

@dataclass
class Weather:
    T: float      # noon temp [°C]
    H: float      # noon RH   [%]
    U: float      # noon wind [km/h] at 10 m
    P: float      # 24h rain  [mm]
    month: int    # 1..12

@dataclass
class Codes:
    FFMC: float = 85.0
    DMC:  float = 6.0
    DC:   float = 15.0

def ffmc_calc(prev_ffmc, T, H, U, P):
    # 1) Μετατροπή FFMC_previous -> m (moisture content)
    m = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)

    # 2) Βροχή (effective rain αν P>0.5)
    if P > 0.5:
        rf = P - 0.5
        if m <= 150.0:
            m = m + 42.5 * rf * math.exp(-100.0 / (251.0 - m)) * (1 - math.exp(-6.93 / rf))
        else:
            m = m + 42.5 * rf * math.exp(-100.0 / (251.0 - m)) * (1 - math.exp(-6.93 / rf)) \
                + 0.0015 * (m - 150.0) ** 2 * math.sqrt(rf)
        m = min(m, 250.0)

    # 3) Ισορροπία ξήρανσης/ύγρανσης
    Ed = 0.942 * (H ** 0.679) + (11.0 * math.exp((H - 100.0) / 10.0)) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))
    Ew = 0.618 * (H ** 0.753) + (10.0 * math.exp((H - 100.0) / 10.0)) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))

    # 4) Ρυθμοί
    if m < Ed:  # drying
        ko = 0.424 * (1.0 - ((H/100.0) ** 1.7)) + (0.0694 * math.sqrt(U)) * (1.0 - ((H/100.0) ** 8))
        kd = ko * 0.581 * math.exp(0.0365 * T)
        m = Ew + (m - Ew) * math.exp(-kd)
    else:       # wetting
        kw = 0.424 * (1.0 - ((H/100.0) ** 1.7)) + 0.0694 * math.sqrt(U) * (1.0 - ((H/100.0) ** 8))
        kw = kw * 0.581 * math.exp(0.0365 * T)
        m = Ed + (m - Ed) * math.exp(-kw)

    # 5) Πίσω σε FFMC
    ffmc = (59.5 * (250.0 - m)) / (147.2 + m)
    return max(0.0, min(101.0, ffmc))

def dmc_calc(prev_dmc, T, H, P, month):
    dmc = prev_dmc
    if P > 1.5:
        re = 0.92 * P - 1.27
        mo = 20.0 + math.exp(5.6348 - prev_dmc / 43.43)
        if prev_dmc <= 33.0:
            b = 100.0 / (0.5 + 0.3 * prev_dmc)
        elif prev_dmc <= 65.0:
            b = 14.0 - 1.3 * math.log(prev_dmc)
        else:
            b = 6.2 * math.log(prev_dmc) - 17.2
        mr = mo + (1000.0 * re) / (48.77 + b * re)
        dmc = 244.72 - 43.43 * math.log(mr - 20.0)
    # drying increment
    T_eff = max(-1.1, T)
    K = 1.894 * (T_eff + 1.1) * (100.0 - H) * LE_DMC[month - 1] * 1e-6
    return max(0.0, dmc + K)

def dc_calc(prev_dc, T, P, month):
    dc = prev_dc
    if P > 2.8:
        rd = 0.83 * P - 1.27
        Qo = 800.0 * math.exp(-prev_dc / 400.0)
        Qr = Qo + 3.937 * rd
        dc = max(0.0, 400.0 * math.log(800.0 / Qr))
    # drying increment
    T_eff = max(-2.8, T)
    V = 0.36 * (T_eff + 2.8) + LF_DC[month - 1]
    V = max(0.0, V)
    return dc + V

def isi_calc(ffmc, U):
    # FFMC -> moisture m
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fU = math.exp(0.05039 * U)
    fF = 91.9 * math.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / (4.93e7))
    return 0.208 * fU * fF

def bui_calc(dmc, dc):
    if dmc <= 0.4 * dc:
        return (0.8 * dmc * dc) / (dmc + 0.4 * dc + 1e-9)
    else:
        return dmc - (1.0 - (0.8 * dc) / (dmc + 0.4 * dc + 1e-9)) * (0.92 + (0.0114 * dmc) ** 1.7)

def fwi_calc(isi, bui):
    if bui <= 0.0:
        F_D = 0.0
    else:
        F_D = (0.626 * (bui ** 0.809) + 2.0) / (1.0 + 0.0186 * bui)
    B = 0.1 * isi * F_D
    if B <= 1.0:
        return B
    else:
        return math.exp(2.72 * ((math.log(B) * 0.434) ** 0.647))

def step_fwi(prev: Codes, w: Weather):
    """Ένα “βήμα ημέρας” FWI με χθεσινούς κώδικες και σημερινό καιρό."""
    ffmc = ffmc_calc(prev.FFMC, w.T, w.H, w.U, w.P)
    dmc  = dmc_calc(prev.DMC,  w.T, w.H, w.P, w.month)
    dc   = dc_calc(prev.DC,    w.T, w.P, w.month)
    isi  = isi_calc(ffmc, w.U)
    bui  = bui_calc(dmc, dc)
    fwi  = fwi_calc(isi, bui)
    return Codes(ffmc, dmc, dc), {"ISI": isi, "BUI": bui, "FWI": fwi}

def fwi_level(fwi):
    # Ενδεικτικές κλίμακες (ευρέως χρησιμοποιούμενες)
    if fwi < 5: return "Low"
    if fwi < 12: return "Moderate"
    if fwi < 21: return "High"
    if fwi < 38: return "Very High"
    return "Extreme"
