#!/usr/bin/env python3
"""
ISOTOPER B15 — Geospatial Radiation Dose Mapping & Interpolation Engine
========================================================================
Fetches all GM-counter telemetry rows directly from the Google Sheet
(1,136+ rows), filters out zero/invalid readings (54 zero rows filtered ->
1,082 valid readings), maps all readings across a user-defined
Latitude/Longitude survey bounding box along a UAV lawnmower flight path,
interpolates a 56 × 38 (2,128-cell) dose distribution grid on standard
OpenStreetMap (OSM) tiles (100% free, no API key required), and generates:
  1. radiation_dispersion_map.html (Dual / Interpolated Distribution View)
  2. radiation_map_light.html      (Discrete Measurement Nodes View)
"""

import argparse
import io
import json
import math
import warnings
from pathlib import Path

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import requests
from folium import LayerControl

warnings.filterwarnings("ignore")

# ============================================================================
# 1. UNIFIED CONFIGURATION (ISOTOPER B15 DEVELOPER HANDOFF SPEC)
# ============================================================================

GOOGLE_SHEET_ID = "1Gaiwuh3fOml1ib8fwBSZfUDDzHrzGYV9crpKF-P2foU"
GOOGLE_SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv"
)
GM_MACRO_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbzCBRfLEIenGJcOjMiCfhb3aIdF_kop9LOhxdMUeVzn__Sl10te8O-uQcWIacMrw4WpWw/exec"
)

# Default Survey Boundary: MIST Campus, Mirpur Cantonment, Dhaka-1216
DEFAULT_SURVEY_BOUNDS = {
    "lat1": 23.837700,
    "lon1": 90.357000,
    "lat2": 23.838500,
    "lon2": 90.357900,
    "label": "MIST Campus, Mirpur Cantonment, Dhaka",
}

GRID_COLS = 56
GRID_ROWS = 38
ANNUAL_HOURS = 2000.0  # 8 h/day × 250 days/yr
PUBLIC_LIMIT_MSV = 1.0
WORKER_LIMIT_MSV = 20.0

DOSE_CONFIG = {
    "minLogDose": 0.1,
    "maxLogDose": 30.0,
    "annualHours": 2000,
    "publicLimitMSv": 1.0,
    "workerLimitMSv": 20.0,
    "gridCols": GRID_COLS,
    "gridRows": GRID_ROWS,
    "tiers": [
        {
            "key": "safe",
            "name": "Safe",
            "min": 0.0,
            "max": 0.20,
            "rangeLabel": "< 0.20 µSv/h",
            "annualLabel": "< 0.4 mSv/yr",
            "meaning": "Natural background (0.10–0.15)",
            "shortMeaning": "Natural background",
            "color": "#22C55E",
            "textColor": "#052E16",
        },
        {
            "key": "low",
            "name": "Low",
            "min": 0.20,
            "max": 0.50,
            "rangeLabel": "0.20 – 0.50 µSv/h",
            "annualLabel": "0.4 – 1 mSv/yr",
            "meaning": "Within 1 mSv/yr public limit",
            "shortMeaning": "Within public limit",
            "color": "#A3E635",
            "textColor": "#1A2E05",
        },
        {
            "key": "moderate",
            "name": "Moderate",
            "min": 0.50,
            "max": 3.00,
            "rangeLabel": "0.50 – 3.0 µSv/h",
            "annualLabel": "1 – 6 mSv/yr",
            "meaning": "Above public limit",
            "shortMeaning": "Above public limit",
            "color": "#FACC15",
            "textColor": "#422006",
        },
        {
            "key": "high",
            "name": "High",
            "min": 3.00,
            "max": 10.00,
            "rangeLabel": "3.0 – 10 µSv/h",
            "annualLabel": "6 – 20 mSv/yr",
            "meaning": "Category A worker band",
            "shortMeaning": "Category A worker band",
            "color": "#F97316",
            "textColor": "#431407",
        },
        {
            "key": "severe",
            "name": "Severe",
            "min": 10.00,
            "max": float("inf"),
            "rangeLabel": "≥ 10 µSv/h",
            "annualLabel": "> 20 mSv/yr",
            "meaning": "Above 20 mSv/yr worker limit",
            "shortMeaning": "Above worker limit",
            "color": "#C81E1E",
            "textColor": "#FFFFFF",
        },
    ],
    "rampStops": [
        {"dose": 0.1, "pos": 0.0, "color": "#16A34A"},
        {"dose": 0.2, "pos": 0.1215, "color": "#84CC16"},
        {"dose": 0.5, "pos": 0.2822, "color": "#FACC15"},
        {"dose": 3.0, "pos": 0.5963, "color": "#F97316"},
        {"dose": 10.0, "pos": 0.8074, "color": "#DC2626"},
        {"dose": 30.0, "pos": 1.0, "color": "#B91C1C"},
    ],
}

log10_300 = math.log10(300.0)


# ============================================================================
# 2. COLOR & TIER UTILITIES
# ============================================================================

def get_tier_for_dose(dose_uSv_h: float) -> dict:
    """Return the tier dictionary from DOSE_CONFIG for a given dose rate."""
    for tier in DOSE_CONFIG["tiers"]:
        if dose_uSv_h < tier["max"]:
            return tier
    return DOSE_CONFIG["tiers"][-1]


def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, int(round(rgb[0])))),
        max(0, min(255, int(round(rgb[1])))),
        max(0, min(255, int(round(rgb[2])))),
    )


def dose_to_continuous_hex(dose_uSv_h: float) -> str:
    """
    Map dose rate (µSv/h) to continuous color ramp on a log10 axis
    from 0.1 to 30 µSv/h, clamped at both ends:
      position = log10(v / 0.1) / log10(300)
    """
    clamped = max(0.1, min(30.0, float(dose_uSv_h)))
    pos = math.log10(clamped / 0.1) / log10_300
    stops = DOSE_CONFIG["rampStops"]

    for i in range(len(stops) - 1):
        s0, s1 = stops[i], stops[i + 1]
        if pos <= s1["pos"] or i == len(stops) - 2:
            span = s1["pos"] - s0["pos"]
            t = 0.0 if span <= 0 else max(0.0, min(1.0, (pos - s0["pos"]) / span))
            c0, c1 = hex_to_rgb(s0["color"]), hex_to_rgb(s1["color"])
            rgb = (
                c0[0] + (c1[0] - c0[0]) * t,
                c0[1] + (c1[1] - c0[1]) * t,
                c0[2] + (c1[2] - c0[2]) * t,
            )
            return rgb_to_hex(rgb)
    return stops[-1]["color"]


# ============================================================================
# 3. FULL GOOGLE SHEET DATA FETCHING & ZERO-FILTERING (ALL 1,136+ ROWS)
# ============================================================================

class RadiationData:
    """
    Fetches all GM-tube detector rows from the Google Sheet CSV endpoint
    (with fallback to the Apps Script macro), filters out zero/invalid readings
    (dose <= 0 or cpm <= 0), assigns coordinates across a user-defined
    bounding box along a serpentine UAV flight path, and performs 2D spatial
    interpolation onto a 56 × 38 grid.
    """

    def __init__(
        self,
        lat1: float = DEFAULT_SURVEY_BOUNDS["lat1"],
        lon1: float = DEFAULT_SURVEY_BOUNDS["lon1"],
        lat2: float = DEFAULT_SURVEY_BOUNDS["lat2"],
        lon2: float = DEFAULT_SURVEY_BOUNDS["lon2"],
        sheet_csv_url: str = GOOGLE_SHEET_CSV_URL,
        api_url: str = GM_MACRO_URL,
    ):
        self.sheet_csv_url = sheet_csv_url
        self.api_url = api_url
        self.lat_min = min(float(lat1), float(lat2))
        self.lat_max = max(float(lat1), float(lat2))
        self.lon_min = min(float(lon1), float(lon2))
        self.lon_max = max(float(lon1), float(lon2))

        self.raw_count = 0
        self.zero_filtered_count = 0
        self.readings = []
        self.nodes = []
        self.grid_matrix = None
        self.fetch_and_process()

    def fetch_and_process(self):
        """Fetch all rows from the Google Sheet CSV export and filter out <= 0 readings."""
        valid_readings = []
        zero_count = 0
        raw_total = 0

        try:
            print(f"[INFO] Fetching full GM counter dataset from Google Sheet ({GOOGLE_SHEET_ID})...")
            resp = requests.get(self.sheet_csv_url, timeout=20)
            resp.raise_for_status()
            df_raw = pd.read_csv(io.StringIO(resp.text))
            df_core = df_raw.iloc[:, :5].copy()
            df_core.columns = ["cps", "cpm", "dose", "status", "timestamp"]

            raw_total = len(df_core)
            for _, row in df_core.iterrows():
                try:
                    dose = float(row["dose"])
                    cpm = float(row["cpm"])
                    cps = float(row["cps"]) if not pd.isna(row["cps"]) else 0.0
                except (TypeError, ValueError):
                    zero_count += 1
                    continue

                if math.isnan(dose) or math.isnan(cpm) or dose <= 0.0 or cpm <= 0.0:
                    zero_count += 1
                    continue

                ts_val = "" if pd.isna(row["timestamp"]) else str(row["timestamp"]).strip()
                st_val = "" if pd.isna(row["status"]) else str(row["status"]).strip()

                valid_readings.append(
                    {
                        "id": len(valid_readings) + 1,
                        "dose": round(dose, 4),
                        "cpm": round(cpm, 1),
                        "cps": round(cps, 2),
                        "status": st_val or get_tier_for_dose(dose)["name"],
                        "timestamp": ts_val,
                    }
                )
        except Exception as exc:
            print(f"[WARN] Full Google Sheet CSV fetch failed ({exc}). Trying macro endpoint...")
            try:
                resp = requests.get(self.api_url, timeout=15)
                resp.raise_for_status()
                payload = resp.json()
                raw_items = payload.get("history", []) if isinstance(payload, dict) else []
                raw_total = len(raw_items)
                for item in raw_items:
                    try:
                        dose = float(item.get("dose", 0) or 0)
                        cpm = float(item.get("cpm", 0) or 0)
                        cps = float(item.get("cps", 0) or 0)
                    except (TypeError, ValueError):
                        zero_count += 1
                        continue
                    if math.isnan(dose) or dose <= 0.0 or cpm <= 0.0:
                        zero_count += 1
                        continue
                    valid_readings.append(
                        {
                            "id": len(valid_readings) + 1,
                            "dose": round(dose, 4),
                            "cpm": round(cpm, 1),
                            "cps": round(cps, 2),
                            "status": str(item.get("status", "")).strip() or get_tier_for_dose(dose)["name"],
                            "timestamp": str(item.get("timestamp", "")),
                        }
                    )
            except Exception as exc2:
                print(f"[WARN] Macro fallback also failed ({exc2}).")

        self.raw_count = raw_total
        self.zero_filtered_count = zero_count
        self.readings = valid_readings

        print(
            f"[INFO] Full Sheet Loaded: {self.raw_count} total rows | "
            f"{self.zero_filtered_count} zero/invalid rows filtered | "
            f"{len(self.readings)} valid readings plotted."
        )
        self.assign_coordinates_and_interpolate()

    def assign_coordinates_and_interpolate(self):
        """
        Distribute all valid GM readings along a serpentine (lawnmower) UAV survey
        path inside [lat_min..lat_max, lon_min..lon_max] and compute the 56 × 38
        interpolated grid using a local hotspot-preserving kernel.
        """
        n = len(self.readings)
        if n == 0:
            return

        rows = max(2, int(round(math.sqrt(n * 0.62))))
        cols = max(2, int(math.ceil(n / rows)))

        lat_span = max(1e-6, self.lat_max - self.lat_min)
        lon_span = max(1e-6, self.lon_max - self.lon_min)

        pad_lat = lat_span * 0.05
        pad_lon = lon_span * 0.05
        inner_lat_min = self.lat_min + pad_lat
        inner_lat_max = self.lat_max - pad_lat
        inner_lon_min = self.lon_min + pad_lon
        inner_lon_max = self.lon_max - pad_lon

        nodes = []
        for idx, r in enumerate(self.readings):
            row_idx = idx // cols
            col_in_row = idx % cols
            pts_in_this_row = min(cols, n - row_idx * cols)

            lat_t = 0.5 if rows <= 1 else row_idx / float(rows - 1)
            lat = inner_lat_max - lat_t * (inner_lat_max - inner_lat_min)

            lon_t = 0.5 if pts_in_this_row <= 1 else col_in_row / float(pts_in_this_row - 1)
            if row_idx % 2 == 1:
                lon_t = 1.0 - lon_t
            lon = inner_lon_min + lon_t * (inner_lon_max - inner_lon_min)

            tier = get_tier_for_dose(r["dose"])
            nodes.append(
                {
                    "id": idx + 1,
                    "lat": round(lat, 7),
                    "lon": round(lon, 7),
                    "dose": r["dose"],
                    "cpm": r["cpm"],
                    "cps": r["cps"],
                    "status": r["status"],
                    "tierKey": tier["key"],
                    "tierName": tier["name"],
                    "color": tier["color"],
                    "annualMSv": round(r["dose"] * (ANNUAL_HOURS / 1000.0), 3),
                    "timestamp": r["timestamp"],
                }
            )

        self.nodes = nodes

        # Local hotspot-preserving IDW + Gaussian interpolation onto 56 × 38 grid
        grid_lats = np.linspace(self.lat_max, self.lat_min, GRID_ROWS)
        grid_lons = np.linspace(self.lon_min, self.lon_max, GRID_COLS)
        lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)

        node_lats = np.array([pt["lat"] for pt in nodes])[:, None, None]
        node_lons = np.array([pt["lon"] for pt in nodes])[:, None, None]
        node_doses = np.maximum(0.01, np.array([pt["dose"] for pt in nodes]))[:, None, None]

        dy = (lat_mesh[None, :, :] - node_lats) / lat_span
        dx = (lon_mesh[None, :, :] - node_lons) / lon_span
        dist_sq = dx * dx + dy * dy

        sigma = 0.028 if n > 200 else 0.11
        weights = np.exp(-dist_sq / (2.0 * sigma * sigma)) + 1.0 / np.power(
            dist_sq + 0.00003, 2.15
        )
        self.grid_matrix = np.sum(weights * node_doses, axis=0) / np.sum(weights, axis=0)

    def get_dataframe(self):
        """Return pandas DataFrame of measurement nodes and summary statistics."""
        df = pd.DataFrame(self.nodes)
        df["Longitude"] = df["lon"]
        df["Latitude"] = df["lat"]
        df["Radiation_μSv_h"] = df["dose"]

        grid_flat = self.grid_matrix.ravel()
        stats = {
            "mean": float(np.mean(grid_flat)),
            "std": float(np.std(grid_flat)),
            "max": float(np.max(grid_flat)),
            "min": float(np.min(grid_flat)),
            "median": float(np.median(grid_flat)),
            "grid_cells": int(grid_flat.size),
            "node_count": int(len(self.nodes)),
        }
        return df, stats


# ============================================================================
# 4. REFERENCE SAMPLE FIELD (48-POINT MULTI-TIER SPEC FROM DESIGN HANDOFF)
# ============================================================================

SAMPLE_FIELD_DOSES = [
    0.112, 0.125, 0.142, 0.158, 0.164, 0.149, 0.131, 0.118,
    0.145, 8.870, 2.980, 0.617, 0.185, 1.110, 0.192, 0.134,
    0.156, 0.735, 2.250, 8.440, 23.20, 17.80, 0.195, 0.148,
    0.162, 3.320, 16.20, 13.20, 5.160, 1.450, 0.245, 0.152,
    0.175, 1.440, 4.080, 7.150, 4.960, 0.643, 0.285, 0.160,
    0.138, 0.182, 0.295, 1.170, 2.310, 1.750, 0.783, 0.215,
]


# ============================================================================
# 5. FOLIUM BASE LAYER + INTERACTIVE HTML APPLICATION GENERATOR
# ============================================================================

class RadiationMapGenerator:
    """
    Generates the redesigned ISOTOPER B15 geospatial radiation HTML artifacts:
      - radiation_dispersion_map.html (Dual / Interpolated Distribution View)
      - radiation_map_light.html      (Discrete Measurement Nodes View)
    """

    def __init__(
        self,
        lat1: float = DEFAULT_SURVEY_BOUNDS["lat1"],
        lon1: float = DEFAULT_SURVEY_BOUNDS["lon1"],
        lat2: float = DEFAULT_SURVEY_BOUNDS["lat2"],
        lon2: float = DEFAULT_SURVEY_BOUNDS["lon2"],
    ):
        self.radiation_data = RadiationData(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)
        self.center_lat = (self.radiation_data.lat_min + self.radiation_data.lat_max) / 2.0
        self.center_lon = (self.radiation_data.lon_min + self.radiation_data.lon_max) / 2.0

    def build_interactive_html(self, default_view: str = "split") -> str:
        """
        Render the complete ISOTOPER B15 Ground Control Radiation Dose Map HTML
        with all 1,082+ valid Google Sheet readings baked in AND live Google Sheet
        CSV synchronization in the browser.
        """
        compact_readings = [
            [r["dose"], r["cpm"], r["cps"], r["timestamp"]]
            for r in self.radiation_data.readings
        ]

        initial_payload = {
            "sheetCsvUrl": GOOGLE_SHEET_CSV_URL,
            "apiUrl": GM_MACRO_URL,
            "defaultView": default_view,
            "defaultBounds": {
                "lat1": self.radiation_data.lat_min,
                "lon1": self.radiation_data.lon_min,
                "lat2": self.radiation_data.lat_max,
                "lon2": self.radiation_data.lon_max,
                "label": DEFAULT_SURVEY_BOUNDS["label"],
            },
            "compactReadings": compact_readings,
            "rawCount": self.radiation_data.raw_count,
            "zeroFilteredCount": self.radiation_data.zero_filtered_count,
            "sampleFieldDoses": SAMPLE_FIELD_DOSES,
            "config": {
                "minLogDose": 0.1,
                "maxLogDose": 30.0,
                "annualHours": 2000,
                "publicLimitMSv": 1.0,
                "workerLimitMSv": 20.0,
                "gridCols": GRID_COLS,
                "gridRows": GRID_ROWS,
                "tiers": [
                    {
                        "key": t["key"],
                        "name": t["name"],
                        "min": t["min"],
                        "max": 999999 if math.isinf(t["max"]) else t["max"],
                        "rangeLabel": t["rangeLabel"],
                        "annualLabel": t["annualLabel"],
                        "meaning": t["meaning"],
                        "shortMeaning": t["shortMeaning"],
                        "color": t["color"],
                        "textColor": t["textColor"],
                    }
                    for t in DOSE_CONFIG["tiers"]
                ],
                "rampStops": DOSE_CONFIG["rampStops"],
            },
        }

        payload_json = json.dumps(initial_payload, ensure_ascii=False, separators=(",", ":"))
        return _HTML_TEMPLATE.replace("__INITIAL_PAYLOAD_JSON__", payload_json)

    def save_map(self, filename: str = "radiation_map_light.html") -> str:
        """Save both radiation_dispersion_map.html and radiation_map_light.html."""
        out_path = Path(filename)
        base_dir = out_path.parent if out_path.parent.exists() else Path(".")

        dispersion_file = base_dir / "radiation_dispersion_map.html"
        nodes_file = base_dir / "radiation_map_light.html"

        dispersion_html = self.build_interactive_html(default_view="split")
        nodes_html = self.build_interactive_html(default_view="nodes")

        dispersion_file.write_text(dispersion_html, encoding="utf-8")
        nodes_file.write_text(nodes_html, encoding="utf-8")

        print(f"[OK] Saved Interpolated / Split Dose Map -> {dispersion_file.resolve()}")
        print(f"[OK] Saved Measurement Nodes Map        -> {nodes_file.resolve()}")
        return str(out_path.resolve())


# ============================================================================
# 6. HTML / CSS / JS TEMPLATE (SINGLE-VIEW SCALED, OSM DEFAULT, NO SCROLLBAR)
# ============================================================================

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ISOTOPER B15 · Radiation Dose Map</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        :root {
            --bg-page: #0B0E14;
            --bg-card: #121721;
            --bg-card-inner: #161C28;
            --bg-input: #0E131B;
            --border-subtle: rgba(255, 255, 255, 0.09);
            --border-strong: rgba(255, 255, 255, 0.17);
            --text-main: #F3F4F6;
            --text-secondary: #9CA3AF;
            --text-muted: #6B7280;
            --accent: #38BDF8;
            --tier-safe: #22C55E;
            --tier-low: #A3E635;
            --tier-moderate: #FACC15;
            --tier-high: #F97316;
            --tier-severe: #C81E1E;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-display: 'Space Grotesk', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            background-color: var(--bg-page);
            color: var(--text-main);
            font-family: var(--font-sans);
            width: 100%;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }

        body {
            padding: 12px 14px 14px 14px;
        }

        .b15-wrapper {
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        /* ---------------- HEADER BAR ---------------- */
        .b15-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .b15-title-group {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }

        .b15-title {
            font-family: var(--font-display);
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF;
            line-height: 1.1;
        }

        .b15-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: var(--font-mono);
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-secondary);
        }

        .b15-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 9px;
            border-radius: 5px;
            font-family: var(--font-mono);
            font-size: 0.67rem;
            font-weight: 600;
            border: 1px solid var(--border-strong);
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
            cursor: pointer;
            transition: all 0.18s ease;
        }

        .b15-badge:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        .b15-badge-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #22C55E;
            box-shadow: 0 0 6px #22C55E;
        }

        /* ---------------- TOP CONTROLS ---------------- */
        .b15-controls {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }

        .seg-group {
            display: inline-flex;
            background: var(--bg-card);
            border: 1px solid var(--border-strong);
            border-radius: 7px;
            overflow: hidden;
            padding: 2px;
        }

        .seg-btn {
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-family: var(--font-sans);
            font-size: 0.74rem;
            font-weight: 600;
            padding: 4px 11px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .seg-btn.active {
            background: #F3F4F6;
            color: #0B0E14;
        }

        .chk-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-card);
            border: 1px solid var(--border-strong);
            border-radius: 7px;
            padding: 4px 10px;
            font-size: 0.74rem;
            font-weight: 500;
            color: var(--text-main);
            cursor: pointer;
            user-select: none;
            transition: border-color 0.15s ease;
        }

        .chk-pill:hover {
            border-color: rgba(255, 255, 255, 0.3);
        }

        .chk-pill input {
            display: none;
        }

        .chk-box {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            border: 1.5px solid var(--text-secondary);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            font-weight: 700;
            color: #0B0E14;
            background: transparent;
            transition: all 0.15s ease;
        }

        .chk-pill input:checked + .chk-box {
            background: #F3F4F6;
            border-color: #F3F4F6;
        }

        /* ---------------- USER SURVEY COORDINATE BAR ---------------- */
        .coord-bar {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }

        .coord-bar-left {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }

        .coord-tag {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 5px;
            white-space: nowrap;
        }

        .coord-group {
            display: flex;
            align-items: center;
            gap: 5px;
            background: var(--bg-input);
            border: 1px solid var(--border-subtle);
            padding: 3px 8px;
            border-radius: 6px;
        }

        .coord-group span {
            font-family: var(--font-mono);
            font-size: 0.65rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .coord-input {
            background: transparent;
            border: none;
            color: var(--text-main);
            font-family: var(--font-mono);
            font-size: 0.74rem;
            font-weight: 600;
            width: 86px;
            outline: none;
            padding: 1px 2px;
            border-bottom: 1px solid transparent;
        }

        .coord-input:focus {
            border-bottom-color: var(--accent);
        }

        .coord-actions {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }

        .action-btn {
            font-family: var(--font-mono);
            font-size: 0.69rem;
            font-weight: 600;
            padding: 5px 10px;
            border-radius: 6px;
            border: 1px solid var(--border-strong);
            background: var(--bg-card-inner);
            color: var(--text-main);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .action-btn:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        .action-btn.primary {
            background: #22C55E;
            border-color: #22C55E;
            color: #052E16;
            font-weight: 700;
        }

        .action-btn.primary:hover {
            background: #4ADE80;
            border-color: #4ADE80;
            color: #052E16;
        }

        .action-btn.picking {
            background: rgba(250, 204, 21, 0.16);
            border-color: #FACC15;
            color: #FACC15;
        }

        .feed-status {
            font-family: var(--font-mono);
            font-size: 0.67rem;
            color: var(--text-secondary);
            white-space: nowrap;
        }

        /* ---------------- DOSE RATE LOG SCALE CARD ---------------- */
        .scale-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .scale-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }

        .scale-title {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-secondary);
        }

        .scale-note {
            font-size: 0.68rem;
            color: var(--text-muted);
        }

        .ramp-wrap {
            position: relative;
            padding-bottom: 16px;
        }

        .ramp-bar {
            height: 10px;
            border-radius: 999px;
            width: 100%;
            background: linear-gradient(
                90deg,
                #16A34A 0%,
                #84CC16 12.15%,
                #FACC15 28.22%,
                #F97316 59.63%,
                #DC2626 80.74%,
                #B91C1C 100%
            );
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.4);
        }

        .ramp-bar.stepped {
            background: linear-gradient(
                90deg,
                #22C55E 0%, #22C55E 12.15%,
                #A3E635 12.15%, #A3E635 28.22%,
                #FACC15 28.22%, #FACC15 59.63%,
                #F97316 59.63%, #F97316 80.74%,
                #C81E1E 80.74%, #C81E1E 100%
            );
        }

        .ramp-ticks {
            position: relative;
            width: 100%;
            height: 14px;
            margin-top: 2px;
            font-family: var(--font-mono);
            font-size: 0.66rem;
            color: var(--text-secondary);
        }

        .ramp-tick {
            position: absolute;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            line-height: 1;
        }

        .ramp-tick:first-child {
            transform: translateX(0%);
        }

        .ramp-tick:last-child {
            transform: translateX(-100%);
        }

        .ramp-tick::before {
            content: '';
            width: 1px;
            height: 4px;
            background: var(--text-muted);
            margin-bottom: 2px;
        }

        .tier-cards {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
        }

        .tier-card {
            background: var(--bg-card-inner);
            border: 1px solid var(--border-subtle);
            border-radius: 7px;
            padding: 6px 10px;
            display: flex;
            flex-direction: column;
            gap: 1px;
        }

        .tier-card-head {
            display: flex;
            align-items: center;
            gap: 7px;
            font-weight: 600;
            font-size: 0.78rem;
            color: #FFFFFF;
        }

        .tier-swatch {
            width: 10px;
            height: 10px;
            border-radius: 2.5px;
            flex-shrink: 0;
        }

        .tier-range {
            font-family: var(--font-mono);
            font-size: 0.69rem;
            color: var(--text-main);
            padding-left: 17px;
        }

        .tier-sub {
            font-size: 0.64rem;
            color: var(--text-muted);
            padding-left: 17px;
            line-height: 1.2;
        }

        /* ---------------- DUAL MAP PANELS ---------------- */
        .maps-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .maps-grid.single-interp {
            grid-template-columns: 1fr;
        }

        .maps-grid.single-interp #panel-nodes {
            display: none;
        }

        .maps-grid.single-nodes {
            grid-template-columns: 1fr;
        }

        .maps-grid.single-nodes #panel-interp {
            display: none;
        }

        .map-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 10px 12px 12px 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .map-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }

        .map-panel-title {
            font-family: var(--font-mono);
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-secondary);
        }

        .map-panel-meta {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .mini-toggle-btn {
            background: var(--bg-card-inner);
            border: 1px solid var(--border-strong);
            color: var(--text-main);
            font-family: var(--font-mono);
            font-size: 0.64rem;
            padding: 2px 7px;
            border-radius: 4px;
            cursor: pointer;
        }

        .mini-toggle-btn:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        .map-viewport {
            position: relative;
            width: 100%;
            height: 360px;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: #E5E7EB;
        }

        .maps-grid.single-interp .map-viewport,
        .maps-grid.single-nodes .map-viewport {
            height: 430px;
        }

        .leaflet-map {
            width: 100%;
            height: 100%;
            background: #E2E8F0 !important;
        }

        /* Leaflet Controls */
        .leaflet-control-zoom {
            border: 1px solid var(--border-strong) !important;
            border-radius: 7px !important;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
        }

        .leaflet-control-zoom a {
            background: rgba(18, 23, 33, 0.92) !important;
            color: #F3F4F6 !important;
            border-bottom: 1px solid var(--border-subtle) !important;
            width: 28px !important;
            height: 28px !important;
            line-height: 28px !important;
            font-family: var(--font-mono) !important;
            font-size: 14px !important;
        }

        .leaflet-control-zoom a:hover {
            background: #1F2937 !important;
            color: #FFFFFF !important;
        }

        .leaflet-control-scale-line {
            background: rgba(15, 23, 42, 0.85) !important;
            border: 1px solid rgba(255, 255, 255, 0.4) !important;
            border-top: none !important;
            color: #F8FAFC !important;
            font-family: var(--font-mono) !important;
            font-size: 10px !important;
            padding: 2px 6px !important;
        }

        .leaflet-control-attribution {
            display: none !important;
        }

        .map-attr-pill {
            position: absolute;
            right: 8px;
            bottom: 8px;
            z-index: 650;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid var(--border-strong);
            border-radius: 5px;
            padding: 3px 8px;
            font-family: var(--font-sans);
            font-size: 0.65rem;
            color: #E2E8F0;
            pointer-events: none;
        }

        .nodes-legend-pill {
            position: absolute;
            right: 8px;
            bottom: 8px;
            z-index: 650;
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid var(--border-strong);
            border-radius: 5px;
            padding: 4px 9px;
            display: flex;
            align-items: center;
            gap: 9px;
            font-size: 0.67rem;
            color: #F8FAFC;
        }

        .nodes-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .nodes-legend-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
        }

        /* Survey Boundary Label */
        .survey-boundary-label span {
            display: inline-block;
            background: rgba(15, 23, 42, 0.88);
            color: #F8FAFC;
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 4px;
            padding: 1px 6px;
            font-family: var(--font-sans);
            font-size: 10.5px;
            font-weight: 600;
            white-space: nowrap;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
            pointer-events: none;
        }

        /* Readout Card (Peak & Hover Cursor) */
        .readout-card {
            background: rgba(13, 17, 24, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 8px;
            padding: 6px 10px;
            min-width: 142px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.55);
            pointer-events: none;
            font-family: var(--font-sans);
        }

        .readout-eyebrow {
            font-family: var(--font-mono);
            font-size: 0.6rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 1px;
        }

        .readout-val {
            font-family: var(--font-mono);
            font-size: 0.98rem;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.2;
        }

        .readout-val small {
            font-size: 0.68rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .readout-sub {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.66rem;
            color: #E5E7EB;
            margin-top: 2px;
            white-space: nowrap;
        }

        .readout-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        /* Node value pill on Measurement Nodes map */
        .node-val-label {
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 4px;
            padding: 1px 5px;
            font-family: var(--font-mono);
            font-size: 9.5px;
            font-weight: 700;
            color: #F8FAFC;
            white-space: nowrap;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.45);
            pointer-events: none;
        }

        /* 40px Cluster Badge from Image 1 Spec (fill = highest reading inside) */
        .b15-cluster-badge {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            border: 2px solid #0F1217;
            box-shadow: 0 0 0 1.5px rgba(255, 255, 255, 0.8), 0 4px 10px rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--font-mono);
            font-size: 11.5px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.15s ease;
        }

        .b15-cluster-badge:hover {
            transform: scale(1.1);
        }

        /* ---------------- BOTTOM 3 STATISTICS CARDS ---------------- */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1.15fr 1fr;
            gap: 12px;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 11px 15px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 8px;
        }

        .stat-card-title {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: var(--text-secondary);
        }

        .field-metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px 16px;
        }

        .field-metric-label {
            font-size: 0.68rem;
            color: var(--text-secondary);
        }

        .field-metric-val {
            font-family: var(--font-mono);
            font-size: 1.28rem;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.15;
        }

        .stat-card-foot {
            font-size: 0.66rem;
            color: var(--text-muted);
        }

        /* Risk Distribution Card */
        .risk-bar-stack {
            height: 8px;
            width: 100%;
            border-radius: 999px;
            overflow: hidden;
            display: flex;
            background: var(--bg-card-inner);
        }

        .risk-bar-seg {
            height: 100%;
            transition: width 0.3s ease;
        }

        .risk-list {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }

        .risk-row {
            display: grid;
            grid-template-columns: 1fr auto 48px;
            align-items: center;
            gap: 8px;
            font-size: 0.72rem;
        }

        .risk-row-name {
            display: flex;
            align-items: center;
            gap: 7px;
            color: var(--text-main);
        }

        .risk-row-count {
            font-family: var(--font-mono);
            color: var(--text-secondary);
            text-align: right;
        }

        .risk-row-pct {
            font-family: var(--font-mono);
            color: var(--text-muted);
            text-align: right;
        }

        /* Annual Dose at Peak Card */
        .peak-annual-val {
            font-family: var(--font-mono);
            font-size: 1.55rem;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.1;
        }

        .peak-annual-val small {
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .peak-annual-tier {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.69rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        .limit-bars {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .limit-row-head {
            display: flex;
            justify-content: space-between;
            font-size: 0.69rem;
            color: var(--text-secondary);
            margin-bottom: 2px;
        }

        .limit-row-head strong {
            font-family: var(--font-mono);
            color: var(--text-main);
            font-weight: 600;
        }

        .limit-track {
            width: 100%;
            height: 5px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
        }

        .limit-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.3s ease;
        }

        /* Dark Leaflet Popup */
        .leaflet-popup-content-wrapper {
            background: #121721 !important;
            color: #F3F4F6 !important;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.65) !important;
        }

        .leaflet-popup-tip {
            background: #121721 !important;
        }

        @media (max-width: 1024px) {
            .tier-cards {
                grid-template-columns: repeat(3, 1fr);
            }
            .maps-grid {
                grid-template-columns: 1fr;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="b15-wrapper" id="b15RootWrapper">
        <!-- HEADER BAR -->
        <header class="b15-header">
            <div class="b15-title-group">
                <h1 class="b15-title">Radiation dose map</h1>
                <div class="b15-eyebrow">
                    <span>ISOTOPER B15</span>
                    <button class="b15-badge" id="btnSourceToggle" title="Click to toggle between Full Google Sheet GM Data and 48-pt Sample Field">
                        <span class="b15-badge-dot" id="sourceDot"></span>
                        <span id="sourceBadgeText">FULL GM SHEET</span>
                    </button>
                </div>
            </div>

            <div class="b15-controls">
                <div class="seg-group" role="group" aria-label="Color scale mode">
                    <button class="seg-btn active" id="btnModeCont" onclick="setScaleMode('continuous')">Continuous</button>
                    <button class="seg-btn" id="btnModeStep" onclick="setScaleMode('stepped')">Stepped tiers</button>
                </div>

                <label class="chk-pill">
                    <input type="checkbox" id="chkHeat" checked onchange="toggleLayerVisibility()" />
                    <span class="chk-box">✓</span>
                    <span>Heat field</span>
                </label>

                <label class="chk-pill">
                    <input type="checkbox" id="chkPoints" checked onchange="toggleLayerVisibility()" />
                    <span class="chk-box">✓</span>
                    <span>Sample points</span>
                </label>

                <label class="chk-pill">
                    <input type="checkbox" id="chkPath" checked onchange="toggleLayerVisibility()" />
                    <span class="chk-box">✓</span>
                    <span>Flight path</span>
                </label>

                <div class="seg-group" role="group" aria-label="Basemap tile style">
                    <button class="seg-btn active" id="btnTileOsm" onclick="setBasemapStyle('osm')">OSM</button>
                    <button class="seg-btn" id="btnTileSat" onclick="setBasemapStyle('sat')">Satellite</button>
                </div>
            </div>
        </header>

        <!-- USER SURVEY BOUNDARY (LAT/LON) INPUT & SYNC BAR -->
        <section class="coord-bar" aria-label="Survey Area Coordinate Selection">
            <div class="coord-bar-left">
                <div class="coord-tag">
                    <span>⌖ SURVEY AREA (START → END)</span>
                </div>
                <div class="coord-group" title="Start Corner Latitude (South/West)">
                    <span>Lat 1</span>
                    <input type="number" step="0.0001" id="inLat1" class="coord-input" value="23.837700" />
                </div>
                <div class="coord-group" title="Start Corner Longitude (South/West)">
                    <span>Lon 1</span>
                    <input type="number" step="0.0001" id="inLon1" class="coord-input" value="90.357000" />
                </div>
                <div class="coord-group" title="End Corner Latitude (North/East)">
                    <span>Lat 2</span>
                    <input type="number" step="0.0001" id="inLat2" class="coord-input" value="23.838500" />
                </div>
                <div class="coord-group" title="End Corner Longitude (North/East)">
                    <span>Lon 2</span>
                    <input type="number" step="0.0001" id="inLon2" class="coord-input" value="90.357900" />
                </div>
            </div>

            <div class="coord-actions">
                <button class="action-btn primary" onclick="applyUserCoordinates()">
                    <span>▶ Apply &amp; Interpolate</span>
                </button>
                <button class="action-btn" id="btnPickCorners" onclick="startCornerPicking()">
                    <span>✥ Pick 2 Corners</span>
                </button>
                <button class="action-btn" onclick="resetMistCoordinates()">
                    <span>↺ MIST Default</span>
                </button>
                <button class="action-btn" id="btnSyncSheet" onclick="fetchFullGoogleSheet(true)">
                    <span>⟳ Sync Sheet</span>
                </button>
                <div class="feed-status" id="feedStatusText">
                    Ready
                </div>
            </div>
        </section>

        <!-- DOSE RATE LOG SCALE LEGEND & 5 TIER CARDS -->
        <section class="scale-card">
            <div class="scale-header">
                <div class="scale-title">DOSE RATE · µSV/H · LOG SCALE</div>
                <div class="scale-note">
                    Annual figures assume 8 h/day, 250 days (2,000 h/yr) · Public limit 1 mSv/yr · Worker limit 20 mSv/yr
                </div>
            </div>

            <div class="ramp-wrap">
                <div class="ramp-bar" id="rampBar"></div>
                <div class="ramp-ticks">
                    <div class="ramp-tick" style="left: 0%;">0.1</div>
                    <div class="ramp-tick" style="left: 12.15%;">0.2</div>
                    <div class="ramp-tick" style="left: 28.22%;">0.5</div>
                    <div class="ramp-tick" style="left: 40.37%;">1</div>
                    <div class="ramp-tick" style="left: 59.63%;">3</div>
                    <div class="ramp-tick" style="left: 80.74%;">10</div>
                    <div class="ramp-tick" style="left: 100%;">30+</div>
                </div>
            </div>

            <div class="tier-cards" id="tierCardsContainer"></div>
        </section>

        <!-- DUAL GEOSPATIAL MAPS -->
        <section class="maps-grid" id="mapsGridContainer">
            <!-- LEFT: INTERPOLATED DISTRIBUTION -->
            <div class="map-panel" id="panel-interp">
                <div class="map-panel-header">
                    <div class="map-panel-title">INTERPOLATED DISTRIBUTION</div>
                    <div class="map-panel-meta" id="interpMetaLabel">56 × 38 grid · continuous log ramp</div>
                </div>
                <div class="map-viewport">
                    <div id="mapInterp" class="leaflet-map"></div>
                    <div class="map-attr-pill" id="interpBasemapAttr">Leaflet · OpenStreetMap Standard</div>
                </div>
            </div>

            <!-- RIGHT: MEASUREMENT NODES -->
            <div class="map-panel" id="panel-nodes">
                <div class="map-panel-header">
                    <div class="map-panel-title">MEASUREMENT NODES</div>
                    <div class="map-panel-meta">
                        <span id="nodesMetaLabel">-- GM-tube readings · tier colors</span>
                        <button class="mini-toggle-btn" id="btnClusterToggle" onclick="toggleNodeClusterMode()">
                            Mode: All Nodes
                        </button>
                    </div>
                </div>
                <div class="map-viewport">
                    <div id="mapNodes" class="leaflet-map"></div>
                    <div class="nodes-legend-pill" id="nodesLegendPill"></div>
                </div>
            </div>
        </section>

        <!-- BOTTOM 3 STATISTICS CARDS -->
        <section class="stats-grid">
            <!-- CARD 1: FIELD STATISTICS -->
            <div class="stat-card">
                <div class="stat-card-title">FIELD STATISTICS</div>
                <div class="field-metrics-grid">
                    <div>
                        <div class="field-metric-label">Average</div>
                        <div class="field-metric-val" id="statAvg">--</div>
                    </div>
                    <div>
                        <div class="field-metric-label">Maximum</div>
                        <div class="field-metric-val" id="statMax">--</div>
                    </div>
                    <div>
                        <div class="field-metric-label">Minimum</div>
                        <div class="field-metric-val" id="statMin">--</div>
                    </div>
                    <div>
                        <div class="field-metric-label">Std deviation</div>
                        <div class="field-metric-val" id="statStd">--</div>
                    </div>
                </div>
                <div class="stat-card-foot" id="statCellsFoot">
                    µSv/h across 2,128 grid cells inside the survey boundary
                </div>
            </div>

            <!-- CARD 2: RISK DISTRIBUTION -->
            <div class="stat-card">
                <div class="stat-card-title">RISK DISTRIBUTION</div>
                <div class="risk-bar-stack" id="riskBarStack"></div>
                <div class="risk-list" id="riskListContainer"></div>
            </div>

            <!-- CARD 3: ANNUAL DOSE AT PEAK -->
            <div class="stat-card">
                <div>
                    <div class="stat-card-title">ANNUAL DOSE AT PEAK</div>
                    <div class="peak-annual-val" style="margin-top: 4px;">
                        <span id="peakAnnualNum">--</span> <small>mSv/yr</small>
                    </div>
                    <div class="peak-annual-tier">
                        <span class="readout-dot" id="peakAnnualDot"></span>
                        <span id="peakAnnualTierTxt">-- · If occupied 8 h/day, 250 days/yr</span>
                    </div>
                </div>

                <div class="limit-bars">
                    <div>
                        <div class="limit-row-head">
                            <span>Public limit · 1 mSv/yr</span>
                            <strong id="publicLimitRatio">--</strong>
                        </div>
                        <div class="limit-track">
                            <div class="limit-fill" id="publicLimitBar" style="width: 0%;"></div>
                        </div>
                    </div>

                    <div>
                        <div class="limit-row-head">
                            <span>Worker limit · 20 mSv/yr</span>
                            <strong id="workerLimitRatio">--</strong>
                        </div>
                        <div class="limit-track">
                            <div class="limit-fill" id="workerLimitBar" style="width: 0%;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </div>

    <script>
        // =========================================================================
        // 1. INITIAL PAYLOAD & UNIFIED CONFIGURATION
        // =========================================================================
        const INITIAL = __INITIAL_PAYLOAD_JSON__;
        const CFG = INITIAL.config;
        const STORAGE_KEY = 'isotoper_b15_survey_state_v4';

        const urlParams = new URLSearchParams(window.location.search);
        const requestedLayout = urlParams.get('layout') || INITIAL.defaultView || 'split';

        function expandCompactReadings(compactList) {
            return (compactList || []).map((arr, idx) => ({
                id: idx + 1,
                dose: arr[0],
                cpm: arr[1],
                cps: arr[2],
                timestamp: arr[3] || '',
            }));
        }

        let state = {
            dataSource: 'live', // 'live' (Full Google Sheet 1,082 valid rows) or 'sample' (48-node spec)
            scaleMode: 'continuous', // 'continuous' or 'stepped'
            nodeDisplayMode: 'all', // 'all' or 'clusters'
            basemap: 'osm', // Default to OpenStreetMap (100% free, zero API key required)
            showHeat: true,
            showPoints: true,
            showPath: true,
            bounds: { ...INITIAL.defaultBounds },
            liveReadings: expandCompactReadings(INITIAL.compactReadings),
            rawCount: INITIAL.rawCount || 0,
            zeroFiltered: INITIAL.zeroFilteredCount || 0,
            nodes: [],
            grid: [],
            peakCell: null,
            pickingStep: 0,
            tempCorner1: null,
        };

        try {
            const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
            if (saved && saved.bounds) {
                state.bounds = saved.bounds;
            }
            if (saved && saved.dataSource) {
                state.dataSource = saved.dataSource;
            }
        } catch (e) {}

        // Notify parent iframe in index.html of exact content height so scrollbar never appears
        function notifyParentHeight() {
            try {
                if (window.parent && window.parent !== window) {
                    const wrapper = document.getElementById('b15RootWrapper');
                    const h = Math.ceil((wrapper ? wrapper.getBoundingClientRect().height : document.body.scrollHeight) + 28);
                    window.parent.postMessage({
                        type: 'b15-iframe-resize',
                        layout: requestedLayout,
                        height: h,
                    }, '*');
                }
            } catch (e) {}
        }

        // =========================================================================
        // 2. COLOR & TIER FUNCTIONS (LOG10 AXIS 0.1 TO 30 µSV/H)
        // =========================================================================
        function getTier(dose) {
            for (const t of CFG.tiers) {
                if (dose < t.max) return t;
            }
            return CFG.tiers[CFG.tiers.length - 1];
        }

        function hexToRgb(hex) {
            const h = hex.replace('#', '');
            return [
                parseInt(h.substring(0, 2), 16),
                parseInt(h.substring(2, 4), 16),
                parseInt(h.substring(4, 6), 16),
            ];
        }

        function doseToRgb(dose, mode) {
            if (mode === 'stepped') {
                return hexToRgb(getTier(dose).color);
            }
            const clamped = Math.max(CFG.minLogDose, Math.min(CFG.maxLogDose, dose));
            const pos = Math.log10(clamped / 0.1) / Math.log10(300.0);
            const stops = CFG.rampStops;
            for (let i = 0; i < stops.length - 1; i++) {
                const s0 = stops[i];
                const s1 = stops[i + 1];
                if (pos <= s1.pos || i === stops.length - 2) {
                    const span = s1.pos - s0.pos;
                    const t = span <= 0 ? 0 : Math.max(0, Math.min(1, (pos - s0.pos) / span));
                    const c0 = hexToRgb(s0.color);
                    const c1 = hexToRgb(s1.color);
                    return [
                        Math.round(c0[0] + (c1[0] - c0[0]) * t),
                        Math.round(c0[1] + (c1[1] - c0[1]) * t),
                        Math.round(c0[2] + (c1[2] - c0[2]) * t),
                    ];
                }
            }
            return hexToRgb(stops[stops.length - 1].color);
        }

        function formatDose(v) {
            if (v >= 10) return v.toFixed(1);
            if (v >= 1) return v.toFixed(2);
            return v.toFixed(3);
        }

        // =========================================================================
        // 3. BUILD STATIC UI ELEMENTS FROM CONFIG
        // =========================================================================
        function initStaticUI() {
            const gridEl = document.getElementById('mapsGridContainer');
            if (requestedLayout === 'interpolated') {
                gridEl.classList.add('single-interp');
            } else if (requestedLayout === 'nodes') {
                gridEl.classList.add('single-nodes');
            }

            document.getElementById('inLat1').value = Number(state.bounds.lat1).toFixed(6);
            document.getElementById('inLon1').value = Number(state.bounds.lon1).toFixed(6);
            document.getElementById('inLat2').value = Number(state.bounds.lat2).toFixed(6);
            document.getElementById('inLon2').value = Number(state.bounds.lon2).toFixed(6);

            const tierContainer = document.getElementById('tierCardsContainer');
            tierContainer.innerHTML = CFG.tiers.map(t => `
                <div class="tier-card">
                    <div class="tier-card-head">
                        <span class="tier-swatch" style="background:${t.color};"></span>
                        <span>${t.name}</span>
                    </div>
                    <div class="tier-range">${t.rangeLabel}</div>
                    <div class="tier-sub">${t.annualLabel} · ${t.shortMeaning}</div>
                </div>
            `).join('');

            const pill = document.getElementById('nodesLegendPill');
            pill.innerHTML = CFG.tiers.map(t => `
                <span class="nodes-legend-item">
                    <span class="nodes-legend-dot" style="background:${t.color};"></span>
                    <span>${t.name}</span>
                </span>
            `).join('');

            document.getElementById('btnSourceToggle').addEventListener('click', () => {
                state.dataSource = state.dataSource === 'live' ? 'sample' : 'live';
                saveState();
                recomputeAndRender(false);
            });

            document.getElementById('feedStatusText').textContent =
                `${state.liveReadings.length.toLocaleString()} valid pts (${state.zeroFiltered} zeros filtered)`;
        }

        // =========================================================================
        // 4. LEAFLET DUAL MAPS INITIALIZATION (100% FREE OSM DEFAULT, NO API KEY)
        // =========================================================================
        const TILE_URLS = {
            osm: {
                url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attr: 'Leaflet · OpenStreetMap Standard',
                maxNativeZoom: 19,
                maxZoom: 22,
            },
            sat: {
                url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr: 'Leaflet · Esri World Imagery Satellite',
                maxNativeZoom: 19,
                maxZoom: 22,
            },
        };

        let mapInterp, mapNodes;
        let tileLayerInterp, tileLayerNodes;
        let heatOverlayInterp = null;
        let boundaryLayerInterp, boundaryLayerNodes;
        let pathLayerInterp, pathLayerNodes;
        let pointsLayerInterp, pointsLayerNodes;
        let labelsLayerNodes;
        let peakLayerInterp;
        let cursorMarkerInterp = null;
        let isSyncingMaps = false;

        function initMaps() {
            const centerLat = (state.bounds.lat1 + state.bounds.lat2) / 2;
            const centerLon = (state.bounds.lon1 + state.bounds.lon2) / 2;

            mapInterp = L.map('mapInterp', {
                center: [centerLat, centerLon],
                zoom: 18,
                maxZoom: 22,
                zoomControl: true,
                attributionControl: false,
                preferCanvas: true,
            });

            mapNodes = L.map('mapNodes', {
                center: [centerLat, centerLon],
                zoom: 18,
                maxZoom: 22,
                zoomControl: true,
                attributionControl: false,
                preferCanvas: true,
            });

            L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(mapInterp);
            L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(mapNodes);

            tileLayerInterp = L.tileLayer(TILE_URLS.osm.url, {
                maxNativeZoom: TILE_URLS.osm.maxNativeZoom,
                maxZoom: TILE_URLS.osm.maxZoom,
            }).addTo(mapInterp);

            tileLayerNodes = L.tileLayer(TILE_URLS.osm.url, {
                maxNativeZoom: TILE_URLS.osm.maxNativeZoom,
                maxZoom: TILE_URLS.osm.maxZoom,
            }).addTo(mapNodes);

            boundaryLayerInterp = L.layerGroup().addTo(mapInterp);
            boundaryLayerNodes = L.layerGroup().addTo(mapNodes);
            pathLayerInterp = L.layerGroup().addTo(mapInterp);
            pathLayerNodes = L.layerGroup().addTo(mapNodes);
            pointsLayerInterp = L.layerGroup().addTo(mapInterp);
            pointsLayerNodes = L.layerGroup().addTo(mapNodes);
            labelsLayerNodes = L.layerGroup().addTo(mapNodes);
            peakLayerInterp = L.layerGroup().addTo(mapInterp);

            function syncMap(source, target) {
                source.on('moveend zoomend', () => {
                    if (isSyncingMaps) return;
                    isSyncingMaps = true;
                    target.setView(source.getCenter(), source.getZoom(), { animate: false });
                    isSyncingMaps = false;
                });
            }
            syncMap(mapInterp, mapNodes);
            syncMap(mapNodes, mapInterp);

            mapNodes.on('zoomend', () => {
                if (state.nodeDisplayMode === 'all') {
                    renderHotspotLabelsOnNodesMap();
                }
            });

            mapInterp.on('mousemove', onInterpMouseMove);
            mapInterp.on('mouseout', () => {
                if (cursorMarkerInterp) {
                    mapInterp.removeLayer(cursorMarkerInterp);
                    cursorMarkerInterp = null;
                }
            });

            mapInterp.on('click', onMapClickForCorner);
            mapNodes.on('click', onMapClickForCorner);

            if (typeof ResizeObserver !== 'undefined') {
                let resizeTimer = null;
                const ro = new ResizeObserver(() => {
                    notifyParentHeight();
                    clearTimeout(resizeTimer);
                    resizeTimer = setTimeout(() => {
                        mapInterp.invalidateSize();
                        mapNodes.invalidateSize();
                    }, 80);
                });
                ro.observe(document.body);
            }
        }

        // =========================================================================
        // 5. SPATIAL INTERPOLATION ENGINE (HOTSPOT-PRESERVING 56 × 38 GRID)
        // =========================================================================
        function getActiveReadings() {
            if (state.dataSource === 'sample') {
                return INITIAL.sampleFieldDoses.map((d, idx) => ({
                    id: idx + 1,
                    dose: d,
                    cpm: Math.round((d / 0.00812) * 10) / 10,
                    cps: Math.round(((d / 0.00812) / 60) * 100) / 100,
                    timestamp: 'Reference Survey Field',
                }));
            }
            return state.liveReadings;
        }

        function computeNodesAndGrid() {
            const readings = getActiveReadings();
            const n = readings.length;
            if (n === 0) return;

            const latMin = Math.min(state.bounds.lat1, state.bounds.lat2);
            const latMax = Math.max(state.bounds.lat1, state.bounds.lat2);
            const lonMin = Math.min(state.bounds.lon1, state.bounds.lon2);
            const lonMax = Math.max(state.bounds.lon1, state.bounds.lon2);

            const latSpan = Math.max(1e-6, latMax - latMin);
            const lonSpan = Math.max(1e-6, lonMax - lonMin);

            const rows = state.dataSource === 'sample'
                ? 6
                : Math.max(2, Math.round(Math.sqrt(n * 0.62)));
            const cols = Math.max(2, Math.ceil(n / rows));

            const padLat = latSpan * 0.05;
            const padLon = lonSpan * 0.05;
            const innerLatMin = latMin + padLat;
            const innerLatMax = latMax - padLat;
            const innerLonMin = lonMin + padLon;
            const innerLonMax = lonMax - padLon;

            const nodes = [];
            for (let idx = 0; idx < n; idx++) {
                const r = readings[idx];
                const rowIdx = Math.floor(idx / cols);
                const colInRow = idx % cols;
                const ptsInRow = Math.min(cols, n - rowIdx * cols);

                const latT = rows <= 1 ? 0.5 : rowIdx / (rows - 1);
                const lat = innerLatMax - latT * (innerLatMax - innerLatMin);

                let lonT = ptsInRow <= 1 ? 0.5 : colInRow / (ptsInRow - 1);
                if (rowIdx % 2 === 1) lonT = 1.0 - lonT;
                const lon = innerLonMin + lonT * (innerLonMax - innerLonMin);

                const tier = getTier(r.dose);
                nodes.push({
                    id: idx + 1,
                    rowIdx,
                    colInRow,
                    lat,
                    lon,
                    dose: r.dose,
                    cpm: r.cpm,
                    cps: r.cps,
                    timestamp: r.timestamp,
                    tier,
                    annualMSv: r.dose * (CFG.annualHours / 1000.0),
                });
            }
            state.nodes = nodes;

            // Local hotspot-preserving IDW + Gaussian interpolation onto 56 × 38 grid
            const G_ROWS = CFG.gridRows;
            const G_COLS = CFG.gridCols;
            const grid = [];
            let maxDose = -1;
            let peakCell = null;

            const doses = new Float64Array(n);
            const normLats = new Float64Array(n);
            const normLons = new Float64Array(n);
            for (let k = 0; k < n; k++) {
                doses[k] = Math.max(0.01, nodes[k].dose);
                normLats[k] = (nodes[k].lat - latMin) / latSpan;
                normLons[k] = (nodes[k].lon - lonMin) / lonSpan;
            }

            const sigma = n > 200 ? 0.028 : 0.11;
            const sigma2 = 2.0 * sigma * sigma;
            const eps = n > 200 ? 0.00003 : 0.0006;

            for (let r = 0; r < G_ROWS; r++) {
                const rowArr = [];
                const cellLat = latMax - ((r + 0.5) / G_ROWS) * latSpan;
                const ny = (cellLat - latMin) / latSpan;

                for (let c = 0; c < G_COLS; c++) {
                    const cellLon = lonMin + ((c + 0.5) / G_COLS) * lonSpan;
                    const nx = (cellLon - lonMin) / lonSpan;

                    let wSum = 0;
                    let vSum = 0;
                    for (let k = 0; k < n; k++) {
                        const dx = nx - normLons[k];
                        const dy = ny - normLats[k];
                        const d2 = dx * dx + dy * dy;
                        if (n > 200 && d2 > 0.035) continue;
                        const w = Math.exp(-d2 / sigma2) + 1.0 / Math.pow(d2 + eps, 2.15);
                        wSum += w;
                        vSum += w * doses[k];
                    }
                    const dose = wSum > 0 ? (vSum / wSum) : 0.15;
                    rowArr.push(dose);

                    if (dose > maxDose) {
                        maxDose = dose;
                        peakCell = {
                            row: r + 1,
                            col: c + 1,
                            lat: cellLat,
                            lon: cellLon,
                            dose,
                            tier: getTier(dose),
                            annualMSv: dose * (CFG.annualHours / 1000.0),
                        };
                    }
                }
                grid.push(rowArr);
            }

            state.grid = grid;
            state.peakCell = peakCell;
        }

        // =========================================================================
        // 6. RENDER HEAT OVERLAY, SURVEY BOUNDARY, FLIGHT PATH, NODES & CLUSTERS
        // =========================================================================
        function renderHeatCanvasDataUrl() {
            const G_ROWS = CFG.gridRows;
            const G_COLS = CFG.gridCols;
            const scale = 6;
            const width = G_COLS * scale;
            const height = G_ROWS * scale;

            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            const imgData = ctx.createImageData(width, height);
            const data = imgData.data;

            for (let py = 0; py < height; py++) {
                const gy = (py / (height - 1)) * (G_ROWS - 1);
                const r0 = Math.floor(gy);
                const r1 = Math.min(G_ROWS - 1, r0 + 1);
                const ty = gy - r0;

                for (let px = 0; px < width; px++) {
                    const gx = (px / (width - 1)) * (G_COLS - 1);
                    const c0 = Math.floor(gx);
                    const c1 = Math.min(G_COLS - 1, c0 + 1);
                    const tx = gx - c0;

                    const v00 = state.grid[r0][c0];
                    const v10 = state.grid[r0][c1];
                    const v01 = state.grid[r1][c0];
                    const v11 = state.grid[r1][c1];
                    const dose =
                        (1 - tx) * (1 - ty) * v00 +
                        tx * (1 - ty) * v10 +
                        (1 - tx) * ty * v01 +
                        tx * ty * v11;

                    const rgb = doseToRgb(dose, state.scaleMode);
                    const idx = (py * width + px) * 4;
                    data[idx] = rgb[0];
                    data[idx + 1] = rgb[1];
                    data[idx + 2] = rgb[2];
                    data[idx + 3] = 206; // ~81% opacity
                }
            }
            ctx.putImageData(imgData, 0, 0);
            return canvas.toDataURL('image/png');
        }

        function bindNodePopup(marker, pt) {
            const popupHtml = `
                <div style="font-family:var(--font-sans);min-width:210px;padding:2px;">
                    <div style="font-family:var(--font-mono);font-size:0.68rem;color:#9CA3AF;text-transform:uppercase;">
                        NODE #${pt.id} · ${pt.tier.name.toUpperCase()}
                    </div>
                    <div style="font-family:var(--font-mono);font-size:1.12rem;font-weight:700;color:#FFF;margin:3px 0;">
                        ${formatDose(pt.dose)} <span style="font-size:0.74rem;color:#9CA3AF;">µSv/h</span>
                    </div>
                    <div style="font-size:0.73rem;color:${pt.tier.color};font-weight:600;margin-bottom:5px;">
                        ● ${pt.tier.name} · ${pt.annualMSv.toFixed(2)} mSv/yr
                    </div>
                    <div style="font-family:var(--font-mono);font-size:0.69rem;color:#D1D5DB;line-height:1.45;border-top:1px solid rgba(255,255,255,0.1);padding-top:4px;">
                        <div>CPM: ${pt.cpm} | CPS: ${pt.cps}</div>
                        ${pt.timestamp ? `<div>Time: ${pt.timestamp}</div>` : ''}
                        <div>Lat: ${pt.lat.toFixed(6)}°N</div>
                        <div>Lon: ${pt.lon.toFixed(6)}°E</div>
                    </div>
                </div>
            `;
            marker.bindPopup(popupHtml);
        }

        function renderHotspotLabelsOnNodesMap() {
            labelsLayerNodes.clearLayers();
            if (state.nodeDisplayMode !== 'all' || state.nodes.length === 0) return;

            const sortedHot = [...state.nodes]
                .filter(pt => pt.dose >= 0.5)
                .sort((a, b) => b.dose - a.dose);

            const placedBoxes = [];
            for (const pt of sortedHot) {
                if (placedBoxes.length >= 14) break;
                const pxPt = mapNodes.latLngToContainerPoint([pt.lat, pt.lon]);
                const collide = placedBoxes.some(
                    b => Math.abs(b.x - pxPt.x) < 58 && Math.abs(b.y - pxPt.y) < 22
                );
                if (collide) continue;
                placedBoxes.push(pxPt);

                L.marker([pt.lat, pt.lon], {
                    icon: L.divIcon({
                        className: '',
                        html: `<span class="node-val-label">${formatDose(pt.dose)}</span>`,
                        iconSize: [42, 16],
                        iconAnchor: [-7, 8],
                    }),
                    interactive: false,
                }).addTo(labelsLayerNodes);
            }
        }

        function renderMeasurementNodesLayer(latMin, latMax, lonMin, lonMax) {
            pointsLayerNodes.clearLayers();
            labelsLayerNodes.clearLayers();
            const n = state.nodes.length;
            if (n === 0) return;

            if (state.nodeDisplayMode === 'clusters') {
                const cRows = 5;
                const cCols = 7;
                const latStep = (latMax - latMin) / cRows;
                const lonStep = (lonMax - lonMin) / cCols;
                const buckets = new Map();

                state.nodes.forEach(pt => {
                    const br = Math.min(cRows - 1, Math.max(0, Math.floor((latMax - pt.lat) / latStep)));
                    const bc = Math.min(cCols - 1, Math.max(0, Math.floor((pt.lon - lonMin) / lonStep)));
                    const key = `${br}_${bc}`;
                    if (!buckets.has(key)) {
                        buckets.set(key, { count: 0, maxDose: -1, maxPt: null, sumLat: 0, sumLon: 0 });
                    }
                    const b = buckets.get(key);
                    b.count++;
                    b.sumLat += pt.lat;
                    b.sumLon += pt.lon;
                    if (pt.dose > b.maxDose) {
                        b.maxDose = pt.dose;
                        b.maxPt = pt;
                    }
                });

                buckets.forEach(b => {
                    const tier = getTier(b.maxDose);
                    const cLat = b.sumLat / b.count;
                    const cLon = b.sumLon / b.count;
                    const badgeHtml = `
                        <div class="b15-cluster-badge" style="background:${tier.color};color:${tier.textColor};" title="Cluster (${b.count} readings) · Peak: ${formatDose(b.maxDose)} µSv/h (${tier.name})">
                            ${b.count}
                        </div>
                    `;
                    const m = L.marker([cLat, cLon], {
                        icon: L.divIcon({
                            className: '',
                            html: badgeHtml,
                            iconSize: [38, 38],
                            iconAnchor: [19, 19],
                        }),
                    }).addTo(pointsLayerNodes);
                    bindNodePopup(m, b.maxPt);
                });
                return;
            }

            // Plot all 1,082 discrete points; draw Safe/Low first, then Moderate/High/Severe on top
            const isDense = n > 200;
            const orderedNodes = [...state.nodes].sort((a, b) => a.dose - b.dose);

            orderedNodes.forEach(pt => {
                const isHot = pt.dose >= 0.5;
                const r = isDense ? (pt.dose >= 10 ? 6.2 : (isHot ? 5.0 : 3.8)) : 6.8;

                if (!isDense || isHot) {
                    L.circleMarker([pt.lat, pt.lon], {
                        radius: r + 1.0,
                        color: 'rgba(15, 23, 42, 0.75)',
                        weight: 1.2,
                        fill: false,
                        interactive: false,
                    }).addTo(pointsLayerNodes);
                }

                const marker = L.circleMarker([pt.lat, pt.lon], {
                    radius: r,
                    color: '#0F172A',
                    weight: isDense ? 1.2 : 1.8,
                    fillColor: pt.tier.color,
                    fillOpacity: 0.96,
                }).addTo(pointsLayerNodes);

                bindNodePopup(marker, pt);
            });

            renderHotspotLabelsOnNodesMap();
        }

        function renderMaps(fitCamera = false) {
            const latMin = Math.min(state.bounds.lat1, state.bounds.lat2);
            const latMax = Math.max(state.bounds.lat1, state.bounds.lat2);
            const lonMin = Math.min(state.bounds.lon1, state.bounds.lon2);
            const lonMax = Math.max(state.bounds.lon1, state.bounds.lon2);
            const surveyBounds = L.latLngBounds([latMin, lonMin], [latMax, lonMax]);

            // 1. Heat Overlay clipped strictly to survey boundary
            if (heatOverlayInterp) {
                mapInterp.removeLayer(heatOverlayInterp);
                heatOverlayInterp = null;
            }
            if (state.showHeat && state.grid.length > 0) {
                const dataUrl = renderHeatCanvasDataUrl();
                heatOverlayInterp = L.imageOverlay(dataUrl, surveyBounds, {
                    opacity: 1.0,
                    interactive: false,
                }).addTo(mapInterp);
            }

            // 2. Dashed Survey Boundary Box + Label on both maps (high contrast on OSM & Satellite)
            boundaryLayerInterp.clearLayers();
            boundaryLayerNodes.clearLayers();

            [boundaryLayerInterp, boundaryLayerNodes].forEach(layer => {
                L.rectangle(surveyBounds, {
                    color: '#0F172A',
                    weight: 2.0,
                    opacity: 0.85,
                    dashArray: '6, 5',
                    fill: false,
                    interactive: false,
                }).addTo(layer);

                L.marker([latMax, lonMin], {
                    icon: L.divIcon({
                        className: 'survey-boundary-label',
                        html: '<span>Survey boundary</span>',
                        iconSize: [120, 20],
                        iconAnchor: [0, 22],
                    }),
                    interactive: false,
                }).addTo(layer);
            });

            // 3. Dashed Flight Path on both maps
            pathLayerInterp.clearLayers();
            pathLayerNodes.clearLayers();
            if (state.showPath && state.nodes.length > 1) {
                const latlngs = state.nodes.map(pt => [pt.lat, pt.lon]);
                L.polyline(latlngs, {
                    color: '#0F172A',
                    weight: 1.1,
                    opacity: 0.4,
                    dashArray: '3, 5',
                    interactive: false,
                }).addTo(pathLayerInterp);

                L.polyline(latlngs, {
                    color: '#1E293B',
                    weight: 1.2,
                    opacity: 0.55,
                    dashArray: '4, 5',
                    interactive: false,
                }).addTo(pathLayerNodes);
            }

            // 4. Sample Points on Interpolated map
            pointsLayerInterp.clearLayers();
            if (state.showPoints) {
                const isDense = state.nodes.length > 200;
                state.nodes.forEach(pt => {
                    L.circleMarker([pt.lat, pt.lon], {
                        radius: isDense ? 1.8 : 2.6,
                        color: '#0F1217',
                        weight: 0.9,
                        fillColor: '#FFFFFF',
                        fillOpacity: isDense ? 0.72 : 0.95,
                        interactive: false,
                    }).addTo(pointsLayerInterp);
                });
            }

            // 5. Measurement Nodes (All 1,082+ points or 40px Cluster Badges)
            renderMeasurementNodesLayer(latMin, latMax, lonMin, lonMax);

            // 6. Peak Reading Ring (26px) + Readout Card on Interpolated Map
            peakLayerInterp.clearLayers();
            if (state.peakCell) {
                const pk = state.peakCell;
                L.circleMarker([pk.lat, pk.lon], {
                    radius: 12,
                    color: '#FFFFFF',
                    weight: 2.2,
                    fillColor: pk.tier.color,
                    fillOpacity: 0.92,
                    interactive: false,
                }).addTo(peakLayerInterp);

                const cardHtml = `
                    <div class="readout-card">
                        <div class="readout-eyebrow">PEAK READING</div>
                        <div class="readout-val">${formatDose(pk.dose)} <small>µSv/h</small></div>
                        <div class="readout-sub">
                            <span class="readout-dot" style="background:${pk.tier.color};"></span>
                            <span>${pk.tier.name} · ${pk.annualMSv.toFixed(1)} mSv/yr</span>
                        </div>
                    </div>
                `;
                L.marker([pk.lat, pk.lon], {
                    icon: L.divIcon({
                        className: '',
                        html: cardHtml,
                        iconSize: [160, 62],
                        iconAnchor: [172, 31],
                    }),
                    interactive: false,
                }).addTo(peakLayerInterp);
            }

            document.getElementById('interpMetaLabel').textContent =
                `${CFG.gridCols} × ${CFG.gridRows} grid · ${state.scaleMode === 'continuous' ? 'continuous log ramp' : 'stepped tiers'}`;
            document.getElementById('nodesMetaLabel').textContent =
                `${state.nodes.length.toLocaleString()} GM-tube readings · tier colors`;

            if (fitCamera) {
                const padded = surveyBounds.pad(0.14);
                isSyncingMaps = true;
                mapInterp.fitBounds(padded, { animate: false });
                mapNodes.fitBounds(padded, { animate: false });
                setTimeout(() => {
                    isSyncingMaps = false;
                    renderHotspotLabelsOnNodesMap();
                }, 120);
            }
        }

        function toggleNodeClusterMode() {
            state.nodeDisplayMode = state.nodeDisplayMode === 'all' ? 'clusters' : 'all';
            document.getElementById('btnClusterToggle').textContent =
                state.nodeDisplayMode === 'all' ? 'Mode: All Nodes' : 'Mode: 40px Clusters';
            renderMaps(false);
        }

        // =========================================================================
        // 7. HOVER CURSOR READOUT CARD ON INTERPOLATED GRID
        // =========================================================================
        function onInterpMouseMove(e) {
            if (!state.grid || state.grid.length === 0) return;
            const latMin = Math.min(state.bounds.lat1, state.bounds.lat2);
            const latMax = Math.max(state.bounds.lat1, state.bounds.lat2);
            const lonMin = Math.min(state.bounds.lon1, state.bounds.lon2);
            const lonMax = Math.max(state.bounds.lon1, state.bounds.lon2);

            const { lat, lng } = e.latlng;
            if (lat < latMin || lat > latMax || lng < lonMin || lng > lonMax) {
                if (cursorMarkerInterp) {
                    mapInterp.removeLayer(cursorMarkerInterp);
                    cursorMarkerInterp = null;
                }
                return;
            }

            const r = Math.min(
                CFG.gridRows - 1,
                Math.max(0, Math.floor(((latMax - lat) / (latMax - latMin)) * CFG.gridRows))
            );
            const c = Math.min(
                CFG.gridCols - 1,
                Math.max(0, Math.floor(((lng - lonMin) / (lonMax - lonMin)) * CFG.gridCols))
            );

            const dose = state.grid[r][c];
            const tier = getTier(dose);
            const annualMSv = dose * (CFG.annualHours / 1000.0);

            const html = `
                <div style="position:relative;">
                    <div style="width:13px;height:13px;border-radius:50%;border:2px solid #FFF;background:${tier.color};transform:translate(-6.5px,-6.5px);box-shadow:0 0 8px rgba(0,0,0,0.8);"></div>
                    <div class="readout-card" style="position:absolute;left:13px;top:-26px;">
                        <div class="readout-eyebrow">CELL ${c + 1} · ${r + 1}</div>
                        <div class="readout-val">${formatDose(dose)} <small>µSv/h</small></div>
                        <div class="readout-sub">
                            <span class="readout-dot" style="background:${tier.color};"></span>
                            <span>${tier.name} · ${annualMSv.toFixed(2)} mSv/yr</span>
                        </div>
                    </div>
                </div>
            `;

            if (!cursorMarkerInterp) {
                cursorMarkerInterp = L.marker([lat, lng], {
                    icon: L.divIcon({ className: '', html, iconSize: [1, 1], iconAnchor: [0, 0] }),
                    interactive: false,
                    zIndexOffset: 1000,
                }).addTo(mapInterp);
            } else {
                cursorMarkerInterp.setLatLng([lat, lng]);
                cursorMarkerInterp.setIcon(
                    L.divIcon({ className: '', html, iconSize: [1, 1], iconAnchor: [0, 0] })
                );
            }
        }

        // =========================================================================
        // 8. UPDATE BOTTOM 3 STATISTICS CARDS
        // =========================================================================
        function renderStatsCards() {
            const flat = state.grid.flat();
            const totalCells = flat.length;
            if (totalCells === 0) return;

            let sum = 0;
            let minV = Infinity;
            let maxV = -Infinity;
            const counts = { safe: 0, low: 0, moderate: 0, high: 0, severe: 0 };

            for (const v of flat) {
                sum += v;
                if (v < minV) minV = v;
                if (v > maxV) maxV = v;
                const t = getTier(v);
                counts[t.key] = (counts[t.key] || 0) + 1;
            }

            const mean = sum / totalCells;
            let sqDiff = 0;
            for (const v of flat) {
                sqDiff += (v - mean) * (v - mean);
            }
            const std = Math.sqrt(sqDiff / totalCells);

            document.getElementById('statAvg').textContent = formatDose(mean);
            document.getElementById('statMax').textContent = formatDose(maxV);
            document.getElementById('statMin').textContent = formatDose(minV);
            document.getElementById('statStd').textContent = formatDose(std);
            document.getElementById('statCellsFoot').textContent =
                `µSv/h across ${totalCells.toLocaleString()} grid cells (${state.nodes.length.toLocaleString()} GM points inside survey boundary)`;

            const stackEl = document.getElementById('riskBarStack');
            const listEl = document.getElementById('riskListContainer');

            stackEl.innerHTML = CFG.tiers.map(t => {
                const pct = (counts[t.key] / totalCells) * 100;
                return `<div class="risk-bar-seg" style="width:${pct.toFixed(2)}%;background:${t.color};"></div>`;
            }).join('');

            listEl.innerHTML = CFG.tiers.map(t => {
                const cnt = counts[t.key];
                const pct = (cnt / totalCells) * 100;
                return `
                    <div class="risk-row">
                        <div class="risk-row-name">
                            <span class="tier-swatch" style="background:${t.color};"></span>
                            <span>${t.name}</span>
                        </div>
                        <div class="risk-row-count">${cnt}</div>
                        <div class="risk-row-pct">${pct.toFixed(1)}%</div>
                    </div>
                `;
            }).join('');

            const peakAnnualMSv = maxV * (CFG.annualHours / 1000.0);
            const peakTier = getTier(maxV);

            document.getElementById('peakAnnualNum').textContent =
                peakAnnualMSv >= 10 ? peakAnnualMSv.toFixed(1) : peakAnnualMSv.toFixed(2);
            document.getElementById('peakAnnualDot').style.background = peakTier.color;
            document.getElementById('peakAnnualTierTxt').textContent =
                `${peakTier.name} · If occupied 8 h/day, 250 days/yr`;

            const pubRatio = peakAnnualMSv / CFG.publicLimitMSv;
            const pubBar = document.getElementById('publicLimitBar');
            pubBar.style.width = `${Math.min(100, pubRatio * 100).toFixed(1)}%`;
            pubBar.style.background = pubRatio >= 1 ? '#C81E1E' : peakTier.color;
            document.getElementById('publicLimitRatio').textContent =
                pubRatio >= 1
                    ? `Exceeds · ${pubRatio >= 10 ? pubRatio.toFixed(0) : pubRatio.toFixed(1)}×`
                    : `${(pubRatio * 100).toFixed(0)}% of limit`;

            const wrkRatio = peakAnnualMSv / CFG.workerLimitMSv;
            const wrkBar = document.getElementById('workerLimitBar');
            wrkBar.style.width = `${Math.min(100, wrkRatio * 100).toFixed(1)}%`;
            wrkBar.style.background = wrkRatio >= 1 ? '#C81E1E' : peakTier.color;
            document.getElementById('workerLimitRatio').textContent =
                wrkRatio >= 1
                    ? `Exceeds · ${wrkRatio.toFixed(1)}×`
                    : `${(wrkRatio * 100).toFixed(1)}% of limit`;

            const badgeTxt = document.getElementById('sourceBadgeText');
            const badgeDot = document.getElementById('sourceDot');
            if (state.dataSource === 'live') {
                badgeTxt.textContent = `FULL GM SHEET (${state.liveReadings.length.toLocaleString()} PTS)`;
                badgeDot.style.background = '#22C55E';
                badgeDot.style.boxShadow = '0 0 6px #22C55E';
            } else {
                badgeTxt.textContent = 'SAMPLE FIELD (48 PTS)';
                badgeDot.style.background = '#FACC15';
                badgeDot.style.boxShadow = '0 0 6px #FACC15';
            }
        }

        // =========================================================================
        // 9. USER CONTROLS & FULL GOOGLE SHEET LIVE CSV PARSER
        // =========================================================================
        function recomputeAndRender(fitCamera = false) {
            computeNodesAndGrid();
            renderMaps(fitCamera);
            renderStatsCards();
            notifyParentHeight();
        }

        function saveState() {
            try {
                localStorage.setItem(
                    STORAGE_KEY,
                    JSON.stringify({
                        bounds: state.bounds,
                        dataSource: state.dataSource,
                        updatedAt: Date.now(),
                    })
                );
            } catch (e) {}
        }

        function applyUserCoordinates() {
            const lat1 = parseFloat(document.getElementById('inLat1').value);
            const lon1 = parseFloat(document.getElementById('inLon1').value);
            const lat2 = parseFloat(document.getElementById('inLat2').value);
            const lon2 = parseFloat(document.getElementById('inLon2').value);

            if (
                Number.isNaN(lat1) || Number.isNaN(lon1) ||
                Number.isNaN(lat2) || Number.isNaN(lon2) ||
                Math.abs(lat1 - lat2) < 0.00002 ||
                Math.abs(lon1 - lon2) < 0.00002
            ) {
                document.getElementById('feedStatusText').textContent =
                    '⚠ Enter two distinct valid Lat/Lon corners';
                return;
            }

            state.bounds = {
                lat1: Math.min(lat1, lat2),
                lon1: Math.min(lon1, lon2),
                lat2: Math.max(lat1, lat2),
                lon2: Math.max(lon1, lon2),
            };
            saveState();
            recomputeAndRender(true);
            document.getElementById('feedStatusText').textContent =
                `✓ Interpolated ${state.nodes.length.toLocaleString()} pts`;
        }

        function resetMistCoordinates() {
            state.bounds = { ...INITIAL.defaultBounds };
            document.getElementById('inLat1').value = state.bounds.lat1.toFixed(6);
            document.getElementById('inLon1').value = state.bounds.lon1.toFixed(6);
            document.getElementById('inLat2').value = state.bounds.lat2.toFixed(6);
            document.getElementById('inLon2').value = state.bounds.lon2.toFixed(6);
            saveState();
            recomputeAndRender(true);
            document.getElementById('feedStatusText').textContent =
                '✓ Reset to MIST Campus default';
        }

        function startCornerPicking() {
            const btn = document.getElementById('btnPickCorners');
            if (state.pickingStep > 0) {
                state.pickingStep = 0;
                btn.classList.remove('picking');
                btn.innerHTML = '<span>✥ Pick 2 Corners</span>';
                document.getElementById('feedStatusText').textContent = 'Picking cancelled';
                return;
            }
            state.pickingStep = 1;
            btn.classList.add('picking');
            btn.innerHTML = '<span>Click Corner 1...</span>';
            document.getElementById('feedStatusText').textContent =
                'Click first corner (Start Lat/Lon) on map';
        }

        function onMapClickForCorner(e) {
            if (state.pickingStep === 0) return;
            const btn = document.getElementById('btnPickCorners');

            if (state.pickingStep === 1) {
                state.tempCorner1 = e.latlng;
                document.getElementById('inLat1').value = e.latlng.lat.toFixed(6);
                document.getElementById('inLon1').value = e.latlng.lng.toFixed(6);
                state.pickingStep = 2;
                btn.innerHTML = '<span>Click Corner 2...</span>';
                document.getElementById('feedStatusText').textContent =
                    'Now click opposite corner (End Lat/Lon)';
            } else if (state.pickingStep === 2) {
                document.getElementById('inLat2').value = e.latlng.lat.toFixed(6);
                document.getElementById('inLon2').value = e.latlng.lng.toFixed(6);
                state.pickingStep = 0;
                btn.classList.remove('picking');
                btn.innerHTML = '<span>✥ Pick 2 Corners</span>';
                applyUserCoordinates();
            }
        }

        function setScaleMode(mode) {
            state.scaleMode = mode;
            document.getElementById('btnModeCont').classList.toggle('active', mode === 'continuous');
            document.getElementById('btnModeStep').classList.toggle('active', mode === 'stepped');
            document.getElementById('rampBar').classList.toggle('stepped', mode === 'stepped');
            renderMaps(false);
        }

        function toggleLayerVisibility() {
            state.showHeat = document.getElementById('chkHeat').checked;
            state.showPoints = document.getElementById('chkPoints').checked;
            state.showPath = document.getElementById('chkPath').checked;
            renderMaps(false);
        }

        function setBasemapStyle(styleKey) {
            state.basemap = styleKey;
            document.getElementById('btnTileOsm').classList.toggle('active', styleKey === 'osm');
            document.getElementById('btnTileSat').classList.toggle('active', styleKey === 'sat');

            const spec = TILE_URLS[styleKey] || TILE_URLS.osm;
            tileLayerInterp.setUrl(spec.url);
            tileLayerNodes.setUrl(spec.url);
            document.getElementById('interpBasemapAttr').textContent = spec.attr;
        }

        function parseCsvLine(line) {
            const out = [];
            let cur = '';
            let inQuotes = false;
            for (let i = 0; i < line.length; i++) {
                const ch = line[i];
                if (ch === '"') {
                    inQuotes = !inQuotes;
                } else if (ch === ',' && !inQuotes) {
                    out.push(cur.trim());
                    cur = '';
                } else {
                    cur += ch;
                }
            }
            out.push(cur.trim());
            return out;
        }

        async function fetchFullGoogleSheet(manualClick = false) {
            const statusEl = document.getElementById('feedStatusText');
            statusEl.textContent = 'Syncing Google Sheet...';
            try {
                const resp = await fetch(INITIAL.sheetCsvUrl);
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const csvText = await resp.text();
                const lines = csvText.split(/\r?\n/).filter(l => l.trim().length > 0);

                let zeroDropped = 0;
                const valid = [];
                for (let i = 1; i < lines.length; i++) {
                    const cols = parseCsvLine(lines[i]);
                    if (cols.length < 3) continue;
                    const cps = parseFloat(cols[0]);
                    const cpm = parseFloat(cols[1]);
                    const dose = parseFloat(cols[2]);
                    const timestamp = cols[4] || '';

                    if (Number.isNaN(dose) || Number.isNaN(cpm) || dose <= 0 || cpm <= 0) {
                        zeroDropped++;
                        continue;
                    }
                    valid.push({
                        id: valid.length + 1,
                        dose,
                        cpm,
                        cps: Number.isNaN(cps) ? 0 : cps,
                        timestamp,
                    });
                }

                if (valid.length > 0) {
                    state.liveReadings = valid;
                    state.rawCount = lines.length - 1;
                    state.zeroFiltered = zeroDropped;
                    if (manualClick) state.dataSource = 'live';
                    recomputeAndRender(false);
                    statusEl.textContent =
                        `${valid.length.toLocaleString()} valid pts (${zeroDropped} zeros filtered)`;
                }
            } catch (err) {
                statusEl.textContent =
                    `${state.liveReadings.length.toLocaleString()} valid pts (${state.zeroFiltered} zeros filtered)`;
            }
        }

        window.addEventListener('storage', (ev) => {
            if (ev.key === STORAGE_KEY && ev.newValue) {
                try {
                    const parsed = JSON.parse(ev.newValue);
                    if (parsed && parsed.bounds) {
                        state.bounds = parsed.bounds;
                        state.dataSource = parsed.dataSource || state.dataSource;
                        document.getElementById('inLat1').value = Number(state.bounds.lat1).toFixed(6);
                        document.getElementById('inLon1').value = Number(state.bounds.lon1).toFixed(6);
                        document.getElementById('inLat2').value = Number(state.bounds.lat2).toFixed(6);
                        document.getElementById('inLon2').value = Number(state.bounds.lon2).toFixed(6);
                        recomputeAndRender(true);
                    }
                } catch (e) {}
            }
        });

        window.addEventListener('DOMContentLoaded', () => {
            initStaticUI();
            initMaps();
            recomputeAndRender(true);
            fetchFullGoogleSheet(false);
            setTimeout(notifyParentHeight, 150);
            setTimeout(notifyParentHeight, 500);
        });
    </script>
</body>
</html>
"""


# ============================================================================
# 7. CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate ISOTOPER B15 Radiation Dose Maps from Full Google Sheet"
    )
    parser.add_argument("--lat1", type=float, default=DEFAULT_SURVEY_BOUNDS["lat1"], help="Start Latitude")
    parser.add_argument("--lon1", type=float, default=DEFAULT_SURVEY_BOUNDS["lon1"], help="Start Longitude")
    parser.add_argument("--lat2", type=float, default=DEFAULT_SURVEY_BOUNDS["lat2"], help="End Latitude")
    parser.add_argument("--lon2", type=float, default=DEFAULT_SURVEY_BOUNDS["lon2"], help="End Longitude")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt interactively for Start/End Latitude & Longitude in terminal",
    )
    args = parser.parse_args()

    lat1, lon1, lat2, lon2 = args.lat1, args.lat1, args.lat2, args.lon2
    lat1, lon1, lat2, lon2 = args.lat1, args.lon1, args.lat2, args.lon2

    if args.interactive:
        print("=== ISOTOPER B15 Survey Boundary Input ===")
        try:
            u_lat1 = input(f"Start Latitude  [{lat1}]: ").strip()
            u_lon1 = input(f"Start Longitude [{lon1}]: ").strip()
            u_lat2 = input(f"End Latitude    [{lat2}]: ").strip()
            u_lon2 = input(f"End Longitude   [{lon2}]: ").strip()
            if u_lat1:
                lat1 = float(u_lat1)
            if u_lon1:
                lon1 = float(u_lon1)
            if u_lat2:
                lat2 = float(u_lat2)
            if u_lon2:
                lon2 = float(u_lon2)
        except ValueError:
            print("[WARN] Invalid number entered; using default MIST coordinates.")

    generator = RadiationMapGenerator(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)
    out_dir = Path(__file__).resolve().parent
    generator.save_map(str(out_dir / "radiation_map_light.html"))


if __name__ == "__main__":
    main()
