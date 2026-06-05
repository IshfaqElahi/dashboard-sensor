# ISOTOPER B15: Core Telemetry & Spatial Analysis Framework

## 📝 Overview
**ISOTOPER B15** is an advanced, real-time web dashboard designed to visualize critical telemetry and environmental data intercepted by specialized UAV hardware. This framework processes and maps live atmospheric conditions, hazardous gas concentrations, and radiation dosimetry into a highly responsive, tactical interface.

## ⚡ Key Features
* **Real-Time Sensor Telemetry:** Live synchronization with hardware arrays processing Ambient Temperature, Relative Humidity, Hydrogen Gas [MQ-8] concentration, and Radiation Dosimetry.
* **Gaussian Dispersion Engine:** Integrated mathematical atmospheric modeling cross-linked with live meteorological APIs to generate 3D pollution and radiation plume trajectories.
* **Dual-Matrix Spatial Mapping:** Interactive cartography featuring both discrete UAV measurement nodes and continuous, algorithmically interpolated radiation distribution fields.
* **Critical Alert Protocol:** Built-in threshold logic that automatically triggers pulsating visual warnings and UI shifts when hazardous parameters are intercepted.
* **Fluid Responsive Interface:** Custom-engineered layout that morphs seamlessly across mobile and desktop displays without sacrificing data density.

## 🛠️ Architecture & Tech Stack
The dashboard is built on a lightweight, highly optimized client-side architecture designed for rapid deployment and zero-latency UI updates in the field.
* **Core:** HTML5, CSS3, Vanilla JavaScript (ES6+).
* **Styling Engine:** Custom CSS Variables, Flexbox/Grid architecture, smooth `cubic-bezier` hardware-accelerated animations, and dynamic theme toggling.
* **CI/CD:** Automated static deployment pipeline via GitHub Actions.

### Data & Analytics
Data streams and historical matrices are handled via a robust integration of specialized visualization libraries and backend bridging scripts:
* **Geospatial Rendering:** `Leaflet.js` integrated with custom awesome-markers and heatmap overlays.
* **Time-Series Analytics:** `Chart.js` for rendering live historical data vectors tracking atmospheric drift and dose rates.
* **Bridging & Aggregation:** Google Apps Script utilized as the lightweight API bridge between the physical ESP32/Hardware nodes and the frontend dashboard.
* **Interpolation Processing:** Python environment (`pandas`, `folium`, `numpy`) utilized for generating the predictive spatial data models.

## 📁 Repository Structure

```text
dashboard-sensor-main/
│
├── index.html                       # Main Telemetry Dashboard & UI Hub
├── gpm.html                         # Gaussian Dispersion Protocol Engine
├── radiation_map_light.html         # Absolute Node Cartography Map
├── radiation_dispersion_map.html    # Interpolated Distribution Matrix Map
├── radiation_map_light.py           # Python script for geospatial data processing
│
└── .github/workflows/
    └── static.yml                   # GitHub Actions automated deployment configuration
