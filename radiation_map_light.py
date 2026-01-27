import folium
import pandas as pd
import numpy as np
from folium.plugins import HeatMap, MarkerCluster, MeasureControl, MiniMap
from folium import FeatureGroup, LayerControl
import branca.colormap as cm
import warnings
warnings.filterwarnings('ignore')


class RadiationData:
    """Handles radiation measurement data from GM tube detector"""

    def __init__(self):
        # Sample data - Replace with actual measurements
        # Format: [longitude, latitude, radiation_μSv_h]
        self.data_points = [
            [90.35718591116404, 23.837775061102857, 0.4385],
            [90.35789165161599, 23.83827828945288, 0.00],
            [90.35706162131505, 23.838201481146786, 0.00],
            [90.35770007023059, 23.837719225607806, 0.00]

        ]

    def get_dataframe(self):
        """Convert data to pandas DataFrame with statistics"""
        df = pd.DataFrame(self.data_points,
                         columns=['Longitude', 'Latitude', 'Radiation_μSv_h'])
        stats = {
            'mean': df['Radiation_μSv_h'].mean(),
            'std': df['Radiation_μSv_h'].std(),
            'max': df['Radiation_μSv_h'].max(),
            'min': df['Radiation_μSv_h'].min(),
            'median': df['Radiation_μSv_h'].median()
        }
        return df, stats

    def get_heatmap_data(self):
        """Format data for heatmap visualization"""
        df, _ = self.get_dataframe()
        return [[row['Latitude'], row['Longitude'], row['Radiation_μSv_h']]
                for _, row in df.iterrows()]


def assess_risk_corrected(radiation_level):
    """
    Risk assessment based on IAEA GSG-8 and ICRP 103 standards
    Natural background: 0.1-0.15 μSv/h
    """
    if radiation_level < 0.15:
        return {
            "level": "SAFE",
            "color": "green",
            "action": "No restrictions needed",
            "description": "Within natural background range"
        }
    elif radiation_level < 0.3:
        return {
            "level": "LOW",
            "color": "yellow",
            "action": "Monitor regularly",
            "description": "Slightly above natural background"
        }
    elif radiation_level < 0.5:
        return {
            "level": "MODERATE",
            "color": "orange",
            "action": "Investigate possible sources",
            "description": "Above normal levels"
        }
    elif radiation_level < 1.0:
        return {
            "level": "HIGH",
            "color": "red",
            "action": "Restrict prolonged access",
            "description": "Requires attention"
        }
    else:
        return {
            "level": "SEVERE",
            "color": "purple",
            "action": "Immediate action required",
            "description": "Emergency situation"
        }

def calculate_exposure_safety(radiation_level):
    """
    Calculate safe exposure times based on international limits:
    Public: 1000 μSv/year (1 mSv/year)
    Workers: 20000 μSv/year (20 mSv/year)
    """
    if radiation_level <= 0:
        return {"status": "No radiation detected"}

    # Annual limits in microsieverts
    PUBLIC_LIMIT = 1000    # 1 mSv
    WORKER_LIMIT = 20000   # 20 mSv

    # Hours to reach annual limit at this radiation level
    hours_to_public_limit = PUBLIC_LIMIT / radiation_level
    hours_to_worker_limit = WORKER_LIMIT / radiation_level

    # Convert to days (8 hours/day exposure) # Assuming 8 hours/day exposure for calculation purposes
    days_to_public_limit = hours_to_public_limit / 8
    days_to_worker_limit = hours_to_worker_limit / 8

    # Calculate annual dose for typical exposure (8hr/day, 250 days)
    annual_dose_8hr = radiation_level * 8 * 250

    return {
        "public_safety": f"Safe for {days_to_public_limit:.0f} days (8hr/day)",
        "worker_safety": f"Safe for {days_to_worker_limit:.0f} days (8hr/day)",
        "annual_dose_estimate": f"{annual_dose_8hr:.0f} μSv/year",
        "percent_of_public_limit": f"{(annual_dose_8hr/PUBLIC_LIMIT)*100:.1f}%",
        "percent_of_worker_limit": f"{(annual_dose_8hr/WORKER_LIMIT)*100:.1f}%"
    }



class RadiationMapGenerator:
    """Creates interactive radiation maps with risk assessment"""

    def __init__(self, center_lat=23.8103, center_lon=90.4125):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radiation_data = RadiationData()

    def create_base_map(self):
        """Initialize the base map with Google Maps light theme"""
        # Create base map with OpenStreetMap as default
        radiation_map = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=17,
            tiles='OpenStreetMap',
            control_scale=True,
            width='100%',
            height='90%'
        )

        # Add Google Maps layers (light theme)
        # Note: For Google Maps tiles, you need to sign up for Google Maps API
        # Here's an alternative using OpenStreetMap light themes

        # Light theme options
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            attr='© OpenStreetMap contributors, © CARTO',
            name='Light Theme',
            control=True
        ).add_to(radiation_map)

        # CartoDB Positron (very clean light theme)
        folium.TileLayer('CartoDB positron', name='Positron Light').add_to(radiation_map)

        # Stamen Toner Lite (another light theme)
        folium.TileLayer(
            tiles='Stamen Toner',
            name='Toner Light',
            attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
            control=True
        ).add_to(radiation_map)

        # Add regular OpenStreetMap
        folium.TileLayer('OpenStreetMap', name='Standard Map').add_to(radiation_map)

        # Add satellite view
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            control=True
        ).add_to(radiation_map)

        # Set Light Theme as default
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            attr='© OpenStreetMap contributors, © CARTO',
            name='Light Theme',
            control=False
        ).add_to(radiation_map)

        # Add tools
        MiniMap(toggle_display=True, tile_layer='CartoDB positron').add_to(radiation_map)
        MeasureControl(position='topleft').add_to(radiation_map)

        return radiation_map

    def create_google_style_map(self):
        """Alternative method with Google Maps-like appearance"""
        # Create map with CartoDB Positron (closest to Google Maps light theme)
        google_style_map = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=17,
            tiles='CartoDB positron',  # This looks very similar to Google Maps
            attr='© OpenStreetMap contributors, © CARTO',
            control_scale=True,
            width='100%',
            height='90%'
        )

        # Add other layers as options
        folium.TileLayer('OpenStreetMap', name='Street View').add_to(google_style_map)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite View',
            control=True
        ).add_to(google_style_map)

        # Add terrain layer for variety
        folium.TileLayer(
            tiles='Stamen Terrain',
            name='Terrain',
            attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
            control=True
        ).add_to(google_style_map)

        # Add tools
        MiniMap(toggle_display=True, tile_layer='CartoDB positron').add_to(google_style_map)
        MeasureControl(position='topleft').add_to(google_style_map)

        return google_style_map

    def add_heatmap_layer(self, map_obj):
        """Add radiation heatmap"""
        heatmap_data = self.radiation_data.get_heatmap_data()

        heatmap = HeatMap(
            heatmap_data,
            name='Radiation Heatmap',
            min_opacity=0.3,
            max_opacity=0.8,
            radius=25,
            blur=15,
            gradient={
                0.1: 'green',    # Background
                0.2: 'yellow',   # Low
                0.3: 'orange',   # Moderate
                0.4: 'red',      # High
                0.5: 'purple'    # Severe
            }
        )

        heatmap_group = FeatureGroup(name='Radiation Heatmap', show=True)
        heatmap_group.add_child(heatmap)
        map_obj.add_child(heatmap_group)

        return map_obj

    def add_measurement_points(self, map_obj):
        """Add interactive measurement markers"""
        df, _ = self.radiation_data.get_dataframe()

        marker_cluster = MarkerCluster(
            name='Measurement Points',
            options={'showCoverageOnHover': True}
        ).add_to(map_obj)

        for idx, row in df.iterrows():
            risk = assess_risk_corrected(row['Radiation_μSv_h'])
            safety = calculate_exposure_safety(row['Radiation_μSv_h'])

            # Determine safety message based on radiation level
            safety_info_html = ""
            if 'status' in safety:
                safety_info_html = f"<p><b>Safety Info:</b> {safety['status']}</p>"
            else:
                safety_info_html = f"""
                <p><b>Public Safety:</b> {safety['public_safety']}</p>
                <p><b>Worker Safety:</b> {safety['worker_safety']}</p>
                """

            # Create detailed popup
            popup_html = f"""
            <div style="font-family: Arial; width: 280px;">
                <h4 style="color: {risk['color']}; margin: 5px 0;">
                     Point {idx+1} - {risk['level']}
                </h4>
                <hr style="margin: 5px 0;">
                <p><b>Radiation Level:</b> {row['Radiation_μSv_h']:.3f} μSv/h</p>
                <p><b>Risk Assessment:</b> {risk['description']}</p>
                <p><b>Recommended Action:</b> {risk['action']}</p>
                {safety_info_html}
                <p><b>Coordinates:</b><br>
                {row['Latitude']:.6f}°N, {row['Longitude']:.6f}°E</p>
                <hr style="margin: 5px 0;">
                <small>GM Tube Detector • University Research Project</small>
            </div>
            """

            # Add marker with popup
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color=risk['color'],
                fill=True,
                fill_color=risk['color'],
                fill_opacity=0.7,
                weight=2
            ).add_to(marker_cluster)

        return map_obj

    def add_risk_zones(self, map_obj):
        """Add visual risk zone overlays"""
        df, _ = self.radiation_data.get_dataframe()
        risk_group = FeatureGroup(name='Risk Zones', show=False)

        for _, row in df.iterrows():
            risk = assess_risk_corrected(row['Radiation_μSv_h'])

            # Determine zone radius based on risk level
            zone_sizes = {
                'SAFE': 10, 'LOW': 20,
                'MODERATE': 30, 'HIGH': 40, 'SEVERE': 50
            }
            radius = zone_sizes[risk['level']]

            # Create risk zone circle
            folium.Circle(
                location=[row['Latitude'], row['Longitude']],
                radius=radius,
                color=risk['color'],
                fill=True,
                fill_opacity=0.15,
                weight=2,
                popup=f"<b>{risk['level']} Risk Zone</b><br>{risk['description']}"
            ).add_to(risk_group)

        map_obj.add_child(risk_group)
        return map_obj

    def add_statistics_panel(self, map_obj):
        """Add statistics panel to map"""
        df, stats = self.radiation_data.get_dataframe()

        # Count risk levels
        risk_counts = {'SAFE': 0, 'LOW': 0, 'MODERATE': 0, 'HIGH': 0, 'SEVERE': 0}
        for _, row in df.iterrows():
            risk = assess_risk_corrected(row['Radiation_μSv_h'])
            risk_counts[risk['level']] += 1

        # Create statistics HTML
        stats_html = f"""
        <div style="
            position: fixed;
            top: 10px;
            right: 10px;
            width: 320px;
            background: white;
            border: 2px solid #2c3e50;
            border-radius: 5px;
            padding: 10px;
            font-family: Arial;
            font-size: 12px;
            z-index: 9999;
            opacity: 0.95;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
        ">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50;">
                 Radiation Statistics
            </h4>
            <hr style="margin: 5px 0;">
            <p><b>Average:</b> {stats['mean']:.3f} μSv/h</p>
            <p><b>Maximum:</b> {stats['max']:.3f} μSv/h</p>
            <p><b>Minimum:</b> {stats['min']:.3f} μSv/h</p>
            <p><b>Std Deviation:</b> {stats['std']:.3f} μSv/h</p>
            <hr style="margin: 5px 0;">
            <p><b>Risk Distribution:</b></p>
            <div style="margin-left: 10px;">
                <p>• Safe: {risk_counts['SAFE']} points</p>
                <p>• Low: {risk_counts['LOW']} points</p>
                <p>• Moderate: {risk_counts['MODERATE']} points</p>
                <p>• High: {risk_counts['HIGH']} points</p>
                <p>• Severe: {risk_counts['SEVERE']} points</p>
            </div>
            <hr style="margin: 5px 0;">
            <small>Natural background: 0.10-0.15 μSv/h</small>
        </div>
        """

        map_obj.get_root().html.add_child(folium.Element(stats_html))

        # Add color scale legend
        colormap = cm.LinearColormap(
            colors=['green', 'yellow', 'orange', 'red', 'purple'],
            vmin=df['Radiation_μSv_h'].min(),
            vmax=df['Radiation_μSv_h'].max(),
            caption='Radiation Level (μSv/h)'
        )
        colormap.add_to(map_obj)

        return map_obj

    def generate_light_theme_map(self):
        """Generate complete map with Google Maps-like light theme"""
        print("  Creating interactive radiation map with light theme...")

        # Create base map with light theme
        radiation_map = self.create_google_style_map()

        # Add all layers
        radiation_map = self.add_heatmap_layer(radiation_map)
        radiation_map = self.add_measurement_points(radiation_map)
        radiation_map = self.add_risk_zones(radiation_map)
        radiation_map = self.add_statistics_panel(radiation_map)

        # Add layer control
        LayerControl(collapsed=False, position='topright').add_to(radiation_map)

        # Add title banner
        title_html = '''
        <div style="
            position: fixed;
            top: 10px;
            left: 50px;
            z-index: 9999;
            background: rgba(255,255,255,0.95);
            padding: 10px 15px;
            border-radius: 5px;
            border: 2px solid #2c3e50;
            font-family: Arial;
            max-width: 500px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
        ">
            <h3 style="margin: 0; color: #2c3e50;"> Radiation Mapping System</h3>
            <p style="margin: 5px 0 0 0; font-size: 12px; color: #34495e;">
                GM Tube Detection • Drone Survey • University Research Project<br>
                <span style="color: #e74c3c;"> Safety Limits: Public 1000 μSv/year • Workers 20000 μSv/year</span>
            </p>
        </div>
        '''

        radiation_map.get_root().html.add_child(folium.Element(title_html))

        return radiation_map

    def generate_complete_map(self):
        """Legacy method - use generate_light_theme_map instead"""
        return self.generate_light_theme_map()

    def save_map(self, filename="radiation_map_light.html"):
        """Save map to HTML file"""
        radiation_map = self.generate_light_theme_map()
        radiation_map.save(filename)
        return filename

# ============================================================================
# 4. RISK REPORT GENERATOR
# ============================================================================

def generate_risk_report(radiation_data):
    """Generate comprehensive risk assessment report"""
    df, stats = radiation_data.get_dataframe()

    # Analyze risk distribution
    risk_counts = {'SAFE': 0, 'LOW': 0, 'MODERATE': 0, 'HIGH': 0, 'SEVERE': 0}
    hotspots = []

    for idx, row in df.iterrows():
        risk = assess_risk_corrected(row['Radiation_μSv_h'])
        risk_counts[risk['level']] += 1

        if risk['level'] in ['MODERATE', 'HIGH', 'SEVERE']:
            hotspots.append({
                'point': idx + 1,
                'radiation': row['Radiation_μSv_h'],
                'risk': risk['level']
            })

    # Calculate overall risk index (0-100)
    max_reading = stats['max']
    mean_reading = stats['mean']
    risk_index = min(100, (max_reading * 70) + (mean_reading * 30))

    # Determine overall risk level
    if risk_index < 30:
        overall_risk = "SAFE"
        recommendations = [
            "Continue routine monitoring",
            "Maintain baseline records",
            "No restrictions needed"
        ]
    elif risk_index < 50:
        overall_risk = "LOW"
        recommendations = [
            "Monitor monthly",
            "Investigate minor elevations",
            "Inform local authorities"
        ]
    elif risk_index < 70:
        overall_risk = "MODERATE"
        recommendations = [
            "Increase monitoring to weekly",
            "Restrict prolonged access to elevated areas",
            "Investigate potential sources"
        ]
    elif risk_index < 90:
        overall_risk = "HIGH"
        recommendations = [
            "Restrict area access",
            "Notify radiation safety officer",
            "Begin source investigation",
            "Implement protective measures"
        ]
    else:
        overall_risk = "SEVERE"
        recommendations = [
            "IMMEDIATE AREA RESTRICTION",
            "Contact nuclear regulatory body",
            "Deploy emergency response",
            "Evacuate non-essential personnel"
        ]

    # Generate report
    report = f"""
{'='*70}
RADIATION RISK ASSESSMENT REPORT
Environmental Monitoring System - Isotopper B15 Project
{'='*70}

SURVEY SUMMARY:
---------------
• Data Points Collected: {len(df)}
• Maximum Radiation: {stats['max']:.3f} μSv/h
• Average Radiation: {stats['mean']:.3f} μSv/h
• Natural Background Range: 0.10 - 0.15 μSv/h

RISK ASSESSMENT:
----------------
• Overall Risk Level: {overall_risk} (Index: {risk_index:.1f}/100)
• Risk Distribution:
   - Safe: {risk_counts['SAFE']} points
   - Low: {risk_counts['LOW']} points
   - Moderate: {risk_counts['MODERATE']} points
   - High: {risk_counts['HIGH']} points
   - Severe: {risk_counts['SEVERE']} points

SAFETY ANALYSIS:
----------------
• Public Annual Limit: 1000 μSv (1 mSv)
• Worker Annual Limit: 20000 μSv (20 mSv)
• Highest Reading Annual Dose (8hr/day, 250 days): {max_reading * 8 * 250:.0f} μSv
• % of Public Limit: {(max_reading * 8 * 250 / 1000) * 100:.1f}%
• % of Worker Limit: {(max_reading * 8 * 250 / 20000) * 100:.1f}%

HOTSPOTS IDENTIFIED:
--------------------
"""

    if hotspots:
        for hotspot in hotspots:
            exposure = calculate_exposure_safety(hotspot['radiation'])
            report += f"""• Point {hotspot['point']}: {hotspot['radiation']:.3f} μSv/h ({hotspot['risk']} Risk)
   - Public Safety: {exposure['public_safety']}
   - Worker Safety: {exposure['worker_safety']}
"""
    else:
        report += "• No significant hotspots detected (all readings within safe limits)\n"

    report += f"""
RECOMMENDED ACTIONS:
-------------------
"""
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"

    report += f"""
CONCLUSION:
-----------
Based on IAEA/ICRP international standards and {len(df)} measurement points,
the surveyed area presents {overall_risk.lower()} risk level.
{("All readings are within acceptable limits for both public and occupational exposure."
   if overall_risk in ['SAFE', 'LOW'] else
   "Some areas require attention but do not pose immediate danger with proper controls.")}

{'='*70}
Report Generated:  Radiation Monitoring System
Reference: IAEA GSG-8, ICRP Publication 103
{'='*70}
"""

    return report

# Create and display the light theme map
map_generator = RadiationMapGenerator()
light_theme_map = map_generator.generate_light_theme_map()
light_theme_map
# Generate and save the light theme map
print("Generating measurement points map...")
map_generator = RadiationMapGenerator()

# Save the map
map_filename = "radiation_map_light.html"
map_generator.save_map(map_filename)

print(f"✅ Map saved as: {map_filename}")

# Download to your computer
from google.colab import files
files.download(map_filename)

print("📥 Check your Downloads folder!")