import pandas as pd
import folium
import json
from folium.plugins import BeautifyIcon  # Import BeautifyIcon for custom markers

# Initialize a map centered at Ho Chi Minh City
m = folium.Map(location=[10.8231, 106.6297], zoom_start=13)

# Load GeoJSON file containing Ho Chi Minh City boundary and districts
boundary_file = 'hcm.geojson'  # Replace with the path to the Ho Chi Minh.geojson file

# Load the GeoJSON data
with open(boundary_file, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

# Define a style function to add color and borders for district boundaries
def style_function(feature):
    return {
        'fillColor': '#ffcccb',
        'color': 'black',
        'weight': 2,
        'fillOpacity': 0.3
    }

# Create a dictionary to store the feature groups (one per district)
district_layers = {}

# Filter and create layers for each district
for feature in geojson_data['features']:
    name_vi = feature['properties'].get('name', 'Unnamed region')
    
    if name_vi == 'Unnamed region':
        continue

    if name_vi not in district_layers:
        district_layers[name_vi] = folium.FeatureGroup(name=f"Region Name: {name_vi}")

    folium.GeoJson(
        feature,
        style_function=style_function,
        popup=folium.Popup(f"Region Name: {name_vi}", max_width=300)
    ).add_to(district_layers[name_vi])

for district_name, district_layer in district_layers.items():
    district_layer.add_to(m)

# Create layer groups for new and existing charging stations (to be toggled)
new_stations_layer = folium.FeatureGroup(name='New Charging Stations', show=True)
existing_stations_layer = folium.FeatureGroup(name='Existing Charging Stations', show=True)

# Create layer groups for line connections (filterable)
new_station_lines_layer = folium.FeatureGroup(name='Lines to New Charging Stations', show=True)
existing_station_lines_layer = folium.FeatureGroup(name='Lines to Existing Charging Stations', show=True)

# Load the charging stations data
file_path_1 = 'output_visualize/Combined_Output.csv'
data_1 = pd.read_csv(file_path_1)
file_path_2 = 'EXISTING.csv'
data_2 = pd.read_csv(file_path_2)

# Group new charging stations by their x and y coordinates
new_station_groups = data_1.groupby(['x coordinate', 'y coordinate'])

# Add new charging stations with aggregated Type and Value information for duplicate points
for (lat, lon), group in new_station_groups:
    unit = group['Unit'].iloc[0]  # Assume all stations in the same location belong to the same unit
    
    # Collect all Type and Value information for stations at this location
    station_types = group['Type'].tolist()
    values = group['Value'].tolist()
    
    # Create the popup content with Unit, and list all Type and Value pairs for the same location
    popup_content = f"Unit: {unit}<br>X Coordinate: {lat}<br>Y Coordinate: {lon}<br><br>"
    
    # Append each Type and Value entry
    for i in range(len(station_types)):
        popup_content += f"Type: {station_types[i]}<br>Value: {values[i]}<br><br>"
    
    # Add the CircleMarker to the new_stations_layer
    folium.CircleMarker(
        location=[lat, lon],
        radius=1,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=1,
        popup=popup_content
    ).add_to(new_stations_layer)


# Add existing charging stations with additional information (Unit, Name, Type, Value), handling duplicates
station_groups = data_2.groupby(['x coordinate', 'y coordinate'])

for (lat, lon), group in station_groups:
    unit = group['Unit'].iloc[0]  # Assume all stations in the same location belong to the same unit
    name = group['Name'].iloc[0]  # Assume Name is the same for stations at the same coordinates
    
    station_types = group['Type'].tolist()  # Get all types for this coordinate
    values = group['Value'].tolist()  # Get all values for this coordinate
    
    # Create the popup content with Unit and Name once, and list all Type and Value pairs below
    popup_content = f"Unit: {unit}<br>Name: {name}<br>X Coordinate: {lat}<br>Y Coordinate: {lon}<br><br>"
    
    # Append Type and Value information
    for i in range(len(station_types)):
        popup_content += f"Type: {station_types[i]}<br>Value: {values[i]}<br><br>"
    
    # Add the CircleMarker to the existing_stations_layer
    folium.CircleMarker(
        location=[lat, lon],
        radius=1,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=1,
        popup=popup_content
    ).add_to(existing_stations_layer)



new_stations_layer.add_to(m)
existing_stations_layer.add_to(m)

# Load PEOPLE.csv data for house markers
people_file_path = 'PEOPLE_OLD.csv'
people_data = pd.read_csv(people_file_path)

# Create a feature group for house markers
houses_layer = folium.FeatureGroup(name='Houses', show=True)

# Track houses without connections
unconnected_houses = []

# Add house markers and draw lines from houses to charging stations in the same region
for idx, house_row in people_data.iterrows():
    house_lat = house_row['Lat2']
    house_lon = house_row['Lon2']
    house_unit = house_row['Unit']
    
    # Skip houses with missing coordinates or unit
    if pd.isna(house_lat) or pd.isna(house_lon) or pd.isna(house_unit):
        continue
    
    # Filter new charging stations by the same region (unit)
    new_stations_in_region = data_1[data_1['Unit'] == house_unit]
    existing_stations_in_region = data_2[data_2['Unit'] == house_unit]
    
    # Skip houses with no charging stations in their region
    if new_stations_in_region.empty and existing_stations_in_region.empty:
        print(f"House at {house_lat}, {house_lon} in {house_unit} has no charging stations in its region.")
        unconnected_houses.append((house_lat, house_lon, house_unit))
        continue
    
    # Add a star icon marker for the house using BeautifyIcon
    icon_star = BeautifyIcon(
        icon='star',
        inner_icon_style='color:green;font-size:10px;',  # Adjust the color and size of the star
        background_color='transparent',
        border_color='transparent'
    )
    
    popup_content = f"House in {house_unit}<br>Latitude: {house_lat}<br>Longitude: {house_lon}"
    folium.Marker(
        location=[house_lat, house_lon],
        icon=icon_star,  # Use the star icon
        popup=popup_content
    ).add_to(houses_layer)
    
    # Draw gray and thin lines from the house to each new charging station (add to new_station_lines_layer)
    for _, station_row in new_stations_in_region.iterrows():
        station_lat = station_row['x coordinate']
        station_lon = station_row['y coordinate']
        folium.PolyLine(
            locations=[[house_lat, house_lon], [station_lat, station_lon]],
            color='#000000',  # Gray color
            weight=0.1  # Thin line
        ).add_to(new_station_lines_layer)
    
    # Draw gray and thin lines from the house to each existing charging station (add to existing_station_lines_layer)
    for _, station_row in existing_stations_in_region.iterrows():
        station_lat = station_row['x coordinate']
        station_lon = station_row['y coordinate']
        folium.PolyLine(
            locations=[[house_lat, house_lon], [station_lat, station_lon]],
            color='#000000',  # Gray color
            weight=0.1  # Thin line
        ).add_to(existing_station_lines_layer)

# Add the house layer to the map
houses_layer.add_to(m)

# Add line layers to the map (filterable)
new_station_lines_layer.add_to(m)
existing_station_lines_layer.add_to(m)

# Optional: Print a summary of houses that weren't connected
if unconnected_houses:
    print(f"Unconnected Houses: {unconnected_houses}")
else:
    print("All houses are connected to at least one charging station.")

# Add LayerControl to switch between regions, charging stations, lines, and house locations
folium.LayerControl(collapsed=False).add_to(m)

# Add a custom legend to the map
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; 
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; padding: 10px; text-align: left; display: flex; flex-direction: column; justify-content: center; width: auto; height: auto;">
    <b style="text-align: center; margin-bottom: 5px;">Note</b>
    <div style="text-align: center;">
        <i class="fa fa-bolt" style="color:red"></i> New charging station <br>
        <i class="fa fa-bolt" style="color:blue"></i> Existing charging station <br>
        <i class="fa fa-star" style="color:green"></i> House (Star Icon) <br>
        <i style="color:#808080;">--- </i> Line to Charging Station (Gray) <br>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
output_map = 'output_map.html'
m.save(output_map)

print(f"Map has been saved to {output_map}")
