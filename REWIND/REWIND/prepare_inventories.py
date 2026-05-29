print(">>> USING THIS FILE <<<")
import time
from pathlib import Path
from logging import getLogger

import pandas as pd
import bw2data as bd
import bw2io as bi
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point
import geopandas as gpd
from geopy.distance import geodesic
import netCDF4 as nc
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import time
from bw2data.backends.schema import ActivityDataset

logger = getLogger(__name__)
start_time = time.time()

# Constants
_TURBINE_RATED_POWERS = [30, 150, 600, 800, 2000]
_DATA_DIR = Path(__file__).resolve().parent / "data"

def ecoinvent_setup(ei_path):
    # Check if the project exists; create it if not
    if 'wimby' not in bd.projects:
        bd.projects.create_project('wimby')
    bd.projects.set_current('wimby')

    # Setup biosphere database
    mybio = "biosphere3"
    if mybio not in bd.databases:
        bi.bw2setup()
        print("Biosphere database set up successfully.")
        mybio = bd.Database('biosphere3')
    
    # Setup ecoinvent database
    eidb = "ecoinvent-391-cutoff"
    if eidb not in bd.databases:
        ei_importer = bi.SingleOutputEcospold2Importer(ei_path, eidb, use_mp=False)
        ei_importer.apply_strategies()
        ei_importer.statistics()
        ei_importer.write_database()
        print("Ecoinvent database imported successfully.")
        eidb = bd.Database('ecoinvent-391-cutoff')

    else:
        eidb = bd.Database('ecoinvent-391-cutoff')
        mybio = bd.Database('biosphere3')
    return mybio, eidb

def _set_power_as_int(df):
    # Convert power to integer
    for power in _TURBINE_RATED_POWERS:
        df.replace(f"{power}kW", str(power), inplace=True)
    df["Power"] = df["Power"].astype(int)
    return df

def get_iso_country_code(lat, lon):
    world = gpd.read_file(_DATA_DIR / "ne_10m_admin_0_countries.shp")
    point = Point(lon, lat)
    for _, country in world.iterrows():
        if country['geometry'].contains(point):
            print(country.index)
            return country['ISO_A2_EH']
    return None

def find_nearest_shore(lat, lon, step_size=0.1, max_distance=1000):
    for distance in np.arange(0.1, max_distance, step_size):
        for lat_offset in [-distance, 0, distance]:
            for lon_offset in [-distance, 0, distance]:
                if lat_offset == 0 and lon_offset == 0:
                    continue
                iso_code = get_iso_country_code(lat + lat_offset, lon + lon_offset)
                if iso_code:
                    return iso_code
    return None

def get_iso_or_nearest_shore(lat, lon):
    iso_code = get_iso_country_code(lat, lon)
    return iso_code if iso_code else find_nearest_shore(lat, lon)

def adjust_foundation():
    datasets = pd.read_excel(_DATA_DIR / 'Wind turbines inventories_03.xlsx', sheet_name="All", dtype=None, decimal=";", header=0)
    datasets = _set_power_as_int(datasets)
    # Modify dataset based on specified rules
    datasets.loc[(datasets['Component'] == 'Foundation') & (datasets['Market name'] == 'market for concrete, normal strength'), 'Market name'] = 'market for concrete, 30MPa'
    datasets.loc[(datasets['Component'] == 'Foundation') & (datasets['Market name'] == 'market for steel, low-alloyed'), 'Market name'] = 'market for reinforcing steel'
    datasets.loc[(datasets['Component'] == 'Foundation') & (datasets['Market name'] == 'market for reinforcing steel'), 'Dataset'] = 'Reinforcing steel'
    return datasets

def get_activities_from_names(database, names):
    query = (ActivityDataset.database == database.name) & (ActivityDataset.name << names) 
    query_result = ActivityDataset.select().where(query)

    activity_list = list(query_result)
    return activity_list

def get_activities_from_eidb(database, names):
    query = (ActivityDataset.database == database.name) & (ActivityDataset.name == names) 
    query_result = ActivityDataset.select().where(query)

    activity_list = list(query_result)
    return activity_list

def update_datasets_with_uuid_and_location(lon, lat):
    # Get ecoinvent database and updated datasets
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
    datasets = adjust_foundation()

    iso_code = get_iso_or_nearest_shore(lat=lat, lon=lon)
    # db_biosphere = bd.Database("biosphere3")
    all_market_names = list(datasets["Market name"])
    #print(all_market_names)
    #print(type(eidb))
    #print(type(mybio))  # Print the first few elements for inspection if it's a list

    # Get matching activities
    # eidb_act = [[act, act["name"]] for act in eidb if act["name"] in  all_market_names]
    
    eidb_query_res=get_activities_from_names(eidb, all_market_names)

    # bioddb_act = [[act, act["name"]] for act in db_biosphere if act["name"] in all_market_names]
    mybio_query_res = get_activities_from_names(mybio, all_market_names)
    #print(eidb_act)

    def determine_location(row):
        if row['Phase'] == 'Input':
            return ['GLO', 'RoW', 'RER', 'Europe without Switzerland', 'CH']
        elif row['Phase'] == 'Assembly':
            return [iso_code, 'RER', 'Europe without Switzerland', 'GLO', 'RoW']
        elif row['Phase'] == 'Maintenance':
            return [iso_code, 'RER', 'Europe without Switzerland']
        elif row['Phase'] == 'Disposal':
            return [iso_code, 'RER', 'Europe without Switzerland', 'RoW']
        else:
            return []

    datasets['Location'] = datasets.apply(determine_location, axis=1)
    datasets['UUID'] = None
    datasets['Exact Location'] = None

    for index, row in datasets.iterrows():
        market_name = row['Market name'].strip()
        locations = row['Location']

        if 'ecoinvent' in row['Database name']:
            matching_acts = [value for value in eidb_query_res if value.name == market_name]
        elif 'biosphere3' in row['Database name']:
            matching_acts = [value for value in mybio_query_res if value.name == market_name]

        if len(matching_acts) > 1 and locations:
            matching_acts = [act for act in matching_acts if act.location in locations]

        if matching_acts:
            datasets.at[index, 'UUID'] = (matching_acts[0].database,matching_acts[0].code)
            datasets.at[index, 'Exact Location'] = matching_acts[0].location

    datasets = datasets.drop(columns=['Location']).rename(columns={'Exact Location': 'Location'})
    return datasets

def activities_and_uuids(lon, lat):
    df_act = update_datasets_with_uuid_and_location(lon, lat)
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")

    #new_act = [act for act in eidb if 'market for waste' in act['name'] and 'aluminium' in act['name']][0]
    #print(new_act['name'])
    waste_aluminium_query = get_activities_from_names(
        eidb, 
        ['market for waste aluminium']  # Pass names as a list
    )
    new_act = waste_aluminium_query[0]
    #print(new_act)
    #new_act = get_activities_from_eidb(eidb, "market for waste aluminium")
    #print(new_act)
    df_act2 = pd.DataFrame([{
        'Power': None,
        'Phase': 'Disposal',
        'Component': None,
        'Sub-component': None,
        'Dataset': 'Aluminium waste',
        'Unit': 'kg',
        'Quantity': None,
        'Database name': eidb.name,
        'Market name': new_act.name,#['name'], #query.name
        'UUID': new_act.code,#['code'], #query.code
        'Location': new_act.location#['location']
    }])
    df_act = pd.concat([df_act, df_act2], ignore_index=True)

    new_row = df_act[df_act['Dataset'] == 'Steel, inert waste'].iloc[0].copy()
    new_row['Dataset'] = 'Chromium Steel waste'
    df_act = pd.concat([df_act, pd.DataFrame([new_row])], ignore_index=True)

    # Ensure all column names are strings
    df_act.columns = df_act.columns.astype(str)

    # Remove unintended numeric columns
    df_act = df_act.loc[:, ~df_act.columns.str.isnumeric()]

    # Remove columns with only NaN values
    df_act = df_act.dropna(axis=1, how='all')

    # Replace hardcoded database name with the current eidb.name
    df_act = df_act.replace('ecoinvent cutoff 391', eidb.name)

    # Save to file
    df_act.to_pickle('activities_and_uuids.pkl')
    return df_act


def prepare_inventory():
    df_inv = pd.read_excel(_DATA_DIR / 'Wind turbines inventories_03.xlsx')
    _set_power_as_int(df_inv)
    
    df_inv = df_inv.pivot_table(columns= 'Power', values=['Quantity','Unit'] , index=['Phase', 'Component','Sub-component', 'Dataset'], aggfunc='sum')
    df_inv = df_inv.T

    df_inv=df_inv.loc['Quantity']
    df_inv.loc[0]=0
    df_inv=df_inv.sort_index()
        #added: transfering all values to numeric dtype
    df_inv = df_inv.apply(pd.to_numeric, errors='coerce')
    df_inv.interpolate(method = 'index', limit_direction = 'both', inplace = True)
    df_inv = df_inv.drop(df_inv.index[0])
    df_inv.to_pickle('df_inv.pkl')

    return df_inv

def total_inventory():
    # Load or prepare inventory
    df_inv = pd.read_pickle('df_inv.pkl') if Path('df_inv.pkl').exists() else prepare_inventory()

    df_inv_tot = df_inv.T.groupby(level=['Phase','Component']).sum().T
    return df_inv_tot

def percentage_inventory():
    # Load or prepare inventory
    df_inv = pd.read_pickle('df_inv.pkl') if Path('df_inv.pkl').exists() else prepare_inventory()
    df_inv_tot = total_inventory()
    df_perc = df_inv.copy().astype(float)

    for phase in set(df_inv.columns.get_level_values(0)):
        for component in set(df_inv[phase].columns.get_level_values(0)):
            for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
                for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                    percentage = df_inv[phase][component][sub_comp][dataset] / df_inv_tot[phase][component]
                    df_perc.loc[:, (phase, component, sub_comp, dataset)] = percentage
    df_perc.to_pickle('df_perc.pkl')
    return df_perc

def inventory_not_kg():
    df_inv_not_kg = pd.read_excel(_DATA_DIR / 'Wind turbines inventories_03.xlsx')
    _set_power_as_int(df_inv_not_kg)

    df_inv_not_kg = df_inv_not_kg[ df_inv_not_kg.Unit != 'kg']

    df_inv_not_kg = df_inv_not_kg.pivot_table(columns= 'Power', values=['Quantity','Unit'] , index=['Phase', 'Component','Sub-component', 'Dataset'], aggfunc='sum')
    df_inv_not_kg = df_inv_not_kg.T

    df_inv_not_kg.loc[:, ('Assembly', 'Tower', 'Assembly', 'Galvanizing [m]')] = df_inv_not_kg.loc[:, ('Assembly', 'Tower', 'Assembly', 'Steel arc welding [m]')]
    df_inv_not_kg = df_inv_not_kg.loc['Quantity']
    df_inv_not_kg = df_inv_not_kg.drop('Electricity [kWh]', axis = 1, level = 3)
    # Fill NaN values with 0
    #df_inv_not_kg = df_inv_not_kg.fillna(0)
    df_inv_not_kg.to_pickle('df_inv_not_kg.pkl')

    return df_inv_not_kg

def assembly_activity():
    df_inv = pd.read_pickle('df_inv.pkl') if Path('df_inv.pkl').exists() else prepare_inventory()
    assembly_activities = list(set(df_inv['Assembly'].columns.get_level_values(2)))

    return assembly_activities

def disposal_activity():
    df_inv = pd.read_pickle('df_inv.pkl') if Path('df_inv.pkl').exists() else prepare_inventory()
    disposal_activities = list(set(df_inv['Disposal'].columns.get_level_values(2)))

    disposal_activities.append('Aluminium waste')
    disposal_activities.append('Chromium Steel waste')

    return disposal_activities


def determine_location(row, iso_code):
    if row['Phase'] == 'Input':
        return ['GLO', 'RoW', 'RER', 'Europe without Switzerland', 'CH']
    elif row['Phase'] == 'Assembly':
        return [iso_code, 'RER', 'Europe without Switzerland', 'GLO', 'RoW']
    elif row['Phase'] == 'Transport':
        return [iso_code, 'RER', 'Europe without Switzerland', 'GLO', 'RoW']
    elif row['Phase'] == 'Maintenance':
        return [iso_code, 'RER', 'Europe without Switzerland']
    elif row['Phase'] == 'Disposal':
        return [iso_code, 'RER', 'Europe without Switzerland', 'RoW']
    else:
        return []

def find_activities(row, lon, lat, activity_names):
    """
    Find the first matching activity for each specified activity name in the eidb dataset. In case regional datasets are not available, it will return datasets for locations with the following hierarchy:
    'RER', 'Europe without Switzerland', 'GLO', 'RoW'.

    Parameters:
    - row (dict): A dictionary containing the phase information (e.g., {'Phase': 'Assembly'}).
    - iso_code (str): The ISO code to be used in the location hierarchy.
    - eidb (list): A list of activities (dictionaries) to search through.
    - activity_names (list): A list of activity names to search for.

    Returns:
    - dict: A dictionary where keys are activity names and values are the first matching activity found.
    """
    eidb = bd.Database('ecoinvent-391-cutoff')
    iso_code = get_iso_or_nearest_shore(lat, lon)
    preferred_locations = determine_location(row, iso_code)

    activities = {}

    for activity_name in activity_names:
        name_strip = activity_name.strip()

        filtered_db = [act for act in eidb if act['name'] == name_strip]

        if not filtered_db:
            print(f"⚠️ Activity '{activity_name}' not found at all.")
            continue

        # Match the first activity in preferred_locations order
        matching_activities = [
            act for loc in preferred_locations
            for act in filtered_db
            if act['location'] == loc
        ]

        if matching_activities:
            activities[activity_name] = matching_activities[0]
        else:
            print(f"⚠️ No matching location found for '{activity_name}'. Returning generic dataset.")
            # Optional fallback to first available
            activities[activity_name] = filtered_db[0]

    return activities


def find_activities_old(row, lon, lat, activity_names):
    """
    Find the first matching activity for each specified activity name in the eidb dataset.

    Parameters:
    - row (dict): A dictionary containing the phase information (e.g., {'Phase': 'Assembly'}).
    - iso_code (str): The ISO code to be used in the location hierarchy.
    - eidb (list): A list of activities (dictionaries) to search through.
    - activity_names (list): A list of activity names to search for.

    Returns:
    - dict: A dictionary where keys are activity names and values are the first matching activity found.
    """

    eidb = bd.Database('ecoinvent-391-cutoff')
    iso_code = get_iso_or_nearest_shore(lat, lon)

    # Get the list of assembly locations
    assembly_loc = determine_location(row, iso_code)

    # Dictionary to hold the results
    activities = {}

    # Iterate over each activity name and find the corresponding dataset
    for activity_name in activity_names:
        for loc in assembly_loc:
            # Search for the dataset with the given conditions
            matching_activities = [
                act for act in eidb 
                #if activity_name in act['name']
                if act['name']==activity_name.strip() 
                and loc in act['location']
            ]
            
            # If a matching dataset is found, store it and break the inner loop
            if matching_activities:
                activities[activity_name] = matching_activities[0]
                break

    # Return the dictionary with all found activities
    return activities

def transport_cement_elec(lon, lat):
    # Define the row, iso_code, and activity names
    row = {'Phase': 'Assembly'}
    activity_names = [
        'market for transport, freight, lorry >32 metric ton, EURO6',
        'market for transport, freight, inland waterways, barge',
        'market for cement, Portland',
        'market for electricity, high voltage'
    ]

    # Call the function to find all the specified activities
    found_activities = find_activities(row, lon, lat, activity_names)

    # Access the found activities by their name
    Truck_transport = found_activities.get('market for transport, freight, lorry >32 metric ton, EURO6')
    Ship_transport = found_activities.get('market for transport, freight, inland waterways, barge')
    Cement = found_activities.get('market for cement, Portland')
    Electricity_dataset = found_activities.get('market for electricity, high voltage')

    return Truck_transport, Ship_transport, Cement, Electricity_dataset

def steel_dataset(lon, lat):
    # Define the row, iso_code, and activity names
    row = {'Phase': 'Input'}
    steel_dataset = [
        'market for steel, low-alloyed'
    ]

    # Call the function to find all the specified activities
    Steel_dataset = find_activities(row, lon, lat, steel_dataset)

    # Access the found activities by their name
    Steel_dataset = Steel_dataset.get('market for steel, low-alloyed')

    return Steel_dataset

def calculate_closest_distance(lon, lat, print_stats=False):
    """
    Calculate the closest distance from a wind turbine location to the nearest bus in the bus_gdf.
    
    Parameters:
        lon (float): Longitude of the turbine.
        lat (float): Latitude of the turbine.
        bus_gdf (GeoDataFrame): GeoDataFrame containing bus locations.
        print_stats (bool): If True, print summary statistics and plot results. Default is False.
    
    Returns:
        float: Closest distance in meters.
    """
    # Load and prepare bus data
    bus_data = pd.read_csv(_DATA_DIR / "buses.csv")
    #bus_data = pd.read_csv(_DATA_DIR / "transformers.csv")
    bus_data['geometry'] = bus_data['geometry'].apply(wkt.loads)
    bus_gdf = gpd.GeoDataFrame(bus_data, geometry='geometry', crs="EPSG:4326")

    turbine_location = (lat, lon)
    
    # Calculate distances in meters
    bus_gdf['distance_meters'] = bus_gdf.geometry.apply(lambda x: geodesic(turbine_location, (x.y, x.x)).meters)
    
    # Get the closest distance
    closest_distance = bus_gdf['distance_meters'].min()

    # Print statistics and show plots if print_stats is True
    if print_stats:
        # Convert distance to kilometers for summary
        bus_gdf['distance_km'] = bus_gdf['distance_meters'] / 1000
        summary_stats = bus_gdf['distance_km'].describe(percentiles=[0.25, 0.5, 0.75]).to_frame()
        summary_stats.loc['mean'] = bus_gdf['distance_km'].mean()
        summary_stats.loc['std'] = bus_gdf['distance_km'].std()
        
        print("Statistical Summary (in kilometers):")
        print(summary_stats)

        # Identify 10 closest buses
        closest_buses = bus_gdf.nsmallest(10, 'distance_km')

        # Plotting
        plt.figure(figsize=(14, 6))

        # Histogram with median and interquartile range
        plt.subplot(1, 2, 1)
        plt.hist(bus_gdf['distance_km'], bins=30, edgecolor='k', alpha=0.7, color='skyblue')
        plt.axvline(summary_stats.loc['50%', 'distance_km'], color='orange', linestyle='--', label=f"Median: {summary_stats.loc['50%', 'distance_km']:.2f} km")
        plt.axvline(summary_stats.loc['25%', 'distance_km'], color='green', linestyle='--', label=f"25th Percentile: {summary_stats.loc['25%', 'distance_km']:.2f} km")
        plt.axvline(summary_stats.loc['75%', 'distance_km'], color='red', linestyle='--', label=f"75th Percentile: {summary_stats.loc['75%', 'distance_km']:.2f} km")
        plt.xlabel("Distance to Turbine (km)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Distances to Buses with IQR and Median")
        plt.legend()

        # Bar Chart for 10 Closest Buses with distance labels
        plt.subplot(1, 2, 2)
        bars = plt.bar(range(1, 11), closest_buses['distance_km'], color='steelblue')
        plt.xlabel("Closest Bus Rank")
        plt.ylabel("Distance to Turbine (km)")
        plt.title("Distances of the 10 Closest Buses to Turbine")
        plt.xticks(range(1, 11))

        # Adding labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    return closest_distance

# Function to get ISO country code for a given lat/lon
def get_iso_country_code(lat, lon):
    world = gpd.read_file(_DATA_DIR / "ne_10m_admin_0_countries.shp")
    point = Point(lon, lat)
    for _, country in world.iterrows():
        if country['geometry'].contains(point):
            return country['ISO_A2_EH']
    return None

# Function to find the nearest shore point (if the location is in the sea)
def find_nearest_shore(lat, lon, step_size=0.1, max_distance=5000):
    for distance in np.arange(0.1, max_distance, step_size):
        # Check points in a square grid around the original point
        for lat_offset in [-distance, 0, distance]:
            for lon_offset in [-distance, 0, distance]:
                if lat_offset == 0 and lon_offset == 0:
                    continue  # Skip the original point
                iso_code = get_iso_country_code(lat + lat_offset, lon + lon_offset)
                if iso_code:
                    return iso_code, (lat + lat_offset, lon + lon_offset)
    return None, (None, None)

# Combined function to get ISO code or nearest shore point
def get_iso_or_nearest_shore(lat, lon):
    iso_code = get_iso_country_code(lat, lon)
    
    if iso_code:
        return iso_code#, (lat, lon)
    else:
        nearest_iso_code, nearest_shore_point = find_nearest_shore(lat, lon)
        return nearest_iso_code#, nearest_shore_point

# Function to find the closest index in the dataset for a given latitude/longitude
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# Function to get sea depth for a given latitude and longitude
def get_sea_depth(lat, lon):
    balthymetry = nc.Dataset(_DATA_DIR / "GEBCO_2024_sub_ice_topo.nc")
    # Access the bathymetry data (usually stored in variables like 'elevation')
    # bathymetry_data = balthymetry.variables['elevation'][:]
    latitudes = balthymetry.variables['lat'][:]
    longitudes = balthymetry.variables['lon'][:]
    # Find the closest data points
    lat_idx = find_nearest(latitudes, lat)
    lon_idx = find_nearest(longitudes, lon)

    # Extract the depth (negative values indicate sea depth)
    sea_depth = balthymetry.variables['elevation'][lat_idx ,lon_idx]
    
    return float(sea_depth)

if __name__ == "__main__":
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
    row = {'Phase': 'Assembly'}
    activity_names = [
        'market for transport, freight, lorry >32 metric ton, EURO6',
        'market for transport, freight, inland waterways, barge',
        'market for cement, Portland',
        'market for electricity, high voltage'
    ]

    lon_it_on, lat_it_on = 14.422429, 41.029031

    # Call the function to find all the specified activities
    found_activities = find_activities(row, lon_it_on, lat_it_on, activity_names)
    print(found_activities)