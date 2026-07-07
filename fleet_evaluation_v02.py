from built_inventory import create_dictionary_update
from calculations import lca_wimby_fleet_evaluation
from pathlib import Path
import time
import pandas as pd

# Define data directory
_DATA_DIR = Path(__file__).resolve().parent / "data"
ei_path = _DATA_DIR / "datasets"

# Load turbine data
eu_wind_turbines = pd.read_csv(_DATA_DIR / "EU_turbines_input_data.csv")
print(f"Total turbines before filtering: {eu_wind_turbines.shape[0]}")

# Filter to selected countries only (faster execution)
#selected_countries = ['BE', 'NO', 'IT', 'FR', 'DE']
selected_countries = ['DK']
#selected_countries = ['SK', 'IS','SI']
eu_wind_turbines = eu_wind_turbines[eu_wind_turbines['ISO_code'].isin(selected_countries)]
print(f"Total turbines after filtering: {eu_wind_turbines.shape[0]}")

# Ensure correct data types for each column
eu_wind_turbines['P_rated_kW'] = pd.to_numeric(eu_wind_turbines['P_rated_kW'], errors='coerce')
eu_wind_turbines['Longitude'] = pd.to_numeric(eu_wind_turbines['Longitude'], errors='coerce')
eu_wind_turbines['Latitude'] = pd.to_numeric(eu_wind_turbines['Latitude'], errors='coerce')
eu_wind_turbines['Hub_height_m'] = pd.to_numeric(eu_wind_turbines['Hub_height_m'], errors='coerce')
eu_wind_turbines['Diameter_m'] = pd.to_numeric(eu_wind_turbines['Diameter_m'], errors='coerce')
eu_wind_turbines['Lifetime_production_kWh'] = pd.to_numeric(eu_wind_turbines['Lifetime_production_kWh'], errors='coerce')
eu_wind_turbines['park_size'] = eu_wind_turbines['park_size'].astype(int)
eu_wind_turbines['Offshore'] = eu_wind_turbines['Offshore'].astype(int)
eu_wind_turbines['ISO_code'] = eu_wind_turbines['ISO_code'].astype('category')  # Optimized

# Define lifecycle stages
lifecycle_stages_onshore = ['Input', 'Assembly', 'Transport', 'Maintenance', 'Disposal', 'Total']
lifecycle_stages_offshore = ['Input', 'Assembly', 'Transport', 'Disposal', 'Total']

# Initialize lifecycle stage columns dynamically
for stage in lifecycle_stages_onshore:
    if stage not in eu_wind_turbines.columns:
        eu_wind_turbines[stage] = None

# Define climate change impact category
climate_change = ('EF v3.1', 'climate change', 'global warming potential (GWP100)')

# Start processing turbines
start_time_total = time.time()

for country in selected_countries:
    print(f"Starting simulation for {country}...")

    # Filter turbines by country
    country_data = eu_wind_turbines[eu_wind_turbines['ISO_code'] == country].copy()
    start_time = time.time()

    # Process each turbine
    for idx, row in country_data.iterrows():
        try:
            # Step 1: Create dictionary for inventories
            dct_it_offshore = create_dictionary_update(
                P=row['P_rated_kW'],
                lon=row['Longitude'],
                lat=row['Latitude'],
                h=row['Hub_height_m'],
                d=row['Diameter_m'],
                park_size=row['park_size'],
                print_details=False
            )

            # Step 2: Calculate LCA impacts
            results_it_offshore_per_stage = lca_wimby_fleet_evaluation(
                dict_activities=dct_it_offshore,
                impact_category=climate_change,
                aep=row['Lifetime_production_kWh']
            )

            # Handle missing results
            if results_it_offshore_per_stage is None:
                print(f"No results for turbine at index {idx} in {country}. Setting all stages to 0.")
                results_it_offshore_per_stage = {stage: [0] for stage in lifecycle_stages_onshore}

            # Store results based on offshore/onshore
            stages = lifecycle_stages_offshore if row['Offshore'] else lifecycle_stages_onshore
            for stage in stages:
                country_data.loc[idx, stage] = results_it_offshore_per_stage.get(stage, [0])[0]

        except Exception as e:
            print(f"Error processing turbine at index {idx} in {country}: {e}")
            for stage in lifecycle_stages_onshore + lifecycle_stages_offshore:
                country_data.loc[idx, stage] = 0

    # Save results for this country
    output_path_country = _DATA_DIR / f"fleet_impacts_{country}.csv"
    country_data.to_csv(output_path_country, index=False)
    print(f"Finished processing {country}. Results saved to {output_path_country}. Execution time: {time.time() - start_time:.2f} sec")

# Overall execution time
print(f"Finished processing selected countries. Total execution time: {time.time() - start_time_total:.2f} sec")
