from wimby_lca.built_inventory import create_dictionary_update
from wimby_lca.calculations import lca_wimby_fleet_evaluation, lca_wimby_fleet_evaluation_total_impacts
from pathlib import Path
import time
import pandas as pd

# Define data directory
#_DATA_DIR = Path(__file__).resolve().parent / "data"
_DATA_DIR = Path(__file__).resolve().parent / "wimby_lca" / "data"
ei_path = _DATA_DIR / "datasets"

# Load turbine data
eu_wind_turbines = pd.read_excel(_DATA_DIR / "EU_turbines_input_data.xlsx", sheet_name="EU_turbines_input_data")
print(f"Total turbines before filtering: {eu_wind_turbines.shape[0]}")

# After filtering to DE
country = "DE"
country_data = eu_wind_turbines[eu_wind_turbines['ISO_code'] == country].copy()
print(f"Total turbines in {country}: {country_data.shape[0]}")

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

# --- after you've built `country_data`, fixed dtypes, and added lifecycle stage columns ---

# Define climate change impact category (keep as in your script)
climate_change = ('EF v3.1', 'climate change', 'global warming potential (GWP100)')

# Define chunk size
chunk_size = 5000

# Split into chunks
chunks = [country_data.iloc[i:i + chunk_size] for i in range(0, len(country_data), chunk_size)]
print(f"Number of chunks for {country}: {len(chunks)}")

# >>> Choose which chunk to run (e.g. 0 for first 5000, 1 for second, etc.)
chunk_index = 2  # <-- change manually before each run
country_chunk = chunks[chunk_index].copy()
print(f"Running chunk {chunk_index+1} with {country_chunk.shape[0]} turbines")

# Process each turbine in this chunk
start_time = time.time()

for idx, row in country_chunk.iterrows():
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
            country_chunk.loc[idx, stage] = results_it_offshore_per_stage.get(stage, [0])[0]

    except Exception as e:
        print(f"Error processing turbine at index {idx} in {country}: {e}")
        for stage in lifecycle_stages_onshore + lifecycle_stages_offshore:
            country_chunk.loc[idx, stage] = 0

# Save results for this chunk
output_dir = Path(r"C:\Users\dhuber\OneDrive - Vrije Universiteit Brussel\DominikMaeva\01_Projects\05_WIMBY\05_LCA\Evaulation\Fleet_results")
output_path_chunk = output_dir / f"fleet_impacts_{country}{chunk_index+1}_normalized_kWh.csv"
country_chunk.to_csv(output_path_chunk, index=False)

print(f"Finished chunk {chunk_index+1} for {country}. Results saved to {output_path_chunk}. Execution time: {time.time() - start_time:.2f} sec")
