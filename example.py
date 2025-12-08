from wimby_lca.built_inventory import create_dictionary_update
from wimby_lca.calculations import lca_wimby_map, lca_detailed_materials, sum_materials
from wimby_lca.prepare_inventories import transport_cement_elec, find_activities, ecoinvent_setup, get_iso_or_nearest_shore, determine_location
from pathlib import Path
import bw2data as bd
import time

_DATA_DIR = Path(__file__).resolve().parent / "data"
ei_path = _DATA_DIR/ "datasets"

#done
aep_be_onshore = 9700*1000 #WIMBY interactive map
cf_be_onshore = 0.33
#aep_be_onshore = p_onshore*365*24*cf_be_onshore
lon_be_on, lat_be_on = 4.394870, 51.090789

#done
aep_be_offshore = 51600*1000 #WIMBY interactive map
cf_be_offshore = 0.56
#aep_be_offshore = p_offshore*365*24*cf_be_offshore
lon_be_off, lat_be_off =2.648846, 51.184471

start_time = time.time()
climate_change = ('EF v3.1', 'climate change', 'global warming potential (GWP100)')
mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")

dct_be_offshore = create_dictionary_update(P=10000, lon=lon_be_off, lat=lat_be_off, print_details=False)
#print(dct_be_offshore)

results_be_offshore_per_stage = lca_wimby_map(dict_activities=dct_be_offshore, impact_category=climate_change, aep=aep_be_offshore)
print(results_be_offshore_per_stage)

end_time= time.time()
elapsed_time = end_time - start_time
print(f"The function takes that much time:{elapsed_time}")