import time
from logging import getLogger
import numpy as np
from numpy import random
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
import bw2data as bd
import numpy as np
from pathlib import Path

from prepare_inventories import ecoinvent_setup, prepare_inventory, activities_and_uuids, percentage_inventory, inventory_not_kg, assembly_activity, disposal_activity, get_sea_depth, get_iso_or_nearest_shore, calculate_closest_distance, transport_cement_elec, steel_dataset, get_activities_from_names
from power_transformer import transfo_500mva, transfo_10mva
from scaling import (add_to_dict_2, add_to_dict, find_values_sum,
    func_rotor_power, func_height_power, func_nacelle_weight_power, func_rotor_weight_rotor_diameter, func_tower_weight_d2h, scour_volume, grout_and_monopile_requirements,
    foundation_type, transport_requirements, cable_requirements, cable_requirements_v01, cable_requirements_Onshore, cable_requirements_Onshore_v01,
    semi_sub_floating_foundation, spar_buoy_floating_foundation)

logger = getLogger(__name__)
start_time = time.time()
_DATA_DIR = Path(__file__).resolve().parent / "data"

def create_dictionary_update(P, lon, lat, d = None, h = None,  M_tower = None, M_foundation = None, M_reinfSteel_foundation = None, 
                      V_conc_foundation = None, M_nacelle = None, M_power_supply = None, M_rotor = None, 
                      M_electronics = None, park_size = 50, dist_transfo = 1, dist_to_grid = None, 
                      sea_depth = None, print_details = False, lifetime = 20):
    """
    This function generates the lifecycle inventory of a wind turbine
    List of parameters : P (rated power) expressed in kW, d (rotor diameter) expressed in m, h (hub height) expressed in m,
    offshore = True/False (False by default), park_size (number of wind turbines in a park) as integer,
    dist_transfo (ditance to transformer) in meters, disct_coast (distance to coast for offshore) in meters,
    sea_depth (for offshore) in meters, print_details (to print details) as boolean and lifetime (in years) as integer.
    """
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
    df_act = activities_and_uuids( lon=lon, lat=lat)
    df_inv = prepare_inventory()
    df_perc = percentage_inventory()
    df_inv_not_kg = inventory_not_kg()
    assembly_activities = assembly_activity()
    disposal_activities = disposal_activity()

    diesel_burned_activity = get_activities_from_names(eidb, ['market for diesel, burned in building machine'])[0]
    MV_transfo = transfo_10mva()
    HV_transfo = transfo_500mva()
    Copper_wire_drawing = get_activities_from_names(eidb, ['market for wire drawing, copper'])[0]
    Explosive = get_activities_from_names(eidb, ['market for explosive, tovex'])[0]
    Steel_sheet_rolling = get_activities_from_names(eidb, ['market for sheet rolling, steel'])[0]
    Aluminium_sheet_rolling = get_activities_from_names(eidb, ['market for sheet rolling, aluminium'])[0]
    Chromium_sheet_rolling = get_activities_from_names(eidb, ['market for sheet rolling, chromium steel'])[0]
    Road = [act for act in eidb if act['name']=='market for road'.strip()][0]
    Digger = get_activities_from_names(eidb, ['market for excavation, hydraulic digger'])[0]
    iron = get_activities_from_names(eidb, ['market for iron ore, crude ore, 46% Fe'])[0]
    Truck_transport, Ship_transport, Cement, Electricity_dataset = transport_cement_elec(lon=lon, lat=lat)
    Steel_dataset = steel_dataset(lon=lon, lat=lat)

    if sea_depth is None:
        if get_sea_depth(lat, lon)<0:
            sea_depth = (get_sea_depth(lat, lon))*-1
        else:
            sea_depth = (get_sea_depth(lat, lon))*0

    # Determine if offshore if not explicitly provided
    offshore = sea_depth > 0  # Offshore if sea depth is negative (below sea level)
    iso_code = get_iso_or_nearest_shore(lat, lon)
    type_found = foundation_type(offshore, sea_depth)
    
    #Setting parameters for scaling model with onshore-offshore distinctions.
    if offshore==False:
        p_rotor_power = [152.66222073,   136.56772435,  2478.03511414,    16.44042379]
        p_height_power = [116.43035193, 91.64953366, 2391.88662558]
        p_nacelle_weight_power = [  1.66691134e-06,   3.20700974e-02] 
        p_rotor_weight_rotor_diameter = [ 0.00460956,  0.11199577]
    
    if offshore==True:
        p_rotor_power = [191.83651588,   147.37205671,  5101.28555377,   376.62814798]
        p_height_power = [120.75491612, 82.75390577, 4177.56520433]
        p_nacelle_weight_power = [  2.15668283e-06,   3.24712680e-02]
        p_rotor_weight_rotor_diameter = [ 0.0088365,  -0.16435292]

        
    #Using scaling model for missing values
    if d == None:
        d = func_rotor_power(P, *p_rotor_power)
    if h == None:
        h = func_height_power(P, *p_height_power)   
    if M_tower == None:
        M_tower = func_tower_weight_d2h(d, h, *[3.03584782e-04, 9.68652909e+00])
    if M_foundation == None:
        M_foundation = 1696e3 * h/80 * d**2/(100**2)
    if M_reinfSteel_foundation == None:
        M_reinfSteel_foundation = np.interp(P, [750, 2000, 4500], [10210, 27000, 51900])
    if V_conc_foundation == None:
        V_conc_foundation = (M_foundation - M_reinfSteel_foundation) / 2200
    if M_nacelle == None:
        M_nacelle = func_nacelle_weight_power(P, *p_nacelle_weight_power)
    if dist_to_grid == None:
        dist_to_grid = calculate_closest_distance(lon, lat) / 1000
    if M_power_supply == None:
        M_power_supply = 620
    if M_rotor == None:
        M_rotor = func_rotor_weight_rotor_diameter(d, *p_rotor_weight_rotor_diameter)
    if M_electronics == None:
        M_electronics = np.interp(P, [30, 150, 600, 800, 2000], [150, 300, 862, 1112, 3946])
    
    M_all = M_nacelle + M_tower + M_rotor + M_foundation + M_electronics

    trsp_truck_nacelle, trsp_truck_rotor, trsp_truck_tower, trsp_ship_tower,trsp_truck_foundation, trsp_end_of_life, trsp_maintenance_per_year, trsp_ship_offshore = transport_requirements(lon, lat, M_nacelle,  M_tower, M_rotor,  M_foundation,  M_all, lifetime)
    trsp_truck = trsp_truck_nacelle + trsp_truck_rotor + trsp_truck_tower + trsp_truck_foundation + trsp_end_of_life + trsp_maintenance_per_year
         
    #A reprendre avec tous les paramètres possibles
    if print_details:
        print('Nominal power : %s kW' %P)
        print(f'ISO country-code: {iso_code}')
        print(f"Sea depth/elevation: {sea_depth} meters")
        print(f"Offshore: {offshore}")
        print(f"Foundation type: {type_found}")
        print(f"Distance electricity grid: {dist_to_grid} (km)")
        print(f"Truck transportation: {trsp_truck} (tkm)")
        print('Rotor diameter : %s m'%d)
        print('Hub height : %s m'%h)
        print('Tower weight : %s kg'%M_tower)
        #print('Foundation weight : %s kg'%M_foundation)
        #print('Foundation reinforced steel: %s kg'%M_reinfSteel_foundation)
        #print('Foundation concrete volume: %s m3'%V_conc_foundation)
        print('Nacelle weight: %s kg'%M_nacelle)
        print('Rotor weight: %s'%M_rotor)
        print('Electronics: %s'%M_electronics)
        print('Total weight: %s kg'%M_all)

    
    dict_activities={}
    
    #Adding input element expressed in percentage of mass * M
    phase = 'Input'
    if print_details:
        print(phase)
    for component in set(df_inv[phase].columns.get_level_values(0)):
        if print_details:
            print('\t' + component)

        if component == 'Electronics':
            M = M_electronics

        if component == 'Foundation':
            continue           

        if component == 'Nacelle':
            M = M_nacelle

        if component == 'Power supply':
            M = M_power_supply

        if component == 'Rotor':
            M = M_rotor

        if component == 'Tower':
            M = M_tower 

        if print_details:
            print(M)

        for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset] #previous: df_inv
                if print_details:
                    print('\t \t \t' + dataset + '\t M = %s'%M + '\t pctg = %s' %(np.interp(P, df_inv_i.index.values, df_inv_i.values)) + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                add_to_dict_2(dict_activities, phase, component=component, sub_comp=sub_comp, key = bd.get_activity((df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M)


    #Adding onshore foundation:
    if offshore==False:
        if print_details:
            print('\t Onshore foundation')
            print('\t \t M_fond = %s'%M_foundation)
            print('\t \t concrete = %s' %V_conc_foundation + 'm3')
            print('\t \t reinforced steel = %s' %M_reinfSteel_foundation + 'kg')
        #V_conc_foundation
        add_to_dict_2(dict_activities, phase, component='Foundation', sub_comp='Concrete, 30MPa', key = bd.get_activity((df_act[(df_act['Component'] == 'Foundation')&(df_act['Market name'] == 'market for concrete, 30MPa')]['UUID'].iloc[0])), value = V_conc_foundation)
        #M_reinfSteel_foundation
        add_to_dict_2(dict_activities, phase, component='Foundation', sub_comp='Reinforced concrete', key = bd.get_activity((df_act[(df_act['Component'] == 'Foundation')&(df_act['Market name'] == 'market for waste reinforced concrete')]['UUID'].iloc[0])), value = M_reinfSteel_foundation)
    
    #Adding road
    phase = 'Assembly'
    if offshore==False:
        value = np.interp(P, [0, 2000],[0, 8000])/park_size
        add_to_dict_2(dict_activities, phase, key = bd.get_activity((Road.key)), value = value)

    #Adding land use
    if offshore==False:    
        phase = 'Input'
        if print_details:
            print('Land use')
        matching_activities = [act for act in eidb if 'land use for onshore wind turbines' in act['name'] and iso_code in act['location']]
        if len(matching_activities) == 1:
            lus_onshore = matching_activities[0]
            
            # Delete old biosphere exchanges
            old_biosphere_exchanges = [exc for exc in lus_onshore.exchanges() if exc['type'] == 'biosphere']
            for exc in old_biosphere_exchanges:
                exc.delete()

            # Add new biosphere exchanges
            for dataset in df_inv_not_kg['Input']['Foundation']['Land use']:
                # Clean and prepare data
                df_inv_not_kg_i = df_inv_not_kg['Input']['Foundation']['Land use'][dataset].dropna()
                df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce').dropna()
                df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce').dropna()

                # Interpolate values
                interpolated_value = np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)

                if print_details:
                    print(f'\t{dataset}\t {interpolated_value}')
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")

                # Create new exchange for the interpolated value
                act_uuid = df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']
                new_exc = lus_onshore.new_exchange(input=bd.get_activity(act_uuid), amount=interpolated_value, type='biosphere')
                new_exc.save()

            # Add production exchange if not already present
            if not any(exc['type'] == 'production' for exc in lus_onshore.exchanges()):
                lus_onshore.new_exchange(input=lus_onshore.key, amount=1.0, unit='unit', type='production').save()

            # Save the activity with all the new exchanges
            lus_onshore.save()

        # If no matches, or more than 1 match, create a new activity
        elif not matching_activities:
            lus_onshore = eidb.new_activity(name='land use for onshore wind turbines', unit='unit', location=iso_code, amount=1.0, type='process', product='land use for onshore wind turbines', code=random.randint(10000, 100000))
            lus_onshore.save()

            # Repeat adding biosphere exchanges (code same as above)
            
            for dataset in df_inv_not_kg['Input']['Foundation']['Land use']:
                df_inv_not_kg_i = df_inv_not_kg['Input']['Foundation']['Land use'][dataset].dropna()
                df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce').dropna()
                df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce').dropna()

                interpolated_value = np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)

                if print_details:
                    print(f'\t{dataset}\t {interpolated_value}')
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")

                act_uuid = df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']
                new_exc = lus_onshore.new_exchange(input=bd.get_activity(act_uuid), amount=interpolated_value, type='biosphere')
                new_exc.save()

            if not any(exc['type'] == 'production' for exc in lus_onshore.exchanges()):
                lus_onshore.new_exchange(input=lus_onshore.key, amount=1.0, unit='unit', type='production').save()
            lus_onshore.save()

        # Add to dictionary
        add_to_dict_2(dict_activities, phase, component="Foundation", sub_comp="Land use", key=lus_onshore, value=1.0/park_size)
        
    #Maintenance
    if offshore==False:
        phase = 'Maintenance'
        if print_details:
            print('Maintenance')
        dataset = 'Car [km]'
        df_inv_not_kg_i = df_inv_not_kg['Maintenance']['Nacelle']['Transport by car'][dataset].dropna()
        df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce')
        df_inv_not_kg_i = df_inv_not_kg_i.dropna()
        df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce')
        df_inv_not_kg_i = df_inv_not_kg_i.dropna()
        if print_details:
            print('\t' + dataset + '\t %s'%(np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)))
            print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
        add_to_dict_2(dict_activities, phase, key = bd.get_activity((df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values))

    #Assembly activities not in kg
    phase = 'Assembly'
    if print_details:
        print(phase + 'not in kg')
    for component in set(df_inv_not_kg[phase].columns.get_level_values(0)):
        if print_details:
            print('\t' + component)
        for sub_comp in set(df_inv_not_kg[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv_not_kg[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_not_kg_i = df_inv_not_kg[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print(dataset)
                if component == 'Foundation':
                    df_inv_not_kg_i = df_inv_not_kg[phase][component][sub_comp][dataset].dropna()
                    df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce')
                    df_inv_not_kg_i = df_inv_not_kg_i.dropna()
                    df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce')
                    df_inv_not_kg_i = df_inv_not_kg_i.dropna()
                    if not df_inv_not_kg_i.empty:
                        if print_details:
                            print('\t' + dataset + '\t %s' % (np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)))
                            print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                    else:
                        print(f"No valid data available for interpolation in dataset: {dataset}")
                else:
                    s = InterpolatedUnivariateSpline(df_inv_not_kg_i.index.values, df_inv_not_kg_i.values, k=1)
                    if print_details:
                        print('\t' + dataset + '\t %s'%(s(P)))
                        print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                    add_to_dict_2(dict_activities, phase, key = bd.get_activity((df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = s(P))

    #Connexion_requirements
    if offshore==True:
        if print_details:
            print('Offshore connection')
        M, E_CLS = cable_requirements_v01(P = P, lon = lon, lat = lat, park_size = park_size, dist_transfo = dist_transfo)#, dist_coast = dist_coast)
        
        phase = 'Input'
        component = 'Power supply'
        for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print('\t \t \t' + dataset + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                add_to_dict_2(dict_activities, phase, component=component, sub_comp=sub_comp, key = bd.get_activity((df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M)
        
        #Energy for cable laying ship
        phase = "Assembly"
        component = "Power supply" 
        add_to_dict_2(dict_activities, phase=phase, component=component, sub_comp='Energy for cable laying', key = bd.get_activity(key = (diesel_burned_activity.database, diesel_burned_activity.code)), value = E_CLS)
        
        #Transfo
        phase="Input"
        component = "Transformer"
        sub_comp = "HV/MV transformer"
        add_to_dict_2(dict_activities, phase=phase, key = MV_transfo, value = P / 10e3 / 0.85 * lifetime/35, component=component, sub_comp=sub_comp) #calculation prorated to power, factor 0.85 in active and apparent power and service life
        add_to_dict_2(dict_activities, phase=phase, key = HV_transfo, value = P / 500e3 / 0.85 * lifetime/35, component=component, sub_comp=sub_comp)


    # For onshore, medium-voltage transformer in proportion to power, and cable section and length according to power.
    if offshore==False:
        phase = "Input"
        component = "Transformer"
        sub_comp = "MV transformer"   
        add_to_dict_2(dict_activities, phase=phase, key = MV_transfo, value = P / 10e3 / 0.85 * 19/35, component=component, sub_comp=sub_comp)
        
        M  = cable_requirements_Onshore_v01(P, lon=lon, lat=lat)
        phase = 'Input'
        component = 'Power supply'
        
        for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print('\t \t \t' + dataset + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                add_to_dict_2(dict_activities, phase, component=component, sub_comp=sub_comp, key = bd.get_activity((df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M)

    #Transport activities
    phase = 'Transport'
    trsp_truck_nacelle, trsp_truck_rotor, trsp_truck_tower, trsp_ship_tower,trsp_truck_foundation, trsp_end_of_life, trsp_maintenance_per_year, trsp_ship_offshore = transport_requirements(lon, lat, M_nacelle,  M_tower, M_rotor,  M_foundation,  M_all, lifetime)
    
    #Truck transportation
    trsp_truck = trsp_truck_nacelle + trsp_truck_rotor + trsp_truck_tower + trsp_truck_foundation + trsp_end_of_life + trsp_maintenance_per_year
    add_to_dict_2(dict_activities, phase=phase, key = bd.get_activity(Truck_transport.key), value = trsp_truck)
    
    #Ship transportation
    if offshore==True:
        trsp_ship = trsp_ship_tower + trsp_ship_offshore
    else:
        trsp_ship = trsp_ship_tower
    
    add_to_dict_2(dict_activities, phase, key = bd.get_activity(Ship_transport.key), value = trsp_ship)

    #Scour stuff
    phase = 'Assembly'
    if offshore==True:
        scour_poly = scour_volume()
        scour_value = scour_poly(P)
        add_to_dict_2(dict_activities, phase, key = bd.get_activity(key = (Digger.database, Digger.code)), value = scour_value)

    phase = 'Input'
    if offshore==True:
        if sea_depth <= 30:  # Monopile foundation
            # Calculate monopile foundation requirements
            m_grout, m_monopile = grout_and_monopile_requirements(P, sea_depth)
        
            # Add monopile-related materials to the dictionary
            add_to_dict_2(dict_activities, phase, component='Foundation', sub_comp='Cement', key=bd.get_activity(Cement.key), value=m_grout)

            phase = 'Input'
            component = 'Foundation'
            sub_comp = 'Material'
            if sea_depth <= 30:
                M = m_monopile
            elif 30 < sea_depth <= 60:
                M = m_semi_sub
            else:
                M = m_spar_steel + m_spar_iron
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print('\t \t \t' + dataset + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                key_dict = bd.get_activity(df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == 'Input')].iloc[0]['UUID'])
                value_dict = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M
                add_to_dict_2(dict_activities, phase, component=component, sub_comp=sub_comp, key = key_dict, value = value_dict)


        elif 30 < sea_depth <= 60:  # Semi-submersible foundation
            # Calculate semi-submersible foundation requirements
            m_semi_sub = semi_sub_floating_foundation(P, sea_depth)
        
            # Add semi-submersible-related materials to the dictionary
            add_to_dict_2(dict_activities, phase, component='Semi-submersible foundation', sub_comp='Steel', key=(Steel_dataset), value=m_semi_sub)
    
        else:  # Spar buoy foundation (sea depth > 60)
            # Calculate spar buoy foundation requirements
            m_spar_steel, m_spar_iron = spar_buoy_floating_foundation(P, sea_depth)
        
            # Add spar buoy-related materials to the dictionary
            add_to_dict_2(dict_activities, phase, component='Spar buoy foundation', sub_comp='Plattform, anchor and chains', key=(Steel_dataset), value=m_spar_steel)
            add_to_dict_2(dict_activities, phase, component='Spar buoy foundation', sub_comp='Balast', key=bd.get_activity(key = (iron.database, iron.code)), value=m_spar_iron)

    #Assembly depending on kg
    phase = 'Assembly'
    for AA in  assembly_activities:
        if print_details:
            print(AA)
        if AA == 'Explosives':
            value = 10
            add_to_dict_2(dict_activities, phase, key = bd.get_activity(key = (Explosive.database, Explosive.code)), value = value )     
        if AA == 'Copper wire drawing':
            target_key = bd.get_activity(df_act[df_act['Dataset'] == 'Copper']['UUID'].iloc[0])
            value = find_values_sum(dict_activities, target_key)
            add_to_dict_2(dict_activities, phase, key =bd.get_activity(key=(Copper_wire_drawing.database, Copper_wire_drawing.code)), value=value)
        if AA == 'Steel sheet rolling':
            target_key_1 = bd.get_activity(df_act[df_act['Dataset'] == 'Low-alloy steel']['UUID'].iloc[0])
            value_1 = find_values_sum(dict_activities, target_key_1)
            target_key_2 = bd.get_activity(df_act[df_act['Dataset'] == 'Cast iron']['UUID'].iloc[0])
            value_2 = find_values_sum(dict_activities, target_key_2)
            add_to_dict_2(dict_activities, phase, key=bd.get_activity(key = (Steel_sheet_rolling.database, Steel_sheet_rolling.code)), value=value_1+value_2)
        if AA == "Aluminium sheet rolling":
            target_key = bd.get_activity(df_act[df_act['Dataset'] == 'Aluminium 0% recycled']['UUID'].iloc[0])
            value = find_values_sum(dict_activities, target_key)
            add_to_dict_2(dict_activities, phase, key=bd.get_activity(key = (Aluminium_sheet_rolling.database, Aluminium_sheet_rolling.code)), value=value)
        if AA == 'Chromium steel sheet rolling': #before: Chromium steel sheet rolling
            target_key = bd.get_activity(df_act[df_act['Dataset'] == 'Chromium steel']['UUID'].iloc[0])
            value = find_values_sum(dict_activities, target_key)
            add_to_dict_2(dict_activities, phase, key=bd.get_activity(key = (Chromium_sheet_rolling.database, Chromium_sheet_rolling.code)), value=value)

    #Disposal activities
    phase= 'Disposal'
    for DA in  disposal_activities:
        if print_details:
            print(DA)
        uuid = df_act[(df_act['Dataset'] == DA) & (df_act['Phase'] == 'Disposal')]['UUID'].iloc[0]

        # Check if the uuid is already a tuple or just a string
        if isinstance(uuid, str):
            ei_key_disposal = (eidb.name, uuid)  # Add database name if UUID is a plain string
        elif isinstance(uuid, tuple) and len(uuid) == 2:
            ei_key_disposal = uuid  # Leave unchanged if already a correct tuple
        DA = DA.replace(' -waste','').replace('Steel, inert waste', 'Low-alloy steel' ).replace('Concrete, inert waste', 'Concrete [m3]').replace('Aluminium waste', 'Aluminium 0% recycled').replace('Chromium Steel waste','Chromium steel')
        ei_key_input = df_act[ (df_act['Dataset'] == DA) & (df_act['Phase'] == 'Input')]['UUID'].iloc[0]
        try:
            value = find_values_sum(dict_activities, ei_key_input)
            add_to_dict_2(dict_activities, phase, key = bd.get_activity(ei_key_disposal), value = value)
        except:
            pass

    #Electicity dataset
    phase= 'Assembly'
    value = 0.5 * (M_nacelle + M_rotor + M_tower)
    add_to_dict_2(dict_activities, phase, key = bd.get_activity(Electricity_dataset.key), value = value)

    return dict_activities

def create_wind_lca_dk(P, lon, lat, d = None, h = None,  M_tower = None, M_foundation = None, M_reinfSteel_foundation = None, 
                      V_conc_foundation = None, M_nacelle = None, M_power_supply = None, M_rotor = None, 
                      M_electronics = None, offshore = False, park_size = 50, dist_transfo = 1, dist_coast = 5,
                      sea_depth = 5, print_details = False, lifetime = 20):
    """
    This function generates the lifecycle inventory of a wind turbine
    List of parameters : P (rated power) expressed in kW, d (rotor diameter) expressed in m, h (hub height) expressed in m,
    offshore = True/False (False by default), park_size (number of wind turbines in a park) as integer,
    dist_transfo (ditance to transformer) in meters, disct_coast (distance to coast for offshore) in meters,
    sea_depth (for offshore) in meters, print_details (to print details) as boolean and lifetime (in years) as integer.
    """
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
    df_act = activities_and_uuids( lon=lon, lat=lat)
    df_inv = prepare_inventory()
    df_perc = percentage_inventory()
    df_inv_not_kg = inventory_not_kg()
    assembly_activities = assembly_activity()
    disposal_activities = disposal_activity()

    diesel_burned_activity = get_activities_from_names(eidb, ['market for diesel, burned in building machine'])[0]
    MV_transfo = transfo_10mva()
    HV_transfo = transfo_500mva()
    Copper_wire_drawing = get_activities_from_names(eidb, ['market for wire drawing, copper'])[0]
    Explosive = get_activities_from_names(eidb, ['market for explosive, tovex'])[0]
    Steel_sheet_rolling = get_activities_from_names(eidb, ['market for sheet rolling, steel'])[0]
    Aluminium_sheet_rolling = get_activities_from_names(eidb, ['market for sheet rolling, aluminium'])[0]
    Chromium_sheet_rolling = get_activities_from_names(eidb, ['market for sheet rolling, chromium steel'])[0]
    Road = [act for act in eidb if act['name']=='market for road'.strip()][0]
    Digger = get_activities_from_names(eidb, ['market for excavation, hydraulic digger'])[0]
    Truck_transport, Ship_transport, Cement, Electricity_dataset = transport_cement_elec(lon=lon, lat=lat)
    #Steel_dataset = steel_dataset(lon=lon, lat=lat)
    
    #Setting parameters for scaling model with onshore-offshore distinctions.
    if offshore==False:
        p_rotor_power = [152.66222073,   136.56772435,  2478.03511414,    16.44042379]
        p_height_power = [116.43035193, 91.64953366, 2391.88662558]
        p_nacelle_weight_power = [  1.66691134e-06,   3.20700974e-02] 
        p_rotor_weight_rotor_diameter = [ 0.00460956,  0.11199577]
    
    if offshore==True:
        p_rotor_power = [191.83651588,   147.37205671,  5101.28555377,   376.62814798]
        p_height_power = [120.75491612, 82.75390577, 4177.56520433]
        p_nacelle_weight_power = [  2.15668283e-06,   3.24712680e-02]
        p_rotor_weight_rotor_diameter = [ 0.0088365,  -0.16435292]

        
    #Using scaling model for missing values
    if d == None:
        d = func_rotor_power(P, *p_rotor_power)
    if h == None:
        h = func_height_power(P, *p_height_power)   
    if M_tower == None:
        M_tower = func_tower_weight_d2h(d, h, *[3.03584782e-04, 9.68652909e+00])
    if M_foundation == None:
        M_foundation = 1696e3 * h/80 * d**2/(100**2)
    if M_reinfSteel_foundation == None:
        M_reinfSteel_foundation = np.interp(P, [750, 2000, 4500], [10210, 27000, 51900])
    if V_conc_foundation == None:
        V_conc_foundation = (M_foundation - M_reinfSteel_foundation) / 2200
    if M_nacelle == None:
        M_nacelle = func_nacelle_weight_power(P, *p_nacelle_weight_power)
    if M_power_supply == None:
        M_power_supply = 620
    if M_rotor == None:
        M_rotor = func_rotor_weight_rotor_diameter(d, *p_rotor_weight_rotor_diameter)
    if M_electronics == None:
        M_electronics = np.interp(P, [30, 150, 600, 800, 2000], [150, 300, 862, 1112, 3946])
    
    M_all = M_nacelle + M_tower + M_rotor + M_foundation + M_electronics

    trsp_truck_nacelle, trsp_truck_rotor, trsp_truck_tower, trsp_ship_tower,trsp_truck_foundation, trsp_end_of_life, trsp_maintenance_per_year, trsp_ship_offshore = transport_requirements(lon, lat, M_nacelle,  M_tower, M_rotor,  M_foundation,  M_all, lifetime)
    trsp_truck = trsp_truck_nacelle + trsp_truck_rotor + trsp_truck_tower + trsp_truck_foundation + trsp_end_of_life + trsp_maintenance_per_year
         
    #A reprendre avec tous les paramètres possibles
    if print_details:
        print('Nominal power : %s kW' %P)
        print('Rotor diameter : %s m'%d)
        print('Hub height : %s m'%h)
        print('Tower weight : %s kg'%M_tower)
        print('Foundation weight : %s kg'%M_foundation)
        print('Foundation reinforced steel: %s kg'%M_reinfSteel_foundation)
        print('Foundation concrete volume: %s m3'%V_conc_foundation)
        print('Nacelle weight: %s kg'%M_nacelle)
        print('Rotor weight: %s'%M_rotor)
        print('Electronics: %s'%M_electronics)
        print('Total weight: %s kg'%M_all)

    
    dict_activities={}
    
    #Adding input element expressed in percentage of mass * M
    phase = 'Input'
    if print_details:
        print(phase)
    for component in set(df_inv[phase].columns.get_level_values(0)):
        if print_details:
            print('\t' + component)

        if component == 'Electronics':
            M = M_electronics

        if component == 'Foundation':
            continue           

        if component == 'Nacelle':
            M = M_nacelle

        if component == 'Power supply':
            M = M_power_supply

        if component == 'Rotor':
            M = M_rotor

        if component == 'Tower':
            M = M_tower 

        if print_details:
            print(M)

        for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset] #previous: df_inv
                if print_details:
                    print('\t \t \t' + dataset + '\t M = %s'%M + '\t pctg = %s' %(np.interp(P, df_inv_i.index.values, df_inv_i.values)) + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                add_to_dict_2(dict_activities, phase, component=component, sub_comp=sub_comp, key = bd.get_activity((df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M)


    #Adding onshore foundation:
    if offshore==False:
        if print_details:
            print('\t Onshore foundation')
            print('\t \t M_fond = %s'%M_foundation)
            print('\t \t concrete = %s' %V_conc_foundation + 'm3')
            print('\t \t reinforced steel = %s' %M_reinfSteel_foundation + 'kg')
        #V_conc_foundation
        add_to_dict_2(dict_activities, phase, component='Foundation', sub_comp='Concrete, 30MPa', key = bd.get_activity((df_act[(df_act['Component'] == 'Foundation')&(df_act['Market name'] == 'market for concrete, 30MPa')]['UUID'].iloc[0])), value = V_conc_foundation)
        #M_reinfSteel_foundation
        add_to_dict_2(dict_activities, phase, component='Foundation', sub_comp='Reinforced concrete', key = bd.get_activity((df_act[(df_act['Component'] == 'Foundation')&(df_act['Market name'] == 'market for waste reinforced concrete')]['UUID'].iloc[0])), value = M_reinfSteel_foundation)
    
    #Adding road
    phase = 'Assembly'
    if offshore==False:
        value = np.interp(P, [0, 2000],[0, 8000])
        add_to_dict_2(dict_activities, phase, key = bd.get_activity((Road.key)), value = value)

    #Adding land use
    if False:    
        phase = 'Input'
        if print_details:
            print('Land use')
        matching_activities = [act for act in eidb if 'land use for onshore wind turbines' in act['name'] and iso_code in act['location']]
        if len(matching_activities) == 1:
            lus_onshore = matching_activities[0]
            
            # Delete old biosphere exchanges
            old_biosphere_exchanges = [exc for exc in lus_onshore.exchanges() if exc['type'] == 'biosphere']
            for exc in old_biosphere_exchanges:
                exc.delete()

            # Add new biosphere exchanges
            for dataset in df_inv_not_kg['Input']['Foundation']['Land use']:
                # Clean and prepare data
                df_inv_not_kg_i = df_inv_not_kg['Input']['Foundation']['Land use'][dataset].dropna()
                df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce').dropna()
                df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce').dropna()

                # Interpolate values
                interpolated_value = np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)

                if print_details:
                    print(f'\t{dataset}\t {interpolated_value}')
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")

                # Create new exchange for the interpolated value
                act_uuid = df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']
                new_exc = lus_onshore.new_exchange(input=bd.get_activity(act_uuid), amount=interpolated_value, type='biosphere')
                new_exc.save()

            # Add production exchange if not already present
            if not any(exc['type'] == 'production' for exc in lus_onshore.exchanges()):
                lus_onshore.new_exchange(input=lus_onshore.key, amount=1.0, unit='unit', type='production').save()

            # Save the activity with all the new exchanges
            lus_onshore.save()

        # If no matches, or more than 1 match, create a new activity
        elif not matching_activities:
            lus_onshore = eidb.new_activity(name='land use for onshore wind turbines', unit='unit', location=iso_code, amount=1.0, type='process', product='land use for onshore wind turbines', code=random.randint(10000, 100000))
            lus_onshore.save()

            # Repeat adding biosphere exchanges (code same as above)
            
            for dataset in df_inv_not_kg['Input']['Foundation']['Land use']:
                df_inv_not_kg_i = df_inv_not_kg['Input']['Foundation']['Land use'][dataset].dropna()
                df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce').dropna()
                df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce').dropna()

                interpolated_value = np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)

                if print_details:
                    print(f'\t{dataset}\t {interpolated_value}')
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")

                act_uuid = df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']
                new_exc = lus_onshore.new_exchange(input=bd.get_activity(act_uuid), amount=interpolated_value, type='biosphere')
                new_exc.save()

            if not any(exc['type'] == 'production' for exc in lus_onshore.exchanges()):
                lus_onshore.new_exchange(input=lus_onshore.key, amount=1.0, unit='unit', type='production').save()
            lus_onshore.save()

        # Add to dictionary
        add_to_dict_2(dict_activities, phase, component="Foundation", sub_comp="Land use", key=lus_onshore, value=1.0/park_size)
        
    #Maintenance
    if False:
        phase = 'Maintenance'
        if print_details:
            print('Maintenance')
        dataset = 'Car [km]'
        df_inv_not_kg_i = df_inv_not_kg['Maintenance']['Nacelle']['Transport by car'][dataset].dropna()
        df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce')
        df_inv_not_kg_i = df_inv_not_kg_i.dropna()
        df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce')
        df_inv_not_kg_i = df_inv_not_kg_i.dropna()
        if print_details:
            print('\t' + dataset + '\t %s'%(np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)))
            print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
        add_to_dict_2(dict_activities, phase, key = bd.get_activity((df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values))

    #Assembly activities not in kg
    phase = 'Assembly'
    if print_details:
        print(phase + 'not in kg')
    for component in set(df_inv_not_kg[phase].columns.get_level_values(0)):
        if print_details:
            print('\t' + component)
        for sub_comp in set(df_inv_not_kg[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv_not_kg[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_not_kg_i = df_inv_not_kg[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print(dataset)
                if component == 'Foundation':
                    df_inv_not_kg_i = df_inv_not_kg[phase][component][sub_comp][dataset].dropna()
                    df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce')
                    df_inv_not_kg_i = df_inv_not_kg_i.dropna()
                    df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce')
                    df_inv_not_kg_i = df_inv_not_kg_i.dropna()
                    if not df_inv_not_kg_i.empty:
                        if print_details:
                            print('\t' + dataset + '\t %s' % (np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)))
                            print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                    else:
                        print(f"No valid data available for interpolation in dataset: {dataset}")
                else:
                    s = InterpolatedUnivariateSpline(df_inv_not_kg_i.index.values, df_inv_not_kg_i.values, k=1)
                    if print_details:
                        print('\t' + dataset + '\t %s'%(s(P)))
                        print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                    add_to_dict_2(dict_activities, phase, key = bd.get_activity((df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = s(P))

    #Connexion_requirements
    if offshore==True:
        if print_details:
            print('Offshore connection')
        M, E_CLS = cable_requirements(P = P, park_size = park_size, dist_transfo = dist_transfo, dist_coast = dist_coast)
        
        phase = 'Input'
        component = 'Power supply'
        for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print('\t \t \t' + dataset + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                add_to_dict_2(dict_activities, phase, component=component, sub_comp=sub_comp, key = bd.get_activity((df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M)
        
        #Energy for cable laying ship
        phase = "Assembly"
        component = "Power supply" 
        add_to_dict_2(dict_activities, phase=phase, component=component, sub_comp='Energy for cable laying', key = bd.get_activity(key = (diesel_burned_activity.database, diesel_burned_activity.code)), value = E_CLS)
        
        #Transfo
        phase="Input"
        component = "Transformer"
        sub_comp = "HV/MV transformer"
        add_to_dict_2(dict_activities, phase=phase, key = MV_transfo, value = P / 10e3 / 0.85 * lifetime/35, component=component, sub_comp=sub_comp) #calculation prorated to power, factor 0.85 in active and apparent power and service life
        add_to_dict_2(dict_activities, phase=phase, key = HV_transfo, value = P / 500e3 / 0.85 * lifetime/35, component=component, sub_comp=sub_comp)


    # For onshore, medium-voltage transformer in proportion to power, and cable section and length according to power.
    if offshore==False:
        phase = "Input"
        component = "Transformer"
        sub_comp = "MV transformer"   
        add_to_dict_2(dict_activities, phase=phase, key = MV_transfo, value = P / 10e3 / 0.85 * 19/35, component=component, sub_comp=sub_comp)
        
        M  = cable_requirements_Onshore(P)
        phase = 'Input'
        component = 'Power supply'
        
        for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print('\t \t \t' + dataset + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                add_to_dict_2(dict_activities, phase, component=component, sub_comp=sub_comp, key = bd.get_activity((df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M)

    #Transport activities
    phase = 'Transport'
    trsp_truck_nacelle, trsp_truck_rotor, trsp_truck_tower, trsp_ship_tower,trsp_truck_foundation, trsp_end_of_life, trsp_maintenance_per_year, trsp_ship_offshore = transport_requirements(lon, lat, M_nacelle,  M_tower, M_rotor,  M_foundation,  M_all, lifetime)
    
    #Truck transportation
    trsp_truck = trsp_truck_nacelle + trsp_truck_rotor + trsp_truck_tower + trsp_truck_foundation + trsp_end_of_life + trsp_maintenance_per_year
    add_to_dict_2(dict_activities, phase=phase, key = bd.get_activity(Truck_transport.key), value = trsp_truck)
    
    #Ship transportation
    if offshore==True:
        trsp_ship = trsp_ship_tower + trsp_ship_offshore
    else:
        trsp_ship = trsp_ship_tower
    
    add_to_dict_2(dict_activities, phase, key = bd.get_activity(Ship_transport.key), value = trsp_ship)

    #Scour stuff
    phase = 'Assembly'
    if offshore==True:
        scour_poly = scour_volume()
        scour_value = scour_poly(P)
        add_to_dict_2(dict_activities, phase, key = bd.get_activity(key = (Digger.database, Digger.code)), value = scour_value)

    phase = 'Input'
    if offshore==True:
        if sea_depth <= 30:  # Monopile foundation
            # Calculate monopile foundation requirements
            m_grout, m_monopile = grout_and_monopile_requirements(P, sea_depth)
        
            # Add monopile-related materials to the dictionary
            add_to_dict_2(dict_activities, phase, component='Foundation', sub_comp='Cement', key=bd.get_activity(Cement.key), value=m_grout)

            phase = 'Input'
            component = 'Foundation'
            sub_comp = 'Material'
            M = m_monopile
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print('\t \t \t' + dataset + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                key_dict = bd.get_activity(df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == 'Input')].iloc[0]['UUID'])
                value_dict = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M
                add_to_dict_2(dict_activities, phase, component=component, sub_comp=sub_comp, key = key_dict, value = value_dict)

    #Assembly depending on kg
    phase = 'Assembly'
    for AA in  assembly_activities:
        if print_details:
            print(AA)
        if AA == 'Explosives':
            value = 10
            add_to_dict_2(dict_activities, phase, key = bd.get_activity(key = (Explosive.database, Explosive.code)), value = value )     
        if AA == 'Copper wire drawing':
            target_key = bd.get_activity(df_act[df_act['Dataset'] == 'Copper']['UUID'].iloc[0])
            value = find_values_sum(dict_activities, target_key)
            add_to_dict_2(dict_activities, phase, key =bd.get_activity(key=(Copper_wire_drawing.database, Copper_wire_drawing.code)), value=value)
        if AA == 'Steel sheet rolling':
            target_key_1 = bd.get_activity(df_act[df_act['Dataset'] == 'Low-alloy steel']['UUID'].iloc[0])
            value_1 = find_values_sum(dict_activities, target_key_1)
            target_key_2 = bd.get_activity(df_act[df_act['Dataset'] == 'Cast iron']['UUID'].iloc[0])
            value_2 = find_values_sum(dict_activities, target_key_2)
            add_to_dict_2(dict_activities, phase, key=bd.get_activity(key = (Steel_sheet_rolling.database, Steel_sheet_rolling.code)), value=value_1+value_2)
        if AA == "Aluminium sheet rolling":
            target_key = bd.get_activity(df_act[df_act['Dataset'] == 'Aluminium 0% recycled']['UUID'].iloc[0])
            value = find_values_sum(dict_activities, target_key)
            add_to_dict_2(dict_activities, phase, key=bd.get_activity(key = (Aluminium_sheet_rolling.database, Aluminium_sheet_rolling.code)), value=value)
        if AA == 'Chromium steel sheet rolling': #before: Chromium steel sheet rolling
            target_key = bd.get_activity(df_act[df_act['Dataset'] == 'Chromium steel']['UUID'].iloc[0])
            value = find_values_sum(dict_activities, target_key)
            add_to_dict_2(dict_activities, phase, key=bd.get_activity(key = (Chromium_sheet_rolling.database, Chromium_sheet_rolling.code)), value=value)

    #Disposal activities
    phase= 'Disposal'
    for DA in  disposal_activities:
        if print_details:
            print(DA)
        uuid = df_act[(df_act['Dataset'] == DA) & (df_act['Phase'] == 'Disposal')]['UUID'].iloc[0]

        # Check if the uuid is already a tuple or just a string
        if isinstance(uuid, str):
            ei_key_disposal = (eidb.name, uuid)  # Add database name if UUID is a plain string
        elif isinstance(uuid, tuple) and len(uuid) == 2:
            ei_key_disposal = uuid  # Leave unchanged if already a correct tuple
        DA = DA.replace(' -waste','').replace('Steel, inert waste', 'Low-alloy steel' ).replace('Concrete, inert waste', 'Concrete [m3]').replace('Aluminium waste', 'Aluminium 0% recycled').replace('Chromium Steel waste','Chromium steel')
        ei_key_input = df_act[ (df_act['Dataset'] == DA) & (df_act['Phase'] == 'Input')]['UUID'].iloc[0]
        try:
            value = find_values_sum(dict_activities, ei_key_input)
            add_to_dict_2(dict_activities, phase, key = bd.get_activity(ei_key_disposal), value = value)
        except:
            pass

    #Electicity dataset
    phase= 'Assembly'
    value = 0.5 * (M_nacelle + M_rotor + M_tower)
    add_to_dict_2(dict_activities, phase, key = bd.get_activity(Electricity_dataset.key), value = value)

    return dict_activities

def create_dictionary_for_monte_carlo(P, lon, lat, d = None, h = None,  M_tower = None, M_foundation = None, M_reinfSteel_foundation = None, 
                      V_conc_foundation = None, M_nacelle = None, M_power_supply = None, M_rotor = None, 
                      M_electronics = None, park_size = 50, dist_transfo = 1, dist_to_grid = None, 
                      sea_depth = None, print_details = False, lifetime = 20):
    """
    This function generates the lifecycle inventory of a wind turbine
    List of parameters : P (rated power) expressed in kW, d (rotor diameter) expressed in m, h (hub height) expressed in m,
    offshore = True/False (False by default), park_size (number of wind turbines in a park) as integer,
    dist_transfo (ditance to transformer) in meters, disct_coast (distance to coast for offshore) in meters,
    sea_depth (for offshore) in meters, print_details (to print details) as boolean and lifetime (in years) as integer.
    """
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
    df_act = activities_and_uuids( lon=lon, lat=lat)
    df_inv = prepare_inventory()
    df_perc = percentage_inventory()
    df_inv_not_kg = inventory_not_kg()
    assembly_activities = assembly_activity()
    disposal_activities = disposal_activity()

    diesel_burned_activity = get_activities_from_names(eidb, ['market for diesel, burned in building machine'])[0]
    MV_transfo = transfo_10mva()
    HV_transfo = transfo_500mva()
    Copper_wire_drawing = get_activities_from_names(eidb, ['market for wire drawing, copper'])[0]
    Explosive = get_activities_from_names(eidb, ['market for explosive, tovex'])[0]
    Steel_sheet_rolling = get_activities_from_names(eidb, ['market for sheet rolling, steel'])[0]
    Aluminium_sheet_rolling = get_activities_from_names(eidb, ['market for sheet rolling, aluminium'])[0]
    Chromium_sheet_rolling = get_activities_from_names(eidb, ['market for sheet rolling, chromium steel'])[0]
    Road = [act for act in eidb if act['name']=='market for road'.strip()][0]
    Digger = get_activities_from_names(eidb, ['market for excavation, hydraulic digger'])[0]
    iron = get_activities_from_names(eidb, ['market for iron ore, crude ore, 46% Fe'])[0]
    Truck_transport, Ship_transport, Cement, Electricity_dataset = transport_cement_elec(lon=lon, lat=lat)
    Steel_dataset = steel_dataset(lon=lon, lat=lat)

    if sea_depth is None:
        if get_sea_depth(lat, lon)<0:
            sea_depth = (get_sea_depth(lat, lon))*-1
        else:
            sea_depth = (get_sea_depth(lat, lon))*0

    # Determine if offshore if not explicitly provided
    offshore = sea_depth > 0  # Offshore if sea depth is negative (below sea level)
    iso_code = get_iso_or_nearest_shore(lat, lon)
    type_found = foundation_type(offshore, sea_depth)
    
    #Setting parameters for scaling model with onshore-offshore distinctions.
    if offshore==False:
        p_rotor_power = [152.66222073,   136.56772435,  2478.03511414,    16.44042379]
        p_height_power = [116.43035193, 91.64953366, 2391.88662558]
        p_nacelle_weight_power = [  1.66691134e-06,   3.20700974e-02] 
        p_rotor_weight_rotor_diameter = [ 0.00460956,  0.11199577]
    
    if offshore==True:
        p_rotor_power = [191.83651588,   147.37205671,  5101.28555377,   376.62814798]
        p_height_power = [120.75491612, 82.75390577, 4177.56520433]
        p_nacelle_weight_power = [  2.15668283e-06,   3.24712680e-02]
        p_rotor_weight_rotor_diameter = [ 0.0088365,  -0.16435292]

        
    #Using scaling model for missing values
    if d == None:
        d = func_rotor_power(P, *p_rotor_power)
    if h == None:
        h = func_height_power(P, *p_height_power)   
    if M_tower == None:
        M_tower = func_tower_weight_d2h(d, h, *[3.03584782e-04, 9.68652909e+00])
    if M_foundation == None:
        M_foundation = 1696e3 * h/80 * d**2/(100**2)
    if M_reinfSteel_foundation == None:
        M_reinfSteel_foundation = np.interp(P, [750, 2000, 4500], [10210, 27000, 51900])
    if V_conc_foundation == None:
        V_conc_foundation = (M_foundation - M_reinfSteel_foundation) / 2200
    if M_nacelle == None:
        M_nacelle = func_nacelle_weight_power(P, *p_nacelle_weight_power)
    if dist_to_grid == None:
        dist_to_grid = calculate_closest_distance(lon, lat) / 1000
    if M_power_supply == None:
        M_power_supply = 620
    if M_rotor == None:
        M_rotor = func_rotor_weight_rotor_diameter(d, *p_rotor_weight_rotor_diameter)
    if M_electronics == None:
        M_electronics = np.interp(P, [30, 150, 600, 800, 2000], [150, 300, 862, 1112, 3946])
    
    M_all = M_nacelle + M_tower + M_rotor + M_foundation + M_electronics

    trsp_truck_nacelle, trsp_truck_rotor, trsp_truck_tower, trsp_ship_tower,trsp_truck_foundation, trsp_end_of_life, trsp_maintenance_per_year, trsp_ship_offshore = transport_requirements(lon, lat, M_nacelle,  M_tower, M_rotor,  M_foundation,  M_all, lifetime)
    trsp_truck = trsp_truck_nacelle + trsp_truck_rotor + trsp_truck_tower + trsp_truck_foundation + trsp_end_of_life + trsp_maintenance_per_year
         
    if print_details:
        print('Nominal power : %s kW' %P)
        print(f'ISO country-code: {iso_code}')
        print(f"Sea depth/elevation: {sea_depth} meters")
        print(f"Offshore: {offshore}")
        print(f"Foundation type: {type_found}")
        print(f"Distance electricity grid: {dist_to_grid} (km)")
        print(f"Truck transportation: {trsp_truck} (tkm)")
        print('Rotor diameter : %s m'%d)
        print('Hub height : %s m'%h)
        print('Tower weight : %s kg'%M_tower)
        #print('Foundation weight : %s kg'%M_foundation)
        #print('Foundation reinforced steel: %s kg'%M_reinfSteel_foundation)
        #print('Foundation concrete volume: %s m3'%V_conc_foundation)
        print('Nacelle weight: %s kg'%M_nacelle)
        print('Rotor weight: %s'%M_rotor)
        print('Electronics: %s'%M_electronics)
        print('Total weight: %s kg'%M_all)

    
    dict_activities={}
    
    #Adding input element expressed in percentage of mass * M
    phase = 'Input'
    if print_details:
        print(phase)
    for component in set(df_inv[phase].columns.get_level_values(0)):
        if print_details:
            print('\t' + component)

        if component == 'Electronics':
            M = M_electronics

        if component == 'Foundation':
            continue           

        if component == 'Nacelle':
            M = M_nacelle

        if component == 'Power supply':
            M = M_power_supply

        if component == 'Rotor':
            M = M_rotor

        if component == 'Tower':
            M = M_tower 

        if print_details:
            print(M)

        for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset] #previous: df_inv
                if print_details:
                    print('\t \t \t' + dataset + '\t M = %s'%M + '\t pctg = %s' %(np.interp(P, df_inv_i.index.values, df_inv_i.values)) + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                add_to_dict(dict_activities, key = bd.get_activity((df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M)


    #Adding onshore foundation:
    if offshore==False:
        if print_details:
            print('\t Onshore foundation')
            print('\t \t M_fond = %s'%M_foundation)
            print('\t \t concrete = %s' %V_conc_foundation + 'm3')
            print('\t \t reinforced steel = %s' %M_reinfSteel_foundation + 'kg')
        #V_conc_foundation
        add_to_dict(dict_activities, key = bd.get_activity((df_act[(df_act['Component'] == 'Foundation')&(df_act['Market name'] == 'market for concrete, 30MPa')]['UUID'].iloc[0])), value = V_conc_foundation)
        #M_reinfSteel_foundation
        add_to_dict(dict_activities, key = bd.get_activity((df_act[(df_act['Component'] == 'Foundation')&(df_act['Market name'] == 'market for waste reinforced concrete')]['UUID'].iloc[0])), value = M_reinfSteel_foundation)
    
    #Adding road
    phase = 'Assembly'
    if offshore==False:
        value = np.interp(P, [0, 2000],[0, 8000])/park_size
        add_to_dict(dict_activities, key = bd.get_activity((Road.key)), value = value)

    #Adding land use
    if offshore==False:    
        phase = 'Input'
        if print_details:
            print('Land use')
        matching_activities = [act for act in eidb if 'land use for onshore wind turbines' in act['name'] and iso_code in act['location']]
        if len(matching_activities) == 1:
            lus_onshore = matching_activities[0]
            
            # Delete old biosphere exchanges
            old_biosphere_exchanges = [exc for exc in lus_onshore.exchanges() if exc['type'] == 'biosphere']
            for exc in old_biosphere_exchanges:
                exc.delete()

            # Add new biosphere exchanges
            for dataset in df_inv_not_kg['Input']['Foundation']['Land use']:
                # Clean and prepare data
                df_inv_not_kg_i = df_inv_not_kg['Input']['Foundation']['Land use'][dataset].dropna()
                df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce').dropna()
                df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce').dropna()

                # Interpolate values
                interpolated_value = np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)

                if print_details:
                    print(f'\t{dataset}\t {interpolated_value}')
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")

                # Create new exchange for the interpolated value
                act_uuid = df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']
                new_exc = lus_onshore.new_exchange(input=bd.get_activity(act_uuid), amount=interpolated_value, type='biosphere')
                new_exc.save()

            # Add production exchange if not already present
            if not any(exc['type'] == 'production' for exc in lus_onshore.exchanges()):
                lus_onshore.new_exchange(input=lus_onshore.key, amount=1.0, unit='unit', type='production').save()

            # Save the activity with all the new exchanges
            lus_onshore.save()

        # If no matches, or more than 1 match, create a new activity
        elif not matching_activities:
            lus_onshore = eidb.new_activity(name='land use for onshore wind turbines', unit='unit', location=iso_code, amount=1.0, type='process', product='land use for onshore wind turbines', code=random.randint(10000, 100000))
            lus_onshore.save()

            # Repeat adding biosphere exchanges (code same as above)
            
            for dataset in df_inv_not_kg['Input']['Foundation']['Land use']:
                df_inv_not_kg_i = df_inv_not_kg['Input']['Foundation']['Land use'][dataset].dropna()
                df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce').dropna()
                df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce').dropna()

                interpolated_value = np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)

                if print_details:
                    print(f'\t{dataset}\t {interpolated_value}')
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")

                act_uuid = df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']
                new_exc = lus_onshore.new_exchange(input=bd.get_activity(act_uuid), amount=interpolated_value, type='biosphere')
                new_exc.save()

            if not any(exc['type'] == 'production' for exc in lus_onshore.exchanges()):
                lus_onshore.new_exchange(input=lus_onshore.key, amount=1.0, unit='unit', type='production').save()
            lus_onshore.save()

        # Add to dictionary
        add_to_dict(dict_activities, key=lus_onshore, value=1.0/park_size)
        
    #Maintenance
    if offshore==False:
        phase = 'Maintenance'
        if print_details:
            print('Maintenance')
        dataset = 'Car [km]'
        df_inv_not_kg_i = df_inv_not_kg['Maintenance']['Nacelle']['Transport by car'][dataset].dropna()
        df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce')
        df_inv_not_kg_i = df_inv_not_kg_i.dropna()
        df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce')
        df_inv_not_kg_i = df_inv_not_kg_i.dropna()
        if print_details:
            print('\t' + dataset + '\t %s'%(np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)))
            print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
        add_to_dict(dict_activities, key = bd.get_activity((df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values))

    #Assembly activities not in kg
    phase = 'Assembly'
    if print_details:
        print(phase + 'not in kg')
    for component in set(df_inv_not_kg[phase].columns.get_level_values(0)):
        if print_details:
            print('\t' + component)
        for sub_comp in set(df_inv_not_kg[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv_not_kg[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_not_kg_i = df_inv_not_kg[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print(dataset)
                if component == 'Foundation':
                    df_inv_not_kg_i = df_inv_not_kg[phase][component][sub_comp][dataset].dropna()
                    df_inv_not_kg_i = pd.to_numeric(df_inv_not_kg_i, errors='coerce')
                    df_inv_not_kg_i = df_inv_not_kg_i.dropna()
                    df_inv_not_kg_i.index = pd.to_numeric(df_inv_not_kg_i.index, errors='coerce')
                    df_inv_not_kg_i = df_inv_not_kg_i.dropna()
                    if not df_inv_not_kg_i.empty:
                        if print_details:
                            print('\t' + dataset + '\t %s' % (np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)))
                            print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                    else:
                        print(f"No valid data available for interpolation in dataset: {dataset}")
                else:
                    s = InterpolatedUnivariateSpline(df_inv_not_kg_i.index.values, df_inv_not_kg_i.values, k=1)
                    if print_details:
                        print('\t' + dataset + '\t %s'%(s(P)))
                        print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                    add_to_dict(dict_activities, key = bd.get_activity((df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = s(P))

    #Connexion_requirements
    if offshore==True:
        if print_details:
            print('Offshore connection')
        M, E_CLS = cable_requirements_v01(P = P, lon = lon, lat = lat, park_size = park_size, dist_transfo = dist_transfo)#, dist_coast = dist_coast)
        
        phase = 'Input'
        component = 'Power supply'
        for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print('\t \t \t' + dataset + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                add_to_dict(dict_activities, key = bd.get_activity((df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M)
        
        #Energy for cable laying ship - per park
        phase = "Assembly"
        component = "Power supply" 
        add_to_dict(dict_activities, key = bd.get_activity(key = (diesel_burned_activity.database, diesel_burned_activity.code)), value = E_CLS)
        
        #Transfo - per park
        phase="Input"
        component = "Transformer"
        sub_comp = "HV/MV transformer"
        add_to_dict(dict_activities, key = MV_transfo, value = P / 10e3 / 0.85 * lifetime/35) #calculation prorated to power, factor 0.85 in active and apparent power and service life
        add_to_dict(dict_activities, key = HV_transfo, value = P / 500e3 / 0.85 * lifetime/35)


    # For onshore, medium-voltage transformer in proportion to power, and cable section and length according to power.
    if offshore==False:
        phase = "Input"
        component = "Transformer"
        sub_comp = "MV transformer"   
        add_to_dict(dict_activities, key = MV_transfo, value = P / 10e3 / 0.85 * 19/35)
        
        M  = cable_requirements_Onshore_v01(P, lon=lon, lat=lat)
        phase = 'Input'
        component = 'Power supply'
        
        for sub_comp in set(df_inv[phase][component].columns.get_level_values(0)):
            if print_details:
                print('\t \t' + sub_comp)
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print('\t \t \t' + dataset + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                add_to_dict(dict_activities, key = bd.get_activity((df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == phase)].iloc[0]['UUID'])), value = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M)

        
    #Transport activities
    phase = 'Transport'
    trsp_truck_nacelle, trsp_truck_rotor, trsp_truck_tower, trsp_ship_tower,trsp_truck_foundation, trsp_end_of_life, trsp_maintenance_per_year, trsp_ship_offshore = transport_requirements(lon, lat, M_nacelle,  M_tower, M_rotor,  M_foundation,  M_all, lifetime)
    
    #Truck transportation
    trsp_truck = trsp_truck_nacelle + trsp_truck_rotor + trsp_truck_tower + trsp_truck_foundation + trsp_end_of_life + trsp_maintenance_per_year
    add_to_dict(dict_activities, key = bd.get_activity(Truck_transport.key), value = trsp_truck)

    
    #Ship transportation
    if offshore==True:
        trsp_ship = trsp_ship_tower + trsp_ship_offshore
    else:
        trsp_ship = trsp_ship_tower
    
    add_to_dict(dict_activities, key = bd.get_activity(Ship_transport.key), value = trsp_ship)

    #Scour stuff
    phase = 'Assembly'
    if offshore==True:
        scour_poly = scour_volume()
        scour_value = scour_poly(P)
        add_to_dict(dict_activities, key = bd.get_activity(key=(Digger.database, Digger.code)), value = scour_value)

    phase = 'Input'
    if offshore==True:
        if sea_depth <= 30:  # Monopile foundation
            # Calculate monopile foundation requirements
            m_grout, m_monopile = grout_and_monopile_requirements(P, sea_depth)
        
            # Add monopile-related materials to the dictionary
            add_to_dict(dict_activities, key=bd.get_activity(Cement.key), value=m_grout)

            phase = 'Input'
            component = 'Foundation'
            sub_comp = 'Material'
            if sea_depth <= 30:
                M = m_monopile
            elif 30 < sea_depth <= 60:
                M = m_semi_sub
            else:
                M = m_spar_steel + m_spar_iron
            for dataset in set(df_inv[phase][component][sub_comp].columns.get_level_values(0)):
                df_inv_i = df_perc[phase][component][sub_comp][dataset].dropna()
                if print_details:
                    print('\t \t \t' + dataset + '\t %s'%(np.interp(P, df_inv_i.index.values, df_inv_i.values)*M))
                    print(f"\t \t \t{df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)].iloc[0]['UUID']}")
                key_dict = bd.get_activity(df_act[((df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)) & (df_act['Phase'] == 'Input')].iloc[0]['UUID'])
                value_dict = np.interp(P, df_inv_i.index.values, df_inv_i.values) * M
                add_to_dict(dict_activities, key = key_dict, value = value_dict)


        elif 30 < sea_depth <= 60:  # Semi-submersible foundation
            # Calculate semi-submersible foundation requirements
            m_semi_sub = semi_sub_floating_foundation(P, sea_depth)
        
            # Add semi-submersible-related materials to the dictionary
            add_to_dict(dict_activities, key=bd.get_activity(Steel_dataset), value=m_semi_sub)
    
        else:  # Spar buoy foundation (sea depth > 60)
            # Calculate spar buoy foundation requirements
            m_spar_steel, m_spar_iron = spar_buoy_floating_foundation(P, sea_depth)
        
            # Add spar buoy-related materials to the dictionary
            add_to_dict(dict_activities, key=Steel_dataset, value=m_spar_steel)
            add_to_dict(dict_activities, key=bd.get_activity(key = (iron.database, iron.code)), value=m_spar_iron)

    #Assembly depending on kg
    phase = 'Assembly'
    for AA in  assembly_activities:
        if print_details:
            print(AA)
        if AA == 'Explosives':
            value = 10
            add_to_dict(dict_activities, key = bd.get_activity(key = (Explosive.database, Explosive.code)), value = value )     
        if AA == 'Copper wire drawing':
            target_key = bd.get_activity(df_act[df_act['Dataset'] == 'Copper']['UUID'].iloc[0])
            value = dict_activities[target_key]
            add_to_dict(dict_activities, key =bd.get_activity(key = (Copper_wire_drawing.database, Copper_wire_drawing.code)), value=value)
        if AA == 'Steel sheet rolling':
            target_key_1 = bd.get_activity(df_act[df_act['Dataset'] == 'Low-alloy steel']['UUID'].iloc[0])
            value_1 = dict_activities[target_key_1]
            target_key_2 = bd.get_activity(df_act[df_act['Dataset'] == 'Cast iron']['UUID'].iloc[0])
            value_2 = dict_activities[target_key_2]
            add_to_dict(dict_activities, key=bd.get_activity(key = (Steel_sheet_rolling.database, Steel_sheet_rolling.code)), value=value_1+value_2)
        if AA == "Aluminium sheet rolling":
            target_key = bd.get_activity(df_act[df_act['Dataset'] == 'Aluminium 0% recycled']['UUID'].iloc[0])
            value = dict_activities[target_key]
            add_to_dict(dict_activities, key=bd.get_activity(key = (Aluminium_sheet_rolling.database, Aluminium_sheet_rolling.code)), value=value)
        if AA == 'Chromium steel sheet rolling': #before: Chromium steel sheet rolling
            target_key = bd.get_activity(df_act[df_act['Dataset'] == 'Chromium steel']['UUID'].iloc[0])
            #value = find_values_sum(dict_activities, target_key)
            value = dict_activities[target_key]
            add_to_dict(dict_activities, key=bd.get_activity(key = (Chromium_sheet_rolling.database, Chromium_sheet_rolling.code)), value=value)

    #Disposal activities
    phase= 'Disposal'
    for DA in  disposal_activities:
        if print_details:
            print(DA)
        ei_key_disposal = df_act[ (df_act['Dataset'] == DA) & (df_act['Phase'] == 'Disposal')]['UUID'].iloc[0]
        DA = DA.replace(' -waste','').replace('Steel, inert waste', 'Low-alloy steel' ).replace('Concrete, inert waste', 'Concrete [m3]').replace('Aluminium waste', 'Aluminium 0% recycled').replace('Chromium Steel waste','Chromium steel')
        ei_key_input = df_act[ (df_act['Dataset'] == DA) & (df_act['Phase'] == 'Input')]['UUID'].iloc[0]
        try:
            target_key = bd.get_activity(ei_key_input) 
            value = - dict_activities[target_key]
            add_to_dict(dict_activities, key = bd.get_activity(ei_key_disposal), value = value)
        except:
            pass

    #Electricity dataset
    phase= 'Assembly'
    value = 0.5 * (M_nacelle + M_rotor + M_tower)
    add_to_dict(dict_activities, key = bd.get_activity(Electricity_dataset.key), value = value)

    return dict_activities

end_time = time.time()
execution_time = end_time - start_time
logger.debug("Execution time:", execution_time, "seconds")