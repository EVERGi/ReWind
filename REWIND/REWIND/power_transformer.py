import time
from pathlib import Path
from logging import getLogger
from prepare_inventories import ecoinvent_setup, get_activities_from_names

import pandas as pd
import bw2data as bd
import bw2io as bi
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

logger = getLogger(__name__)
start_time = time.time()

# Constants
_TURBINE_RATED_POWERS = [30, 150, 600, 800, 2000]
_DATA_DIR = Path(__file__).resolve().parent / "data"

def transfo_500mva():
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
    
    # Check if the transformer dataset already exists
    #existing_transfo = [act for act in eidb if 'Power transformer TrafoStar 500 MVA' in act['name']]
    existing_transfo = [bd.get_activity(code=(get_activities_from_names(eidb, ['Power transformer TrafoStar 500 MVA'])[0]).code)]
    
    if not existing_transfo:  # If the dataset does not exist, create it
        # Find the base activity
        act_transfo = [act for act in eidb if act["name"] == "transformer production, high voltage use"][0]

        # Copy and configure the new dataset
        HV_transfo = act_transfo.copy()
        HV_transfo["name"] = "Power transformer TrafoStar 500 MVA"
        HV_transfo["unit"] = "unit"
        HV_transfo.save()

        # Remove existing exchanges
        for exc in HV_transfo.exchanges():
            exc.delete()
        HV_transfo.save()

        # Define materials and add exchanges
        materials = [
            ("steel production, electric, low-alloyed", "Europe without Switzerland and Austria", 99640, "kilogram"),
            ("market for lubricating oil", "RER", 63000, "kilogram"),
            ("market for copper, cathode", "GLO", 39960, "kilogram"),
            ("market for glass wool mat", "GLO", 6500, "kilogram"),
            ("planing, board, softwood, u=20%", "CH", 15000, "kilogram"),
            ("market for ceramic tile", "GLO", 2650, "kilogram"),
            ("market for steel, unalloyed", "GLO", 53618, "kilogram"),
            ("market for electrostatic paint", "GLO", 2200, "kilogram"),
            ("market for electricity, medium voltage", "SE", 750000, "kilowatt hour"),
            ("heat, from municipal waste incineration to generic market for heat district or industrial, other than natural gas", "SE", 1080000, "megajoule")
        ]

        for name, location, amount, unit in materials:
            material = [act for act in eidb if name in act["name"] and location in act["location"]][0]
            new_exc = HV_transfo.new_exchange(input=material.key, amount=amount, unit=unit, type='technosphere')
            new_exc.save()

        # Add the production exchange
        new_exc = HV_transfo.new_exchange(input=HV_transfo.key, amount=1, unit="unit", categories="", type='production')
        new_exc.save()
        HV_transfo.save()

        print("New HV transformer dataset created.")
    
    else:  # If the dataset already exists, assign it to HV_transfo
        HV_transfo = existing_transfo[0]
        print("Existing HV transformer dataset found.")

    print("Your dataset was successfully built or retrieved!")
    return HV_transfo

    
def transfo_10mva():
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
    
    # Check if the transformer dataset exists
    #existing_transfo = [act for act in eidb if 'Power transformer TrafoStar 10 MVA' in act['name']]
    existing_transfo = [bd.get_activity(code=(get_activities_from_names(eidb, ['Power transformer TrafoStar 10 MVA'])[0]).code)]
    
    if not existing_transfo:  # If the dataset does not exist, create it
        # Find the dataset for the 500 MVA transformer as a base
        #act = [a for a in eidb if "Power transformer TrafoStar 500 MVA" in a["name"]][0]
        act = transfo_500mva()
        MV_transfo = act.copy()
        MV_transfo["name"] = "Power transformer TrafoStar 10 MVA"
        MV_transfo.save()

        # Modify exchanges
        for exc in MV_transfo.exchanges():
            if exc.input['name'] == "steel production, electric, low-alloyed":
                exc["amount"] = 6820
                exc.save()
            elif exc.input['name'] == "market for lubricating oil":
                exc["amount"] = 6780
                exc.save()
            elif exc.input['name'] == "market for copper, cathode":
                exc["amount"] = 3526
                exc.save()
            elif exc.input['name'] == "market for ceramic tile":
                exc["amount"] = 53
                exc.save()
            elif exc.input['name'] == "market for steel, unalloyed":
                exc["amount"] = 9066
                exc.save()
            elif exc.input['name'] == "market for electrostatic paint":
                exc["amount"] = 95
                exc.save()
            elif exc.input['name'] == "market for electricity, medium voltage":
                exc["amount"] = 105200
                exc.save()
            elif exc.input['name'] == "heat, from municipal waste incineration to generic market for heat district or industrial, other than natural gas":
                exc["amount"] = 68760
                exc.save()
            elif exc.input['name'] == "market for aluminium, cast alloy":
                exc["amount"] = 65
                exc.save()
            elif exc.input['name'] in [
                "market for sheet rolling, steel",
                "market for epoxy resin, liquid",
                "market for glass fibre",
                "market for kraft paper, bleached",
                "market for paper, melamine impregnated",
                "market for electrostatic paint",
                "market for glass fibre",
            ]:
                exc.delete()
            elif exc.input['name'] == "Power transformer TrafoStar 250 MVA":
                exc.input = MV_transfo
                exc.save()

        # Add new exchanges for insulation and wood
        insulation = [act for act in eidb if "market for glass wool mat" in act["name"] and "GLO" in act["location"]][0]
        new_exc = MV_transfo.new_exchange(input=insulation.key, amount=337, unit="kilogram", type='technosphere')
        new_exc.save()
        
        wood = [act for act in eidb if "planing, board, softwood, u=20%" in act["name"] and "CH" in act["location"]][0]
        new_exc = MV_transfo.new_exchange(input=wood.key, amount=366, unit="kilogram", type='technosphere')
        new_exc.save()

        MV_transfo.save()
        print("New MV transformer dataset created.")
    else:  # If the dataset already exists, assign it to MV_transfo
        MV_transfo = existing_transfo[0]
        print("Existing MV transformer dataset found.")

    return MV_transfo
