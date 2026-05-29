import pandas as pd
import bw2data as bd
import bw2calc as bc
from pathlib import Path
import matplotlib.pyplot as plt
from prepare_inventories import ecoinvent_setup
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import defaultdict

_DATA_DIR = Path(__file__).resolve().parent / "data"

def flatten_nested_dict(nested_dict):
    """
    Recursively flattens a nested dictionary structure.
    """
    flat_dict = {}

    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # Recursively flatten if the value is a dictionary
            deeper_dict = flatten_nested_dict(value)
            flat_dict.update(deeper_dict)
        else:
            # Otherwise, it's a key-value pair
            flat_dict[key] = value

    return flat_dict

def lca_wimby_map(dict_activities, impact_category, aep, lifetime_wt=20):#, impact_category=('IPCC 2021', 'climate change', 'global warming potential (GWP100)')):
    """
    Perform a Life Cycle Assessment (LCA) using the Brightway2.5 package.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA. Can include nested keys for 'Input'.
    - aep (float): Annual energy production in kWh.
    - lifetime_wt (int): Lifetime of the wind turbine in years. Defaults to 20.
    - impact_category (tuple): Impact assessment method to use. Defaults to ('IPCC 2021', 'climate change', 'global warming potential (GWP100)').

    Returns:
    - DataFrame: A DataFrame with method, unit, life cycle stages, their corresponding impact scores, and a 'Total' column.
    """
    lt_electricity = aep * lifetime_wt
    #impact_category = [m for m in bd.methods if 'EF' in str(m) and  'climate change' in str(m) and 'GWP100' in str(m) and 'no LT' not in str(m) and 'v3.1' in str(m)][0]

    # Initialize an empty dictionary to store results
    results = {}

    # Loop over the life cycle stages in the dictionary
    for stage, materials in dict_activities.items():
        if stage == "Input":
            # Use defaultdict to sum duplicate material quantities
            flattened_materials = defaultdict(float)
            for component, sub_dict in materials.items():
                for sub_comp, activities in sub_dict.items():
                    for material, quantity in activities.items():
                        flattened_materials[material] += quantity  # Sum the quantities

            # Convert defaultdict back to a regular dictionary
            flattened_materials = dict(flattened_materials)


            # Perform LCA for the flattened 'Input' materials
            lca = bc.LCA(flattened_materials, impact_category)#, method=impact_category)
            #lca.redo_lci(demand=)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score / lt_electricity
        else:
            # Perform LCA for non-nested phases
            lca = bc.LCA(materials, impact_category)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score / lt_electricity

    # Extract the method and unit
    method_name = " - ".join(impact_category)  # Join the tuple to create a string for the method
    unit = bd.Method(impact_category).metadata['unit']  # Get the unit from the method

    # Calculate the total impacts across all stages
    total_impact = sum(results.values())

    # Add the 'Total' key with the sum of all stages
    results['Total'] = total_impact

    # Create the DataFrame
    results_df = pd.DataFrame({
        'Method': [method_name],
        'Unit': [unit],
        **results  # Unpack the results dictionary to add stages as columns, including 'Total'
    })

    return results_df

def lca_wimby_map_total(dict_activities, impact_category):#, impact_category=('IPCC 2021', 'climate change', 'global warming potential (GWP100)')):
    """
    Perform a Life Cycle Assessment (LCA) using the Brightway2.5 package.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA. Can include nested keys for 'Input'.
    - impact_category (tuple): Impact assessment method to use. Defaults to ('IPCC 2021', 'climate change', 'global warming potential (GWP100)').

    Returns:
    - DataFrame: A DataFrame with method, unit, life cycle stages, their corresponding impact scores, and a 'Total' column. Results are not normalized per electricity generation
    """
    #impact_category = [m for m in bd.methods if 'EF' in str(m) and  'climate change' in str(m) and 'GWP100' in str(m) and 'no LT' not in str(m) and 'v3.1' in str(m)][0]

    # Initialize an empty dictionary to store results
    results = {}

    # Loop over the life cycle stages in the dictionary
    for stage, materials in dict_activities.items():
        if stage == "Input":
            # Use defaultdict to sum duplicate material quantities
            flattened_materials = defaultdict(float)
            for component, sub_dict in materials.items():
                for sub_comp, activities in sub_dict.items():
                    for material, quantity in activities.items():
                        flattened_materials[material] += quantity  # Sum the quantities

            # Convert defaultdict back to a regular dictionary
            flattened_materials = dict(flattened_materials)


            # Perform LCA for the flattened 'Input' materials
            lca = bc.LCA(flattened_materials, impact_category)#, method=impact_category)
            #lca.redo_lci(demand=)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score
        else:
            # Perform LCA for non-nested phases
            lca = bc.LCA(materials, impact_category)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score

    # Extract the method and unit
    method_name = " - ".join(impact_category)  # Join the tuple to create a string for the method
    unit = bd.Method(impact_category).metadata['unit']  # Get the unit from the method

    # Calculate the total impacts across all stages
    total_impact = sum(results.values())

    # Add the 'Total' key with the sum of all stages
    results['Total'] = total_impact

    # Create the DataFrame
    results_df = pd.DataFrame({
        'Method': [method_name],
        'Unit': [unit],
        **results  # Unpack the results dictionary to add stages as columns, including 'Total'
    })

    return results_df


def lca_detailed_materials(dict_activities, impact_category, aep, lifetime_wt=20):
    """
    Perform a detailed Life Cycle Assessment (LCA) for each material in a nested dictionary.

    Parameters:
    - dict_activities (dict): Nested dictionary containing the functional unit for the LCA.
    - impact_category (tuple): Impact assessment method to use.
    - aep (float): Annual energy production in kWh.
    - lifetime_wt (int): Lifetime of the wind turbine in years. Defaults to 20.

    Returns:
    - DataFrame: A DataFrame with life cycle stage, component, sub-component, material, quantity, and environmental impact.
    """
    lt_electricity = aep * lifetime_wt  # Total electricity produced over the lifetime
    results = []  # List to store results for each material

    for stage, materials in dict_activities.items():
        if stage == "Input":
            for component, sub_dict in materials.items():
                for sub_component, activities in sub_dict.items():
                    for material, quantity in activities.items():
                        # Perform LCA for each material
                        lca = bc.LCA({material: quantity}, impact_category)
                        lca.lci()
                        lca.lcia()
                        impact = lca.score / lt_electricity  # Normalize by total electricity

                        # Append results to the list
                        results.append({
                            "Life Cycle Stage": stage,
                            "Component": component,
                            "Sub-Component": sub_component,
                            "Material": material,
                            "Quantity": quantity,
                            "Environmental Impact": impact
                        })
        else:
            for material, quantity in materials.items():
                # Perform LCA for non-nested materials
                lca = bc.LCA({material: quantity}, impact_category)
                lca.lci()
                lca.lcia()
                impact = lca.score / lt_electricity  # Normalize by total electricity

                # Append results to the list
                results.append({
                    "Life Cycle Stage": stage,
                    "Component": None,
                    "Sub-Component": None,
                    "Material": material,
                    "Quantity": quantity,
                    "Environmental Impact": impact
                })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def lca_wimby_fleet_evaluation(dict_activities, impact_category, aep, lifetime_wt=20):#, impact_category=('IPCC 2021', 'climate change', 'global warming potential (GWP100)')):
    """
    Perform a Life Cycle Assessment (LCA) using the Brightway2.5 package.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA. Can include nested keys for 'Input'.
    - aep (float): Annual energy production in kWh.
    - lifetime_wt (int): Lifetime of the wind turbine in years. Defaults to 20.
    - impact_category (tuple): Impact assessment method to use. Defaults to ('IPCC 2021', 'climate change', 'global warming potential (GWP100)').

    Returns:
    - DataFrame: A DataFrame with method, unit, life cycle stages, their corresponding impact scores, and a 'Total' column.
    """
    #lt_electricity = aep * lifetime_wt
    #impact_category = [m for m in bd.methods if 'EF' in str(m) and  'climate change' in str(m) and 'GWP100' in str(m) and 'no LT' not in str(m) and 'v3.1' in str(m)][0]

    # Initialize an empty dictionary to store results
    results = {}

    # Loop over the life cycle stages in the dictionary
    for stage, materials in dict_activities.items():
        if stage == "Input":
            # Use defaultdict to sum duplicate material quantities
            flattened_materials = defaultdict(float)
            for component, sub_dict in materials.items():
                for sub_comp, activities in sub_dict.items():
                    for material, quantity in activities.items():
                        flattened_materials[material] += quantity  # Sum the quantities

            # Convert defaultdict back to a regular dictionary
            flattened_materials = dict(flattened_materials)


            # Perform LCA for the flattened 'Input' materials
            lca = bc.LCA(flattened_materials, impact_category)#, method=impact_category)
            #lca.redo_lci(demand=)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score / aep#lt_electricity
        else:
            # Perform LCA for non-nested phases
            lca = bc.LCA(materials, impact_category)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score / aep#lt_electricity

    # Extract the method and unit
    method_name = " - ".join(impact_category)  # Join the tuple to create a string for the method
    unit = bd.Method(impact_category).metadata['unit']  # Get the unit from the method

    # Calculate the total impacts across all stages
    total_impact = sum(results.values())

    # Add the 'Total' key with the sum of all stages
    results['Total'] = total_impact

    # Create the DataFrame
    results_df = pd.DataFrame({
        'Method': [method_name],
        'Unit': [unit],
        **results  # Unpack the results dictionary to add stages as columns, including 'Total'
    })

    return results_df

def lca_wimby_fleet_evaluation_total_impacts(dict_activities, impact_category, lifetime_wt=20):#, impact_category=('IPCC 2021', 'climate change', 'global warming potential (GWP100)')):
    """
    Perform a Life Cycle Assessment (LCA) using the Brightway2.5 package.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA. Can include nested keys for 'Input'.
    - aep (float): Annual energy production in kWh.
    - lifetime_wt (int): Lifetime of the wind turbine in years. Defaults to 20.
    - impact_category (tuple): Impact assessment method to use. Defaults to ('IPCC 2021', 'climate change', 'global warming potential (GWP100)').

    Returns:
    - DataFrame: A DataFrame with method, unit, life cycle stages, their corresponding impact scores, and a 'Total' column.
    """
    #lt_electricity = aep * lifetime_wt
    #impact_category = [m for m in bd.methods if 'EF' in str(m) and  'climate change' in str(m) and 'GWP100' in str(m) and 'no LT' not in str(m) and 'v3.1' in str(m)][0]

    # Initialize an empty dictionary to store results
    results = {}

    # Loop over the life cycle stages in the dictionary
    for stage, materials in dict_activities.items():
        if stage == "Input":
            # Use defaultdict to sum duplicate material quantities
            flattened_materials = defaultdict(float)
            for component, sub_dict in materials.items():
                for sub_comp, activities in sub_dict.items():
                    for material, quantity in activities.items():
                        flattened_materials[material] += quantity  # Sum the quantities

            # Convert defaultdict back to a regular dictionary
            flattened_materials = dict(flattened_materials)


            # Perform LCA for the flattened 'Input' materials
            lca = bc.LCA(flattened_materials, impact_category)#, method=impact_category)
            #lca.redo_lci(demand=)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score# / aep#lt_electricity
        else:
            # Perform LCA for non-nested phases
            lca = bc.LCA(materials, impact_category)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score #/ aep#lt_electricity

    # Extract the method and unit
    method_name = " - ".join(impact_category)  # Join the tuple to create a string for the method
    unit = bd.Method(impact_category).metadata['unit']  # Get the unit from the method

    # Calculate the total impacts across all stages
    total_impact = sum(results.values())

    # Add the 'Total' key with the sum of all stages
    results['Total'] = total_impact

    # Create the DataFrame
    results_df = pd.DataFrame({
        'Method': [method_name],
        'Unit': [unit],
        **results  # Unpack the results dictionary to add stages as columns, including 'Total'
    })

    return results_df


def wp4_evaluation(dict_activities, impact_category):
    """
    Perform LCA using Brightway2.5 and return absolute scores (not per kWh).

    Parameters:
    - dict_activities (dict): Dictionary containing functional unit per life cycle stage.
    - impact_category (tuple): Impact assessment method.

    Returns:
    - dict: Dictionary of LCA scores per stage (absolute values).
    """
    results = {}

    for stage, materials in dict_activities.items():
        if stage == "Input":
            flattened_materials = defaultdict(float)
            for component, sub_dict in materials.items():
                for sub_comp, activities in sub_dict.items():
                    for material, quantity in activities.items():
                        flattened_materials[material] += quantity
            flattened_materials = dict(flattened_materials)

            lca = bc.LCA(flattened_materials, impact_category)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score
        else:
            lca = bc.LCA(materials, impact_category)
            lca.lci()
            lca.lcia()
            results[stage] = lca.score

    total_impact = sum(results.values())
    results['Total'] = total_impact

    return results


def sum_materials(data, material_sums=None):
    if material_sums is None:
        material_sums = {}

    if isinstance(data, dict):  # Ensure data is a dictionary before iterating
        for material, quantity in data.items():
            if isinstance(quantity, dict):
                sum_materials(quantity, material_sums)
            else:
                if material in material_sums:
                    material_sums[material] += quantity
                else:
                    material_sums[material] = quantity
    elif isinstance(data, (int, float, np.float64)):  # Handle case where data is a number
        return  # Avoid recursion on numbers

    return material_sums

#Dependent sampling
def monte_carlo_lca(
    activity_dict: dict, 
    aep: float, 
    impact_category: tuple, 
    iterations: int = 20, 
    lifetime: int = 20
) -> pd.DataFrame:
    """
    Perform a Monte Carlo Life Cycle Assessment (LCA) for a given dictionary of activities,
    where the inventory and characterization matrices are sampled once, and the demand vector
    is resampled in each iteration.

    Parameters:
    - activity_dict (dict): Dictionary of ecoinvent activities with their quantities.
    - aep (float): Annual electricity production in kWh.
    - impact_category (tuple): Impact assessment method.
    - iterations (int): Number of Monte Carlo iterations to perform. Default is 20.
    - lifetime (int): Lifetime of the system in years. Default is 20.

    Returns:
    - pd.DataFrame: DataFrame containing Monte Carlo results for each iteration.
    """
    # Calculate lifetime electricity production
    lifetime_electricity = aep * lifetime

    # Initialize LCA object and load method
    lca = bc.LCA(activity_dict, method=impact_category, use_distributions=True)

    # Perform the first LCA to sample the inventory and characterization matrices
    lca.lci()  # Lifecycle Inventory calculation (samples the inventory matrix)
    lca.lcia()  # Lifecycle Impact Assessment calculation (samples the characterization matrix)

    # Get the unit of the impact category
    method_unit = bd.Method(impact_category).metadata['unit']

    # Store the sampled inventory and characterization matrices
    inventory_matrix = lca.inventory  # This is the technosphere matrix
    characterization_matrix = lca.characterization_mm  # Corrected to use characterization_mm

    # Container for Monte Carlo results
    results = []

    # Perform Monte Carlo iterations
    for iteration in range(iterations):
        # Create a new LCA object each iteration to ensure we only resample the demand vector
        lca = bc.LCA(activity_dict, method=impact_category, use_distributions=True)

        # Set the previously sampled inventory and characterization matrices
        lca.inventory = inventory_matrix
        lca.characterization_mm = characterization_matrix  # Set the correct characterization matrix

        # Ensure that the LCA object goes through the calculations before sampling demand vector
        lca.lci()  # This should properly set up the technosphere matrix and demand array
        lca.lcia()  # Ensure LCIA is computed to set up the characterization

        # Resample the demand vector (the quantities in activity_dict) for this iteration
        next(lca)  # Resample the demand vector while keeping the matrices fixed

        # Calculate normalized impact score per kWh of electricity
        impact_per_kwh = lca.score / lifetime_electricity
        results.append({
            'Iteration': iteration + 1,
            'Impact Score': impact_per_kwh,
            'Method': " - ".join(impact_category),
            'Unit': method_unit
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    #return results_df
    return results_df["Impact Score"].iloc[0] 

def plot_monte_carlo(results, countries, impact_category_label, turbine_type, save_path=None):
    """
    Plots a box-and-whisker chart for turbine LCA results (either onshore or offshore).
    
    Parameters:
    - results (list): List of DataFrames for turbines.
    - countries (list): List of country codes corresponding to the results (e.g., ['FR', 'IT', 'NO', 'DE', 'BE']).
    - impact_category_label (str): Label for the impact category (e.g., "GWP100 (kgCO2eq/kWh generated)").
    - turbine_type (str): Type of turbine, either "Onshore" or "Offshore".
    """
    # Validate inputs
    if len(results) != len(countries):
        raise ValueError("Number of result DataFrames must match the number of countries.")
    
    # Prepare data for plotting
    data = [df["Impact Score"] for df in results]  # Assuming 'Impact Score' is the relevant column

    # Create a figure
    plt.figure(figsize=(10, 6))
    boxplot = plt.boxplot(data, showfliers=False, patch_artist=True)

    # Set box colors to white
    for box in boxplot['boxes']:
        box.set(facecolor='white')

    # Customize x-axis
    plt.xticks(range(1, len(countries) + 1), countries, rotation=45)
    plt.xlabel("Country")

    # Add labels and titles
    plt.title(f'{turbine_type} Turbines: Box Plot of {impact_category_label}')
    plt.ylabel(impact_category_label)

    # Adjust layout and show plot
    plt.tight_layout()

        # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved at {save_path}")

    plt.show()

def lca_not_per_functional_unit(dict_activities, impact_category=('IPCC 2021', 'climate change', 'global warming potential (GWP100)')):
    """
    Perform a Life Cycle Assessment (LCA) using the Brightway2.5 package.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA. Can include nested keys for 'Input'.
    - aep (float): Annual energy production in kWh.
    - lifetime_wt (int): Lifetime of the wind turbine in years. Defaults to 20.
    - impact_category (tuple): Impact assessment method to use. Defaults to ('IPCC 2021', 'climate change', 'global warming potential (GWP100)').

    Returns:
    - DataFrame: A DataFrame with method, unit, life cycle stages, their corresponding impact scores, and a 'Total' column.
    """

    # Initialize an empty dictionary to store results
    results_not_normalized = {}

    # Loop over the life cycle stages in the dictionary
    for stage, materials in dict_activities.items():
        if stage == "Input":
            # Use defaultdict to sum duplicate material quantities
            flattened_materials = defaultdict(float)
            for component, sub_dict in materials.items():
                for sub_comp, activities in sub_dict.items():
                    for material, quantity in activities.items():
                        flattened_materials[material] += quantity  # Sum the quantities

            # Convert defaultdict back to a regular dictionary
            flattened_materials = dict(flattened_materials)

            # Perform LCA for the flattened 'Input' materials
            lca = bc.LCA(demand=flattened_materials, method=impact_category)
            lca.lci()
            lca.lcia()
            results_not_normalized[stage] = lca.score
        else:
            # Perform LCA for non-nested phases
            lca = bc.LCA(demand=materials, method=impact_category)
            lca.lci()
            lca.lcia()
            results_not_normalized[stage] = lca.score

    # Extract the method and unit
    method_name = " - ".join(impact_category)  # Join the tuple to create a string for the method
    unit = bd.Method(impact_category).metadata['unit']  # Get the unit from the method

    # Calculate the total impacts across all stages
    total_impact = sum(results_not_normalized.values())

    # Add the 'Total' key with the sum of all stages
    results_not_normalized['Total'] = total_impact

    # Create the DataFrame
    results_total = pd.DataFrame({
        'Method': [method_name],
        'Unit': [unit],
        **results_not_normalized  # Unpack the results dictionary to add stages as columns, including 'Total'
    })

    return results_total

def lca_simple_score(dict_activities, aep, lifetime_wt=20, impact_category=('IPCC 2021', 'climate change', 'global warming potential (GWP100)')):
    """
    Perform a simple Life Cycle Assessment (LCA) using the Brightway2.5 package.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA. Keys are activities and values are quantities.
    - aep (float): Annual energy production in kWh.
    - lifetime_wt (int): Lifetime of the wind turbine in years. Defaults to 20.
    - impact_category (tuple): Impact assessment method to use. Defaults to ('IPCC 2021', 'climate change', 'global warming potential (GWP100)').

    Returns:
    - float: The normalized LCA impact score (impact per kWh).
    """
    # Calculate lifetime electricity production
    lt_electricity = aep * lifetime_wt

    # Ensure the impact category exists
    if impact_category not in bd.methods:
        raise ValueError(f"Impact category {impact_category} not found in Brightway methods.")

    # Perform the LCA
    lca = bc.LCA(demand=dict_activities, method=impact_category)
    lca.lci()
    lca.lcia()

    # Normalize the score by lifetime electricity production
    normalized_score = lca.score / lt_electricity

    return normalized_score

def lca_input_stage_details(dict_activities, aep, lifetime_wt=20, impact_category=('IPCC 2021', 'climate change', 'global warming potential (GWP100)')):
    """
    Perform a detailed LCA for the 'Input' stage and return impacts per component and subcomponent.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA, including nested keys for 'Input'.
    - aep (float): Annual electricity production.
    - lifetime_wt (int): Wind turbine lifetime in years. Default is 20.
    - impact_category (tuple): Impact assessment method to use. Defaults to ('IPCC 2021', 'climate change', 'global warming potential (GWP100)').

    Returns:
    - DataFrame: A DataFrame with components, subcomponents, impacts, shares, and a total row.
    """
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
    lt_electricity = aep * lifetime_wt

    # Ensure the "Input" stage exists in the dictionary
    if "Input" not in dict_activities:
        raise ValueError("The dictionary must contain an 'Input' stage with components and subcomponents.")

    # List to store detailed results for 'Input' stage
    input_details = []

    # Process the nested structure for the 'Input' stage
    for component, sub_dict in dict_activities["Input"].items():
        for sub_comp, activities in sub_dict.items():
            # Perform LCA for each subcomponent
            lca = bc.LCA(demand=activities, method=impact_category)
            lca.lci()
            lca.lcia()
            impact_score = lca.score / lt_electricity  # Normalize by total electricity production over the lifetime

            # Append results to the list
            input_details.append({
                'Component': component,
                'Subcomponent': sub_comp,
                'Impact': impact_score
            })

    # Convert the list to a DataFrame
    input_df = pd.DataFrame(input_details)

    # Calculate the total impact for the 'Input' stage
    total_impact = input_df['Impact'].sum()

    # Add a 'Share' column for the percentage contribution of each subcomponent
    input_df['Share (%)'] = ((input_df['Impact'] / total_impact) * 100).round(1)

    # Append a row for the total impact
    total_row = {
        'Component': 'Total',
        'Subcomponent': '',
        'Impact': total_impact,
        'Share (%)': 100.0
    }
    input_df = pd.concat([input_df, pd.DataFrame([total_row])], ignore_index=True)

    return input_df

def lca_multi_impact(dict_activities, aep, lifetime_wt=20):
    """
    Perform a Life Cycle Assessment (LCA) for EF v3.1 methods using the Brightway2.5 package.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA, organized by life cycle stages.
                               Includes nested keys for the 'Input' stage.
    - aep (float): Annual energy production of the wind turbine in kWh.
    - lifetime_wt (int): Lifetime of the wind turbine in years. Default is 20.

    Returns:
    - DataFrame: A DataFrame with method, unit, life cycle stages, their corresponding impact scores for each method,
                 and a 'Total' column.
    """
    mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
    # Define the EF v3.1 methods, excluding those with 'EN15804' and 'no LT'
    ef_methods = [m for m in bd.methods if 'EF v3.1' in str(m) and 'EN15804' not in str(m) and 'no LT' not in str(m)]

    # Initialize an empty list to store DataFrames
    results_list = []

    # Compute the total electricity produced over the turbine's lifetime
    lt_electricity = aep * lifetime_wt

    # Loop over each EF v3.1 method in the filtered list
    for method in ef_methods:
        # Initialize a dictionary to store results for the current method
        results = {}

        # Loop over the life cycle stages in the dictionary
        for stage, materials in dict_activities.items():
            if stage == "Input":
                # Flatten the nested structure into a single dictionary of materials
                flattened_materials = {}
                for component, sub_dict in materials.items():
                    for sub_comp, activities in sub_dict.items():
                        flattened_materials.update(activities)

                # Perform LCA for the flattened 'Input' materials
                lca = bc.LCA(demand=flattened_materials, method=method)
                lca.lci()
                lca.lcia()
                results[stage] = lca.score / lt_electricity
            else:
                # Perform LCA for non-nested phases
                lca = bc.LCA(demand=materials, method=method)
                lca.lci()
                lca.lcia()
                results[stage] = lca.score / lt_electricity

        # Calculate the total impacts across all stages
        total_impact = sum(results.values())

        # Add the 'Total' key with the sum of all stages
        results['Total'] = total_impact

        # Extract the method name and unit
        method_name = " - ".join(method)
        unit = bd.Method(method).metadata['unit']

        # Create a DataFrame for the current method and append to results_list
        method_df = pd.DataFrame({
            'Method': [method_name],
            'Unit': [unit],
            **results
        })

        results_list.append(method_df)

    # Concatenate all DataFrames into a single DataFrame
    final_results_df = pd.concat(results_list, ignore_index=True)

    return final_results_df

#KEEP
def stacked_bar(overview_df, total_df, x_tick_label="FR_onshore", figsize=(8, 6)):
    """
    Plots a stacked bar chart for the life cycle stage overview with an additional line for the total.

    Parameters:
        overview_df (pd.DataFrame): DataFrame with numeric impact data per life cycle stage.
            Rows should represent the stages and columns should be numeric.
        total_df (pd.DataFrame): DataFrame with a "Total" column (shape: 8x1) to be plotted as a line.
        x_tick_label (str): Label for the x-axis tick.
        figsize (tuple): Size of the plot (width, height).
    """

    stages = ['Component production', 'Assembly', 'Maintenance', 'Transport', 'Disposal']
    custom_colors = ['#008B8B', '#FFAB40', '#DEDEDE', '#C0DFDF', '#595959']

    # Ensure numeric data and rename "Input" to "Component production"
    overview_df = overview_df.rename(columns={"Input": "Component production"})
    numeric_data = overview_df.select_dtypes(include="number").drop(columns="Total", errors="ignore")

    # Convert to g CO2eq/kWh
    numeric_data *= 1000

    # Create the plot
    ax = numeric_data.plot(
        kind='bar',
        stacked=True,
        figsize=figsize,
        legend=True,
        color=custom_colors
    )

    # Add the line for "Total"
    if "Total" in total_df.columns:
        ax.plot(
            numeric_data.index,
            total_df["Total"] * 1000,  # Also convert "Total" to g CO2eq/kWh
            color="black",
            marker="o",
            label="WIND_LCA_DK",
            linewidth=2
        )

    # Customize the plot
    ax.set_title("Life Cycle Stages Overview")
    ax.set_ylabel("g $CO_2$/kWh")
    ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.5, 0.98))
    ax.set_xticks([0])
    ax.set_xticklabels([x_tick_label], rotation=0)
    plt.tight_layout()
    plt.show()


def chart_input(detailed_df, x_tick_label="FR_onshore", figsize=(8, 6)):
    """
    Plots a stacked bar chart for life cycle stages using specified colors.

    Parameters:
        detailed_df (pd.DataFrame): DataFrame containing detailed data with 'Component' and 'Impact' columns.
        x_tick_label (str): Custom label for the x-axis tick (default: 'FR_onshore').
        figsize (tuple): Size of the plot (width, height).
    """
    # Preprocess the data
    test = detailed_df[:-1]  # Remove the last row
    test = test.groupby('Component')['Impact'].sum().reset_index()
    test.set_index('Component', inplace=True)
    test = test.T
    test = test * 1000  # Convert to g CO2eq/kWh

    # Define custom colors
    colors = ['#004B53', '#EF8600', '#586D77', '#0D5BDC', '#00717D', 
              '#93EDE3', '#EEFF41', '#212121', '#FFEED9', '#D9E7FD']
    
    # Plot the data
    ax = test.plot(
        kind='bar',
        stacked=True,
        legend=True,
        figsize=figsize,
        color=colors[:len(test.columns)]  # Apply custom colors
    )

    # Add labels and customize legend
    plt.title("Input stage breakdown")
    plt.ylabel("g $CO_2$/kWh")
    plt.xticks([0], [x_tick_label], rotation=0)
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.39, 0.98))
    plt.tight_layout()
    plt.show()

#KEEP
def contribution_plot(results_df, title="Contribution analysis"):
    """
    Plot a horizontal 100% stacked bar chart for LCA results.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing LCA results with stages as columns and impact categories as rows.
    - stages (list): List of life cycle stages to include in the chart.
    - custom_colors (list): List of custom colors corresponding to each life cycle stage.

    Returns:
    - None: Displays the plot.
    """

    stages = ['Component production', 'Assembly', 'Maintenance', 'Transport', 'Disposal']
    custom_colors = ['#008B8B', '#FFAB40', '#DEDEDE', '#C0DFDF', '#595959']

    # Rename "Input" to "Component production"
    results_df = results_df.rename(columns={"Input": "Component production"})
    
    # Exclude the "Total" column if it exists
    if "Total" in results_df.columns:
        results_df = results_df.drop(columns=["Total"])

    # Filter for the specified stages
    stage_columns = results_df.columns.intersection(stages)
    
    # Normalize data for a 100% stacked bar chart
    data = results_df[stage_columns]
    normalized_data = data.div(data.sum(axis=1), axis=0).fillna(0)  # Handle rows with no data

    # Plotting the horizontal 100% stacked bar chart
    ax = normalized_data.plot(
        kind='barh',
        stacked=True,
        color=custom_colors,
        figsize=(10, 8),
        edgecolor='black'
    )
    
    # Customizing the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Impact share', fontsize=14)
    ax.set_ylabel('Impact Categories', fontsize=14)
    ax.set_yticklabels(results_df['Method'], rotation=0, fontsize=12)
    ax.legend(stages, title='Life Cycle Stages', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # Adjust layout for better visibility
    plt.tight_layout()

    # Display the plot
    plt.show()
