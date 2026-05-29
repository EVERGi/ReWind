from logging import getLogger
from pathlib import Path
import pandas as pd
import bw2data as bd
import bw2calc as bc
import matplotlib.pyplot as plt

logger = getLogger(__name__)

def lca_phase(dict_activities, impact_category=('IPCC 2021', 'climate change', 'global warming potential (GWP100)')):
    aep=(2000*24*365)*0.3*20
    #@Neil: Here we need the electricity production per turbine and multiply it by a lifetime of 20 years.
    """
    Perform a Life Cycle Assessment (LCA) using the Brightway2.5 package.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA.
    - impact_category (tuple): Impact assessment method to use. Defaults to ('IPCC 2021', 'climate change', 'global warming potential (GWP100)').

    Returns:
    - DataFrame: A DataFrame with method, unit, life cycle stages, their corresponding impact scores, and a 'Total' column.
    """

    # Initialize an empty dictionary to store results
    results = {}
    # Loop over the life cycle stages in the dictionary
    for stage, materials in dict_activities.items():
        # Create an LCA object for the materials in this stage
        lca = bc.LCA(demand=materials, method=impact_category)
        lca.lci()
        lca.lcia()
        # Store the result in the dictionary
        results[stage] = lca.score / aep  # Assuming 'aep' is defined elsewhere in your code

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

def lca_phase_pie_chart(dict_activities, impact_category=('IPCC 2021', 'climate change', 'global warming potential (GWP100)')):
    """
    Perform a Life Cycle Assessment (LCA) using the Brightway2.5 package and create a pie chart.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA.
    - P (int): Power of the wind turbine in kW or MW.
    - offshore (bool): Whether the wind turbine is offshore (True) or onshore (False).
    - impact_category (tuple): Impact assessment method to use. Defaults to ('IPCC 2021', 'climate change', 'global warming potential (GWP100)').

    Returns:
    - DataFrame: A DataFrame with method, unit, life cycle stages, their corresponding impact scores, a 'Total' column, and a pie chart showing percentage contribution.
    """

    aep=(2000*24*365)*0.3*20
    #@Neil: Here we need the electricity production per turbine and multiply it by a lifetime of 20 years.

    # Initialize an empty dictionary to store results
    results = {}

    # Loop over the life cycle stages in the dictionary
    for stage, materials in dict_activities.items():
        # Create an LCA object for the materials in this stage
        lca = bc.LCA(demand=materials, method=impact_category)
        lca.lci()
        lca.lcia()

        # Store the result in the dictionary
        results[stage] = lca.score / aep  # Assuming 'aep' is defined elsewhere in your code

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

    # Remove the 'Total' key for the pie chart (as it sums all stages)
    stages = list(results.keys())[:-1]
    stage_values = list(results.values())[:-1]

    # Calculate the percentage contribution of each stage to the total impact
    percentages = [round((value / total_impact) * 100) for value in stage_values]

    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(percentages, labels=stages, autopct='%1.0f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add the turbine power and location to the chart title
    plt.title(f"Climate Change Impact per Stage of a wind turbine")
    plt.show()

    return results_df

def lca_multi_impact(dict_activities):
    """
    Perform a Life Cycle Assessment (LCA) for EF v3.1 methods using the Brightway2.5 package.

    Parameters:
    - dict_activities (dict): Dictionary containing the functional unit for the LCA, organized by life cycle stages.
    - aep (float): Adjustment factor. Assumed to be defined elsewhere.

    Returns:
    - DataFrame: A DataFrame with method, unit, life cycle stages, their corresponding impact scores for each method, and a 'Total' column.
    """

    aep=(2000*24*365)*0.3*20
    #@Neil: Here we need the electricity production per turbine and multiply it by a lifetime of 20 years.
    
    # Define the EF v3.1 methods, excluding those with 'EN15804' and 'no LT'
    ef_methods = [m for m in bd.methods if 'EF v3.1' in str(m) and 'EN15804' not in str(m) and 'no LT' not in str(m)]

    # Initialize an empty list to store DataFrames
    results_list = []

    # Loop over each EF v3.1 method in the filtered list
    for method in ef_methods:
        # Initialize a dictionary to store results for the current method
        results = {}

        # Loop over the life cycle stages in the dictionary
        for stage, materials in dict_activities.items():
            # Perform LCA for the current stage and method
            lca = bc.LCA(demand=materials, method=method, use_distributions=False)
            lca.lci()
            lca.lcia()

            # Store the result for the current stage
            results[stage] = lca.score / aep  # Assuming 'aep' is defined elsewhere

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