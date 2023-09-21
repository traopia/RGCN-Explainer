import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import Counter
import src.kgbench as kg


from kgbench import load, tic, toc, d
from matplotlib.font_manager import FontProperties

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import legend

from src.rgcn_explainer_utils import *
from rgcn import RGCN


import argparse
import os


from networkx import spring_layout
import requests
from bs4 import BeautifulSoup

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
path_components = current_dir.split(os.path.sep)
if path_components[-1] != "RGCN-Explainer":
    os.chdir(parent_dir)

current_dir = os.getcwd()

print("Current Directory:", current_dir)





def metrics(path, per_class = False, sample = True):
    if sample:
        sample= 'sample'
    else:
        sample = 'full'
    df = pd.read_csv(path+f'/Relations_Important_{sample}_threshold.csv', index_col = 'label')
    result_df = pd.DataFrame()
    if per_class:
        group = df.groupby('label').mean()
        result_df['label'] = group.index
    else:
        columns = ['sparsity_threshold','fidelity_minus_threshold','fidelity_plus_threshold','fidelity_minus_random','fidelity_plus_random']
        group = df[columns].mean()
    
    score = np.round(group['sparsity_threshold'],3) + np.round(group['fidelity_minus_threshold'],3) + np.round(group['fidelity_plus_threshold'],3)
    sparsity = np.round(group['sparsity_threshold'],3)
    fidelity_minus = np.round(group['fidelity_minus_threshold'],3)
    fidelity_plus = np.round(group['fidelity_plus_threshold'],3)
    fidelity_minus_random = np.round(group['fidelity_minus_random'],3)
    fidelity_plus_random = np.round(group['fidelity_plus_random'],3)
    if per_class:
        result_df['Score'] = score
        result_df['Sparsity'] = sparsity
        result_df['Fidelity-'] = fidelity_minus
        result_df['Fidelity+'] = fidelity_plus
        result_df['Fidelity- random'] = fidelity_minus_random
        result_df['Fidelity+ random'] = fidelity_plus_random


        table = result_df.to_latex(index=True, caption = path.split('/')[-2].replace('_','-'), label = path.split('/')[-2].replace('_','-'),column_format='|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|')
        latex_table = table.replace('\\midrule', '\\hline')
        latex_table = latex_table.replace('\\toprule', '\\hline')
        latex_table = latex_table.replace('\\bottomrule', '\\hline')
        latex_table = latex_table.replace('\\begin{tabular}', '\\begin{adjustbox}{scale=0.5}\\begin{tabular}')  # Add scaling parameter
        latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\\end{adjustbox}')  # Close the adjustbox environment
        return latex_table

    else: 
        return score, sparsity, fidelity_minus, fidelity_plus, fidelity_minus_random, fidelity_plus_random


def table_metrics_overview(name, params ,sample=True, all_inits=False, per_class =False):
    '''Overview of the metrics for the different initializations
    name: name of the dataset
    params: parameters of the experiment (after init)'''
    path = f'chk/{name}_chk/exp/init_normal_{params}/Relation_Importance'
    if per_class == False:
        if all_inits:
            init = ['normal','overall_frequency','relative_frequency','inverse_relative_frequency','domain_frequency','range_frequency','Domain_Knowledge_forward','Domain_Knowledge_backward']
        else:
            init = ['normal']
        df = pd.DataFrame(columns = ['init','Score','Sparsity', 'Fidelity-', 'Fidelity+', 'Fidelity- random', 'Fidelity+ random'])
        for i in init:

            m = metrics(f'chk/{name}_chk/exp/init_{i}_{params}/Relation_Importance',per_class,sample)
            df.loc[len(df)] = [i] + list(m)

        df.set_index('init', inplace=True)
        table = df.to_latex(index=True, caption = name, label = name,column_format='|c|c|c|c|c|c|c|')
        latex_table = table.replace('\\midrule', '\\hline')
        latex_table = latex_table.replace('\\toprule', '\\hline')
        latex_table = latex_table.replace('\\bottomrule', '\\hline')
        latex_table = latex_table.replace('\\begin{tabular}', '\\begin{adjustbox}{scale=0.5}\\begin{tabular}')  # Add scaling parameter
        latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\\end{adjustbox}')  # Close the adjustbox environment
        print(latex_table)
    else:
        latex_table = metrics(path, per_class , sample)
        print(latex_table)

    return latex_table


#BARPLOTS

def keep_columns_with_non_zero_values(df):
    ''' Keep only columns with non-zero values'''

    df = df.fillna(0)
    # Get the column names with non-zero values
    non_zero_columns = df.columns[df.astype(bool).any(axis=0)]

    # Create a new DataFrame with only the columns containing non-zero values
    modified_df = df[non_zero_columns]

    return modified_df

def select_columns_with_values_above_threshold(df, threshold):
    # Select columns where there is at least one entry higher than 1
    selected_columns = df.columns[(df > threshold).any()]

    # Create a new DataFrame with the selected columns
    modified_df = df[selected_columns]

    return modified_df

def modified_counter(path,relations):
    df = pd.read_csv(path, index_col = 'label')[relations]
    df = keep_columns_with_non_zero_values(df)
    
    mean_df = df.groupby('label').mean()
    mean_df = select_columns_with_values_above_threshold(mean_df, 1)
    return mean_df   


def bar_plot(full, explain):
    ''' Create a bar plot (histograms with adjacent bars) for the full graph and the explanation subgraph'''
    full_graph_df = pd.DataFrame(list(full.items()), columns=['Relation Name', 'Count'])
    explanation_df = pd.DataFrame(list(explain.items()), columns=['Relation Name', 'Count'])

    # Merge the DataFrames and calculate the total counts
    merged_df = pd.merge(full_graph_df, explanation_df, on='Relation Name', how='outer', suffixes=('_full', '_explanation'))
    merged_df = merged_df.fillna(0)
    total_full_graph = merged_df['Count_full'].sum()
    total_explanation = merged_df['Count_explanation'].sum()

    # Calculate the percentages for each relation in both full graph and explanation subgraph
    merged_df['Percentage_full'] = merged_df['Count_full'] / total_full_graph * 100
    merged_df['Percentage_explanation'] = merged_df['Count_explanation'] / total_explanation * 100

    # Sort the DataFrame by the counts in descending order
    sorted_df = merged_df.sort_values(by='Count_full', ascending=False)

    # Create the bar plot (histograms with adjacent bars)
    bar_width = 0.4
    bar_positions_full_graph = np.arange(len(sorted_df))
    bar_positions_explanation = bar_positions_full_graph + bar_width
    return bar_positions_full_graph, bar_positions_explanation, sorted_df, bar_width

def path_get(init,data):
    path = f'chk/{data.name}_chk/exp/init_{init}_lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1/Relation_Importance/'
    return path


#SRC

def keep_columns_with_non_zero_values(df):
    ''' Keep only columns with non-zero values'''

    df = df.fillna(0)
    # Get the column names with non-zero values
    non_zero_columns = df.columns[df.astype(bool).any(axis=0)]

    # Create a new DataFrame with only the columns containing non-zero values
    modified_df = df[non_zero_columns]

    return modified_df

def select_columns_with_values_above_threshold(df, threshold):
    # Select columns where there is at least one entry higher than 1
    selected_columns = df.columns[(df > threshold).any()]

    # Create a new DataFrame with the selected columns
    modified_df = df[selected_columns]

    return modified_df

def modified_counter(path,relations):
    df = pd.read_csv(path, index_col = 'label')[relations]
    df = keep_columns_with_non_zero_values(df)
    
    mean_df = df.groupby('label').mean()
    mean_df = select_columns_with_values_above_threshold(mean_df, 1)
    return mean_df   


def bar_plot(full, explain):
    ''' Create a bar plot (histograms with adjacent bars) for the full graph and the explanation subgraph'''
    full_graph_df = pd.DataFrame(list(full.items()), columns=['Relation Name', 'Count'])
    explanation_df = pd.DataFrame(list(explain.items()), columns=['Relation Name', 'Count'])

    # Merge the DataFrames and calculate the total counts
    merged_df = pd.merge(full_graph_df, explanation_df, on='Relation Name', how='outer', suffixes=('_full', '_explanation'))
    merged_df = merged_df.fillna(0)
    total_full_graph = merged_df['Count_full'].sum()
    total_explanation = merged_df['Count_explanation'].sum()

    # Calculate the percentages for each relation in both full graph and explanation subgraph
    merged_df['Percentage_full'] = merged_df['Count_full'] / total_full_graph * 100
    merged_df['Percentage_explanation'] = merged_df['Count_explanation'] / total_explanation * 100

    # Sort the DataFrame by the counts in descending order
    sorted_df = merged_df.sort_values(by='Count_full', ascending=False)

    # Create the bar plot (histograms with adjacent bars)
    bar_width = 0.4
    bar_positions_full_graph = np.arange(len(sorted_df))
    bar_positions_explanation = bar_positions_full_graph + bar_width
    return bar_positions_full_graph, bar_positions_explanation, sorted_df, bar_width

def bar_plot_and_plot(data,init, relations, sample = True):
    ''' Plot the distribution of relations per class for the full graph and the explanation subgraph'''
    #exp = f'init_{init}_lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1'

    path = f'chk/{data.name}_chk/exp/init_{init}_lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1/Relation_Importance/'
    if sample:
        explain = modified_counter(path+'/Relations_Important_sample_threshold.csv',relations)
    else:
        explain = modified_counter(path+'/Relations_Important_full_threshold.csv',relations)    
    full = modified_counter(path+'/Relations_Important_full_neighborhood.csv',relations)

    # Calculate the number of rows and columns for subplots based on data.num_classes
    num_classes = data.num_classes
    num_rows = min(int(np.ceil(num_classes / 2)), 2)
    num_cols = min(num_classes, 2)  
    if data.name == 'amplus':
        num_rows = 4
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 15))
    else:
        # Create the figure and axes
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 8))
    

    # Iterate through the labels and plot the data
    for label, ax in zip(range(num_classes), axes.flat):
        full_dict = full.loc[label].to_dict()
        explain_dict = explain.loc[label].to_dict()
        bar_positions_full_graph, bar_positions_explanation, sorted_df, bar_width = bar_plot(full_dict, explain_dict)

        ax.bar(bar_positions_full_graph, sorted_df['Percentage_full'], width=bar_width, alpha=0.7, color='blue', edgecolor='black', label='full graph')
        ax.bar(bar_positions_explanation, sorted_df['Percentage_explanation'], width=bar_width, alpha=0.7, color='green', edgecolor='black', label=f'{init} explanation subgraph')
        ax.set_xticks(bar_positions_full_graph + bar_width / 2)
        ax.set_xticklabels(sorted_df['Relation Name'], rotation=75)
        ax.set_ylabel('Relative Frequency (%)')
        ax.set_title(f'Distribution of Relations per class {label}')
        ax.legend(loc='upper right')

    # Adjust the layout and spacing
    plt.tight_layout()
    plt.suptitle(f'Relation distribution {data.name} ', fontsize=16, y=0.95,fontproperties=FontProperties(weight='bold'))
    plt.subplots_adjust(top=0.85) 

    # Show the plots
    plt.show()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name',help='name of the dataset')
    parser.add_argument('--all_inits',action='store_true',help='True if all inits, False if only normal init')
    name = parser.parse_args().name
    all_inits = parser.parse_args().all_inits
    if all_inits == False:
        init = 'normal'

    if name in ['aifb', 'mutag', 'mdgenre', 'amplus', 'mdgender']:
        data = kg.load(name, torch=True, final=False)
    if 'dbo' in name:
        data = torch.load(f'data/DBO/finals/{name}.pt')
    
    data.triples = torch.Tensor(data.triples).to(int)
    data.withheld = torch.Tensor(data.withheld).to(int)
    data.training = torch.Tensor(data.training).to(int)
    relations = get_relations(data)
    dict_classes = d_classes(data)


    #metrics of explanation averaged all together
    table_metrics_overview(name,'lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1', sample=True, all_inits=all_inits, per_class =False)


    #metrics of explanation averaged per class
    if all_inits == False:
        table_metrics_overview(name,'lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1', sample=True, all_inits=all_inits, per_class = True)
    

    #barplot of the metrics per class
    bar_plot_and_plot(data, 'normal',relations)

    








if __name__ == '__main__':
    main()