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

from src.rgcn_explainer_utils import *
from rgcn import RGCN

import argparse

import stat
import numpy as np
from scipy.stats import chi2_contingency
import os

from matplotlib import legend
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




#GET METRICS OF EXPLANATION PERFORMANCE

def metrics_one_node(path, node_idx ):

    df = pd.read_csv(path+f'/Relations_Important_{node_idx}_threshold.csv', index_col = 'label')
    result_df = pd.DataFrame()
    columns = ['score_threshold_prob','sparsity_threshold','fidelity_minus_threshold','fidelity_plus_threshold','fidelity_minus_random','fidelity_plus_random']
    group = df[columns].mean()
    
    score = np.round(group['sparsity_threshold'],5) + np.round(group['fidelity_minus_threshold'],3) + np.round(group['fidelity_plus_threshold'],3)
    sparsity = np.round(group['sparsity_threshold'],5)
    fidelity_minus = np.round(group['fidelity_minus_threshold'],5)
    fidelity_plus = np.round(group['fidelity_plus_threshold'],5)
    fidelity_minus_random = np.round(group['fidelity_minus_random'],5)
    fidelity_plus_random = np.round(group['fidelity_plus_random'],5)
    score_threshold_prob = np.round(group['score_threshold_prob'],5)
    return score,score_threshold_prob, sparsity, fidelity_minus, fidelity_plus, fidelity_minus_random, fidelity_plus_random


def table_metrics_overview_one_node(name, params, node_idx, all_inits):
    '''Overview of the metrics for the different initializations
    name: name of the dataset
    params: parameters of the experiment (after init)'''

    if all_inits:
        init = ['normal','overall_frequency','relative_frequency','inverse_relative_frequency','domain_frequency','range_frequency','Domain_Knowledge_forward','Domain_Knowledge_backward']
    else:
        init = ['normal']

    df = pd.DataFrame(columns = ['init','Score', 'Score Prob','Sparsity', 'Fidelity-', 'Fidelity+', 'Fidelity- random', 'Fidelity+ random'])
    for i in init:

        m = metrics_one_node(f'chk/{name}_chk/exp/init_{i}_{params}/Relation_Importance', node_idx)
        df.loc[len(df)] = [i] + list(m)
    df.set_index('init', inplace=True)
    table = df.to_latex(index=True, caption = name, label = name,column_format='|c|c|c|c|c|c|c|')
    latex_table = table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\toprule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline')
    latex_table = latex_table.replace('\\begin{tabular}', '\\begin{adjustbox}{scale=0.5}\\begin{tabular}')  # Add scaling parameter
    latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\\end{adjustbox}')  # Close the adjustbox environment
    print(latex_table)
    return latex_table



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

def path_get(init,data):
    path = f'chk/{data.name}_chk/exp/init_{init}_lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1/Relation_Importance/'
    return path

import json
path_dict = 'chk/mdgenre_chk/name_relations_mdgenre.json'
# with open(path_dict, 'r') as json_file:
#     dict_name_relations = json.load(json_file)

def path_get(init,data):
    path = f'chk/{data.name}_chk/exp/init_{init}_lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1/Relation_Importance/'
    return path
def modified_counter(path,relations):
    df = pd.read_csv(path, index_col = 'label')[relations]
    df = keep_columns_with_non_zero_values(df)
    mean_df = df.groupby('label').mean()
    mean_df = select_columns_with_values_above_threshold(mean_df, 1)
    return mean_df  

def bar_plot_one(data,init , relations, node_idx, label):
    full = modified_counter(f'chk/{data.name}_chk/exp/init_{init}_lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1/Relation_Importance/Relations_Important_full_neighborhood.csv', relations)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,8))
    full_dict = full.loc[label].to_dict()
    explanation_dicts = {}
    colormap = plt.cm.Set1.colors 
    explanation_methods = ['normal']
    for method in explanation_methods:
        explain_counter = modified_counter(path_get('normal',data)+f'/Relations_Important_{node_idx}_threshold.csv', relations)
        explanation_dicts[method] = explain_counter.loc[label].to_dict()
    bar_positions_full_graph, _, sorted_df, bar_width = bar_plot(full_dict, full_dict)
    bar_width_explanation = bar_width / len(explanation_methods) 

    bar_positions_explanation_adjusted = bar_positions_full_graph + bar_width_explanation
    ax = axes
    width = bar_width / len(explanation_methods)
    ax.bar(bar_positions_full_graph, sorted_df['Percentage_full'], width=width, alpha=0.7, color='black', edgecolor='black', label='Full Graph')

    for idx, method in enumerate(explanation_methods):
        bar_positions_explanation, _, sorted_df, _ = bar_plot(full_dict, explanation_dicts[method])
        ax.bar(bar_positions_explanation_adjusted + idx * bar_width_explanation, sorted_df['Percentage_explanation'], width=width, alpha=0.7, color=colormap[idx], edgecolor='gray', label=f'{method.replace("_", " ").title()} Explanation')

    ax.set_xticks(bar_positions_full_graph + bar_width / 2)
    ax.set_xticklabels(sorted_df['Relation Name'], rotation=55, fontsize=8)
    ax.set_ylabel('Relative Frequency (%)')
    ax.set_title(f'{data.name} : Distribution of Relations per node {node_idx}', fontsize= 10)
    ax.legend(loc='upper right', fontsize=12)  
    pdf_filename = f"Visualizations/{data.name}_relation_distribution_{node_idx}.jpg"
    plt.savefig(pdf_filename, format="jpg")
    plt.show()


#VISUALIZATIONS





def connect_minimally(disconnected_triples, guideline_triples):
    G = nx.Graph()
    G.add_edges_from([(int(u), int(v), {'weight': label}) for u, label, v in disconnected_triples])
    
    is_connected = nx.is_connected(G)
    
    if not is_connected:
        disconnected_nodes = {u for u, _, _ in disconnected_triples} | {v for _, _, v in disconnected_triples}
        subgraph = nx.Graph([(int(u), int(v), {'weight': label}) for u, label, v in guideline_triples if int(u) in disconnected_nodes and int(v) in disconnected_nodes])
        min_spanning_tree = nx.minimum_spanning_tree(subgraph)
        G.add_edges_from(min_spanning_tree.edges(data=True))
    
    return G

def extract_node_label(node_value):
    if 'http' in node_value:
        split_result = node_value.split('/')
        if len(split_result) >= 4:
            if '#' in node_value:
                return split_result[3].split('#')[0]
            else:
                return split_result[3]
    return 'blank'


def get_wikidata_main_title(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        main_title = soup.find("span", {"class": "wikibase-title-label"}).text.strip()
        return main_title
    else:
        pass

def visualize(node_idx,data,init, masked_ver,masked_hor, result_weights=True, low_threshold=False,experiment_name=None, selected_visualization=True, connected_subgraph = True,make_connected=False):
    """ 
    Visualize important nodes for node idx prediction
    """
    name = data.name
    n_hop = 2
    threshold = 0.8

    if selected_visualization:
        sel_masked_ver, sel_masked_hor = sub_sparse_tensor(masked_ver, threshold,data, low_threshold),sub_sparse_tensor(masked_ver, threshold,data, low_threshold)

    else:
        sel_masked_ver, sel_masked_hor = masked_ver, masked_ver
    if len(sel_masked_ver)==0:
        sel_masked_ver=sub_sparse_tensor(masked_ver, 0,data, low_threshold)
    indices_nodes = sel_masked_ver.coalesce().indices().detach().numpy()
    
    G = nx.Graph()

    triples_matched = match_to_triples(sel_masked_ver,sel_masked_hor, data)
    G.add_edges_from([(int(s), int(o), {'weight': p}) for s, p, o in find_repeating_sublists(triples_matched.tolist())])

    if connected_subgraph:
        G = G.subgraph(next(comp for comp in nx.connected_components(G) if node_idx in comp))

    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    weights = [list(set([item] if not isinstance(item, list) else item)) for item in weights]

    ordered_dict = {}
    for node in G.nodes:
        node_value = str(data.i2e[int(node)])
        node_label = extract_node_label(node_value)
        ordered_dict[int(node)] = node_label
        


    dict_node_color = {k: data.entities_classes[v] if v in data.entities_classes else 0 for k, v in ordered_dict.items()} 
    dict_node_color[node_idx] = 7
    labeldict = {int(node): int(node) for node in G.nodes}
    pos = nx.circular_layout(G)

    
    if make_connected:
        disconnected_triples = [[i[0], j[0], i[1]] for i, j in zip(edges, weights)] + [[i[0], j[1], i[1]] for i, j in zip(edges, weights) if len(j) > 1]; guideline_triples = data.triples.tolist()
        G = connect_minimally(disconnected_triples, guideline_triples)
        G = G.subgraph(next(comp for comp in nx.connected_components(G) if node_idx in comp))
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        labeldict = {int(node): int(node) for node in G.nodes}
        if name == 'dbo_gender' :
            labeldict = {int(node): str(data.i2e[int(node)]).split('/')[-1].split(',')[0] for node in G.nodes}

        rel = {k: [v] for k, v in nx.get_edge_attributes(G, 'weight').items()}
            
        rel = {k: [data.i2rel[item][0] for item in v] if isinstance(v, list) else data.i2rel[v][0] for k, v in rel.items()}

        pos = nx.spring_layout(G,seed=55)
            
        min_x = min(pos[node][0] for node in pos)
        max_x = max(pos[node][0] for node in pos)
        shift = (max_x - min_x) * 0.05  # Adjust the value 0.05 to control the shift amount
        for node in pos:
            if pos[node][0] == max_x:
                pos[node] = (pos[node][0] - shift, pos[node][1])

        col_weights = [weights[i] for i in range(len(weights))]

        color_map = plt.cm.get_cmap('prism')
        colors = [color_map(col_weights[i]) for i in range(len(col_weights))]
        
        if name == 'mdgenre':
            label_colors = {get_wikidata_main_title(data.i2rel[i][1]): j for i,j in zip(col_weights,colors)}
        else:
            label_colors = {data.i2rel[i][0]:j for i,j in zip(col_weights,colors)}
        legend_elements = []
       
        for label, color in label_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10))
        legend_elements1 = []
        cmap = plt.cm.get_cmap('tab20')
        mapped_dict2 = {key: cmap(value) for key, value in dict_node_color.items()}
        combined_dict = {ordered_dict[key]: value for key, value in mapped_dict2.items() if key in ordered_dict}
        for label, color in combined_dict.items():
            legend_elements1.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10))



        node_colors = [mapped_dict2[i]  if i in mapped_dict2 else cmap(0) for i in G.nodes]
        if name == 'mdgenre':
            edge_colors = [label_colors[get_wikidata_main_title(data.i2rel[i][1])] for i in col_weights]
        else:
            edge_colors = [label_colors[data.i2rel[i][0]] for i in col_weights]
        nx.draw(G, pos,labels = labeldict, edgelist=edges,node_color = node_colors  ,font_size=7, arrows = True,edge_color=edge_colors)
        plt.legend(handles=legend_elements, loc='lower right',title= 'Relations',fontsize='small')
        ax = plt.gca().twinx()

        ax.legend(handles=legend_elements1,  title='Classes', loc='upper right',fontsize='small')

        plt.title(f" {data.name}: Subgraph Explanation of node {node_idx} with {init} initialization ")
        plt.savefig(f'Visualizations/{data.name}_subggraphExplanation_{node_idx}_{init}.jpg')
        plt.show()
    return edges#, weigths
        



#MDGENRE
genres = {
    0: ['Romance Film', 'https://www.wikidata.org/wiki/Q1054574'],
    1: ['Fiction Film', 'https://www.wikidata.org/wiki/Q12912091'],
    2: ['Drama Film', 'https://www.wikidata.org/wiki/Q130232'],
    3: ['Fantasy Film', 'https://www.wikidata.org/wiki/Q157394'],
    4: ['Comedy Film ', 'https://www.wikidata.org/wiki/Q157443'],
    5: ['Action Film', 'https://www.wikidata.org/wiki/Q188473'],
    6: ['Experimental Film','https://www.wikidata.org/wiki/Q790192'],
    7: ['Film based on Literature', 'https://www.wikidata.org/wiki/Q52162262'],
    8: ['Musical Film', 'https://www.wikidata.org/wiki/Q842256'],
    9: ['Comedy Drama', 'https://www.wikidata.org/wiki/Q859369'],
    10: ['Romantic Comedy','https://www.wikidata.org/wiki/Q860626'],
    11: ['Documentary Film', 'https://www.wikidata.org/wiki/Q93204']
}





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name',help='name of the dataset')
    parser.add_argument('--all_inits',action='store_true',help='True if all inits, False if only normal init')
    parser.add_argument('--node_idx',help='node index', type=int)
    name = parser.parse_args().name
    all_inits = parser.parse_args().all_inits
    node_idx = parser.parse_args().node_idx
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
    label = int(data.withheld[torch.where(data.withheld[:, 0] == torch.tensor([node_idx])),1])

    #performance metrics 
    table_performance_one_node = table_metrics_overview_one_node(name,'lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1', node_idx, all_inits) 
    
    #barplot of relation distribution
    bar_plot_one(data,init , relations, node_idx, label)

    #visualize explanation
    path = f'chk/{name}_chk/exp/init_{init}_lr_0.5_size_0.0005_ent_1_type_1_wd_0.9_MFR_1/masked_adj'
    v,h = torch.load(f'{path}/masked_ver_thresh{node_idx}'),torch.load(f'{path}/masked_hor_thresh{node_idx}')
    h_t, v_t,t,t = threshold_mask(h,v,data,15)
    edges = visualize(node_idx, data,init,  v_t,h_t,result_weights=False, low_threshold=False,experiment_name=None, selected_visualization=True,connected_subgraph=False,make_connected=True)












if __name__ == '__main__':
    main()