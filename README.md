# Introducing RGCNExplainer: The relations that make a difference in a relational world
RGCNExplainer is an extension of GNNExplainer tailored for relational graphs. Its primary purpose is to provide explanations for predictions made in the node classification task within such graph structures. This work also encompasses experiments involving knowledge injection into explanations through various methods of mask initialization.

# Installation
To set up the necessary conda environment and install the required dependencies, follow these steps:
1. Create a conda environment:
```
conda create -n RGCNExplainer python=3.9.16
conda activate RGCNExplainer
pip install -r requirements.txt
pip install . 
```
Ensure that setup.py and the kgbench folder are located in the root directory of your project.
2. If you intend to conduct experiments related to hyperparameter tuning using WANDB (Weights and Biases), export your API key and log in:
```
export WANDB_API_KEY='YOUR_API_KEY'
wandb login
```

# Training your RGCN model
To train your RGCN model, execute the following command, providing the name of the knowledge graph dataset as an argument:

```
python3 RGCNExplainer/rgcn.py 'aifb'
```


# RGCN-Explainer
The pipeline of RGCNExplainer:

![RGCNExplainer_model](https://github.com/traopia/RGCN-Explainer/blob/main/Visualizations/RGCNExplainer_model.jpg)

In order to explain the RGCN prediction on one or more nodes:
The arguments that are to be added are the following:
1. Name of the dataset (in the given examples: 'aifb', 'amplus', 'dbo_gender', 'mdgenre').
2. Mask initialization (choose from: 'normal', 'overall_frequency', 'relative_frequency', 'inverse_relative_frequency', 'domain_frequency', 'range_frequency', 'Domain_Knowledge').
3. If the mask initialization method is 'Domain_Knowledge', provide a relation ID as an integer.
4. If using 'Domain_Knowledge' with baseline domain knowledge, choose between 'forward' and 'backward'.
5. Explain all nodes: --explain all (if True, explain all nodes).
6. Explain one node: --explain_one (if True, explain a random node).
7. Explain a stratified random sample of nodes: --random_sample (if True, explain a stratified per-class random sample of nodes).
8. If using --random_sample, specify the number of samples per class with --num_samples_per_class int.
9. If you want to sweep over the different possible hyperparameters: --sweep
10. If you want to exclude the most frequent relation (typically 'rdf:type') from the explanation: --kill_most_freq_rel.
    

For example, to obtain an explanation for one node:
```
python RGCNExplainer/mainRGCN_explainer.py 'aifb' 'normal' --explain_one
```
Or to get explanations for a stratified sample of nodes:
```
python RGCNExplainer/mainRGCN_explainer.py 'aifb' 'normal' --random_sample --num_samples_per_class 5  
```

# Hyperparameter Configuration
To adjust hyperparameter settings, refer to the configuration file:
```
RGCNExplainer/config.py
```

# Result Analysis
For in-depth analysis, including a table reporting explanation metrics and a barplot comparing relation distribution between the full and explanation subgraph, use the following commands:

For a single explanation:
```
python RGCNExplainer/Result_analysis_one_node.py 'aifb' --node_idx 5731
```

For analysis of explanation results at the class level:

```
python RGCNExplainer/Result_analysis_per_classes.py 'aifb'
```

# Relation Attribution
Another method introduced in this work is relation attribution, which investigates the impact of different relation types on RGCN model performance. Two modalities, 'forward' and 'backward', are explored. 'Forward' predicts node class using only edges of one relation type, while 'backward' iteratively excludes one relation type from predictions made on edges of all other relation types.

To perform experiments with the relation attribution method, run the code with the dataset name and chosen modality as arguments. For example:

```
python3 RGCNExplainer/Relation_Attribution.py 'aifb' 'backward'
```


## DATASET
The experiments conducted in this work utilized datasets introduced in [KGBENCH](http://kgbench.info/).

To use RGCNExplainer with a different knowledge graph, the dataset must be converted to the KGBENCH format following the instructions found in:
```
datasets-conversion/scripts/README.md
```

## PAPER
The Master Thesis associated with this repository is available as [RGCNExplainer.pdf](https://github.com/traopia/RGCN-Explainer/blob/main/RGCNExplainer.pdf).

For any inquiries or further information, please refer to the associated paper and feel free to open a Issue or contact the author.



