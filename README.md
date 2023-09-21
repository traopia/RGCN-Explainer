This work introduces RGCNExplainer, an extension of GNNExplaiener for relational graph, to explain predictions made on the node classification task. 
Further experiments have been performed on knowledge injection in the explanations via different methods for mask initialization. 

# Set up the conda Environment
```
conda create -n RGCNExplainer python=3.9.16
conda activate RGCNExplainer
pip install -r requirements.txt
pip install . (make sure that setup.py and kgbench folder are in the root directory)
```
If you want to perform the experiments on hyperparameter tuning using WANDB: 
```
export WANDB_API_KEY='YOUR_API_KEY'
wandb login
```

# Train your RGCN model
Run this command using as argument the name of the KG (dataset available: 'aifb', 'AMPLUS'):

```
python3 RGCNExplainer/rgcn.py 'aifb'
```


# RGCN-Explainer
The pipeline of RGCNExplainer:

![RGCNExplainer_model](https://github.com/traopia/RGCN-Explainer/blob/main/Visualizations/RGCNExplainer_model.jpg)

In order to explain the RGCN prediction on one or more nodes:
The arguments that are to be added are the following:
1. Name of the dataset (in the given examples: 'aifb', 'AMPLUS', 'dbo_gender', 'mdgenre')
2. Mask initialization (choose from: 'normal', 'overall_frequency', 'relative_frequency', 'inverse_relative_frequency', 'domain_frequency', 'range_frequency', 'Domain_Knowledge')
3. If Mask initialization is 'Domain_Knowledge': relation_id - as an integer
4. If Mask initialization is 'Domain_Knowledge' and use the baseline domain Knowledge: choose between 'forward' and 'backward'
5. Explain all nodes: --explain all , if True explain all the nodes
6. Explain one node : --explain_one, if True explaina random node
7. Explain a stratified random sample of nodes: --random_sample, if True explain a stratified per class random sample of nodes
8. If --random_sample: specify how many samples per class with --num_samples_per_class int
9. If you want to sweep over the different possible hyperparameters: --sweep
10. If you want to not consider in the explanation the most frequent relation (in most cases it´s going to be the rdf:type relation) : --kill_most_freq_rel
    
```
python3 RGCNExplainer/mainRGCN_explainer.py
```
example to get explanation for one node:
```
python RGCNExplainer/mainRGCN_explainer.py 'aifb' 'normal' --explain_one
```
example to get explanations for stratified sample of nodes:
```
python RGCNExplainer/mainRGCN_explainer.py 'aifb' 'normal' --random_sample --num_samples_per_class 
```
Experiments have been conducted to find the best hyperparameter setting, but in order to change the hyperparameter configurations, see:
```
RGCNExplainer/config.py
```

For analysis (include table reporting explanation metrics and barplot of relation distribution comparison between full and explanation subgraph) and visualization of a single explanation:
```
python RGCNExplainer/Result_analysis_one_node.py 'aifb' --node_idx 5731
```

For analysis of the explanation results at class level:

```
python RGCNExplainer/Result_analysis_per_classes.py 'aifb'
```

# RELATION ATTRIBUTION
Another method has been introduced to study the impact of different relation types on the RGCN model performance. 
Two modality are given: called 'forward' and 'backward'. For the 'forward' a prediction on the node class is made having access to only edges of one relation type, whilst in the ´backward´ prediction are made on edges of all relation types except iteratively one. 
In order to perform experiments with the relation attribution method, run the code, with argument the name of the KG and the chosen modality.
Example:

```
python3 RGCNExplainer/Relation_Attribution.py 'aifb' 'backward'
```


## DATASET
The experiments have been performed on datasets introduced in [KGBENCH](http://kgbench.info/).

In order to use RGCNExplainer for a different knowledge graph, the dataset has to be converted in the KGBENCH format following the instructions, which can be found in:
```
datasets-conversion/scripts/README.md
```

## PAPER
The Master Thesis associated to this repository can be found as [RGCNExplainer.pdf](https://github.com/traopia/RGCN-Explainer/blob/main/RGCNExplainer.pdf)



