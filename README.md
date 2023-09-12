# Environment
In order to run the code:
```
conda create -n RGCNExplainer python=3.9
pip install -r req.txt
pip install . (make sure that setup.py and kgbench folder are in the root directory)
export WANDB_API_KEY='YOUR_API_KEY'
wandb login
```


# RGCN-Explainer
In the following work GNNExplainer has been expanded to work with relational graph, and experiments have been performed on mask initializations.
The pipeline of the introduced method RGCNExplainer:

![RGCNExplainer_model](https://github.com/traopia/RGCN-Explainer/assets/91891769/3ca2976a-c5d8-4041-8777-e39573620977)

In order to Explain the prediction on one or more nodes:
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
    
```
python3 RGCN_stuff/mainR_explainer.py
```

In order to change the hyperparameter configurations, see:
```
RGCN_stuff/config.py
```

For an analysis of the results at class level. 
Insert the path of the experiment (which is given as output of r_exp.py).

```
RGCN_stuff/result_analysis.ipynb
```
# RELATION SELECTION
```
python3 RGCN_stuff/baseline.py
```
For an analysis of the results at class level:
```
RGCN_stuff/statistics_baseline.ipynb
```

## DATASET
In order to convert a knowledge graph dataset to the format used in this implementation:
Follow the instructions in 
```
datasets-conversion/scripts/README.md
```
Then remove_relation.py, final_touch.py for the final conversions.






