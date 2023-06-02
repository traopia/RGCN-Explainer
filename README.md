# RGCN-Explainer
In the following work GNNExplainer has been expanded to work with relational graph. 
```
python3 RGCN_stuff/r_exp.py
```

For an analysis of the results at class level. 
Insert the path of the experiment (which is given as output of r_exp.py).

```
RGCN_stuff/statistical_analysis.ipynb
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
Then remove_relation.py, final_touch.py (TO BE REVIEWED the whole pipeline)
### TUTORIAL OF GNN EXPLAINER
In order to expland GNNExplainer for relational graph, I created a tutorial where I go by step by step through the source code of 
[GNN Explainer](https://arxiv.org/abs/1903.03894).

```
GNNExplainer/tutorial.ipynb
```





