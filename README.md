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
Then remove_relation.py, final_touch.py for the final conversions.






