# RGCN-Explainer
The idea is to extend GNNExplainer for relational graphs. 

- GNNExp_paper folder: Run GNN_exp.py 


PROBLEMS: 

    1.  The masked adjacency values are not vaguely similar to the one that you can find by running ( which is the PyG implementation)
    
    2. The masked adjacency values are all in a similar range (by tweaking some parameter I can still see that they are just all around the same values -- I would very much say because the explainer model is not learning) 
    
            Example:
            
            masked_adj tensor(indices=tensor([[0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7,
                        7],
                       [1, 5, 7, 0, 4, 6, 5, 7, 1, 5, 6, 0, 2, 4, 7, 1, 4, 0, 3,
                                5]]),
            values=tensor([0.6832, 0.6398, 0.6301, 0.6832, 0.6613, 0.6634, 0.6789,
                            0.6932, 0.6613, 0.6601, 0.6770, 0.6398, 0.6789, 0.6601,
                            0.6884, 0.6634, 0.6770, 0.6301, 0.6932, 0.6884]),
            size=(8, 8), nnz=20, layout=torch.sparse_coo)


    3. The adj_att which I actually use to make prediction over the training of the explain model is instead more different but still very different from how it should be:
    
    
            Example:
            
            adj here: tensor(indices=tensor([[0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7,
                        7],
                       [1, 5, 7, 0, 4, 6, 5, 7, 1, 5, 6, 0, 2, 4, 7, 1, 4, 0, 3,
                        5]]),
            values=tensor([0.7687, 0.5747, 0.5324, 0.7687, 0.6690, 0.6784, 0.7488,
                            0.8150, 0.6690, 0.6635, 0.7400, 0.5747, 0.7488, 0.6635,
                            0.7926, 0.6784, 0.7400, 0.5324, 0.8150, 0.7926]),
            size=(8, 8), nnz=20, layout=torch.sparse_coo, grad_fn=<ToSparseBackward0>)

    4.  If I use the masked adjacency to make prediction - might get the correct label with argmax - but compared to using the original full matrix way lower (you can see by just seeing the last rows of the outuput) : 
    
            Example:
            
            ypred model tensor([0.0118, 0.0176, 0.0048, 0.9284, 0.0175, 0.0138, 0.0060],
                grad_fn=<SoftmaxBackward0>) tensor(3)
            ypred explainer tensor([0.1449, 0.0242, 0.2411, 0.3232, 0.1908, 0.0513, 0.0245],
                grad_fn=<SoftmaxBackward0>) tensor(3)
            ypred_adj_att tensor([0.0764, 0.0263, 0.1581, 0.5828, 0.0796, 0.0213, 0.0556],
                grad_fn=<SoftmaxBackward0>) tensor(3)
            original tensor(3)


In PyG folder: run GNN_exp_pyg.py

    1.  So for example same node as before (0 in Cora): this is the masked adjacency
        Example:
        
        masked adjacency:
        
        tensor(indices=tensor([[   0,    0,    0,  633,  633,  633,  926, 1166, 1701,1701, 1701, 1862, 1862, 1862, 1862, 1866,           1866,2582,
                        2582, 2582],
                       [ 633, 1862, 2582,    0, 1701, 1866, 1862, 2582,  633,
                        1862, 1866,    0,  926, 1701, 2582,  633, 1701,    0,
                        1166, 1862]]),
       values=tensor([0.7199, 0.7134, 0.7199, 0.7160, 0.2791, 0.2728, 0.7189,
                      0.7317, 0.2841, 0.2909, 0.2749, 0.7137, 0.2771, 0.2656,
                      0.7242, 0.7247, 0.2770, 0.7310, 0.2772, 0.6494]),
       size=(2583, 2583), nnz=20, layout=torch.sparse_coo)

    2. And those are the predictions - model way more suere - namely predictions with masked adjacency are better
    
        true label: tensor(3) 
        full model pred label: tensor(3) 
        full model pred prob: tensor(0.9332, grad_fn=<SelectBackward0>) tensor([0.0118, 0.0129, 0.0082, 0.9332, 0.0255, 0.0057, 0.0027],
            grad_fn=<SoftmaxBackward0>) 
        size of full graph: 10556 
        explained pred label: tensor(3) 
        explained pred prob: tensor(0.9393, grad_fn=<SelectBackward0>) tensor([0.0119, 0.0115, 0.0066, 0.9393, 0.0230, 0.0053, 0.0024],
            grad_fn=<SoftmaxBackward0>) 
        size of explained graph: 20


    ### DIfference; here its not that a neighborhood is chosen and work onky on that - but take neighbors index and then select them from the edge indexes (thats why in the matrix before we have actual indexes vs other implementation)


- In the RGCN folder there is all the stuff for the extension when I still thought that the GNNExp I did was alright :') But if we can make it work (and spot whats wrong)     - then the work in this folder is ready to work meaningfully again!
(anyway for explanation: run rgcn_explainer.py)

    - Just mention the fact that this was the masked adjacencey for instance:
    
    masked_ver 
    tensor(indices=tensor([[ 23430,  23444,  23490,  23546,  24301,  24407,  24427,
                         24475,  24503,  24543,  88607, 196312, 229452, 254307,
                        254307, 254307, 254307, 254307, 254307, 254307, 254307,
                        254307, 254307, 328872],
                       [  5757,   5757,   5757,   5757,   5757,   5757,   5757,
                          5757,   5757,   5757,    908,   2227,   1002,   6860,
                          6874,   6920,   6976,   7731,   7837,   7857,   7905,
                          7933,   7973,   5230]]),
       values=tensor([0.8258, 0.8068, 0.7790, 0.5968, 0.7678, 0.6556, 0.7286,
                      0.6311, 0.7508, 0.6904, 0.7022, 0.7445, 0.6636, 0.0736,
                      0.0582, 0.0719, 0.0713, 0.0708, 0.0774, 0.0694, 0.0696,
                      0.0727, 0.0682, 0.7494]),
       size=(753935, 8285), nnz=24, layout=torch.sparse_coo)

       Here the values are more extreme - which is what we want but I dont have a counterfactual for this (of course) to check if that makes sense/it is correct, so I guess I should assume its not. 
       Interesting that its not like all the inverse relations have been zeroeed out (which is what I would more expect also from a seemingly random approach I guess)

    - I was starting with a more systematic possible approach for the semantic evaluation: namely, check which are the most important relations (the ones that in the masked adj values have higher values) - idea to look per class at all those + the class of the nodes of the most important edges (no domain or range specified so relations can have different classes as subject or object)   
     
    - Then other thing would be to try and maybe select the most important relation among the relations of the same type: if there is one (this indeed valuable both for graph summarization /sampling and for the semantic analysis)














- In the GNN_Explainer_easy_implementation.ipynb - I am trying to understand step by step what the authors of [GNN Explainer](https://arxiv.org/abs/1903.03894) did.

- In cora_chk I stored the: Cora trained model, Cora prediction (made by the saved trained model), and the adjacency matrix of the Cora dataset.

- In RGCN_easy_implementation.ipynb - I am diving into the implementation of the rgcn from [kgbench-loader](https://github.com/pbloem/kgbench-loader)



