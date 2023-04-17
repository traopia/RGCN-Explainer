import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import kgbench as kg
import fire, sys
import math
import os
from kgbench import load, tic, toc, d, Data
from rgcn_model import RGCN
from gpu_functions import *


import numpy as np

from rgcn_explainer_utils import prunee

def go(name='am', lr=0.01, wd=0.0, l2=0.0, epochs=50, prune=True, optimizer='adam', final=False,  emb=16, bases=None, printnorms=True):

    include_val = name in ('aifb','mutag','bgs','am', 'IMDb')
    # -- For these datasets, the validation is added to the training for the final eval.

    
    if name == 'IMDb':
        
        data = torch.load('IMDb_typePeople_data.pt')
        data.training = data.training_people
        data.withheld = data.withheld_people
        if prune:
            data = prunee(data, n=2)

    else:

        data = load(name, torch=True)   
        if prune:
            data = prunee(data, n=2)


    #print(f'{data.triples.size(0)} triples')
    print(f'{data.triples.shape[0]} triples ')
    print(f'{data.num_entities} entities')
    if prune:
        print(f'{data.num_entities_new} entities after pruning')
    print(f'{data.num_relations} relations')
    data.triples = torch.tensor(data.triples, dtype=torch.int32)

    tic()
    rgcn = RGCN(data.triples, n=data.num_entities, r=data.num_relations, numcls=data.num_classes, emb=emb, bases=bases)

    if torch.cuda.is_available():
        print('Using cuda.')
        rgcn.cuda()

        data.training = data.training.cuda()
        data.withheld = data.withheld.cuda()
        clean_gpu()

    print(f'construct: {toc():.5}s')

    if optimizer == 'adam':
        opt = torch.optim.Adam(lr=lr, weight_decay=wd, params=rgcn.parameters())
    elif optimizer == 'adamw':
        opt = torch.optim.AdamW(lr=lr, weight_decay=wd, params=rgcn.parameters())
    else:
        raise Exception(f'Optimizer {optimizer} not known')

    for e in range(epochs):
        clean_gpu()
        tic()
        opt.zero_grad()
        out = rgcn()
        print('out',nn.Softmax(dim=0)(out[0][0 :]))

        idxt, clst = data.training[:, 0], data.training[:, 1]
        idxw, clsw = data.withheld[:, 0], data.withheld[:, 1]
        idxt, clst = idxt.long(), clst.long()
        idxw, clsw = idxw.long(), clsw.long()
        out_train = out[idxt, :]
        loss = F.cross_entropy(out_train, clst, reduction='mean')
        if l2 != 0.0:
            loss = loss + l2 * rgcn.penalty()
        correct = []
        mislabeled = []
        # compute performance metrics
        with torch.no_grad():
            training_acc = (out[idxt, :].argmax(dim=1) == clst).sum().item() / idxt.size(0)
            withheld_acc = (out[idxw, :].argmax(dim=1) == clsw).sum().item() / idxw.size(0)
        #     for i in range(idxw.size(0)):
        #         if out[idxw, :].argmax(dim=1)[i] == clsw[i]:
        #             correct.append(idxw[i])
        #         else:
        #             mislabeled.append(idxw[i])
        # print('correct',correct)
        # print('mislabeled',mislabeled)

            #torch.save(out[idxw, :].argmax(dim=1) , 'aifb_chk/prediction_aifb')
        loss.backward()
        opt.step()
        clean_gpu()

        if printnorms is not None:
            # Print relation norms layer 1
            nr = data.num_relations
            weights = rgcn.weights1 if bases is None else rgcn.comps1

            ctr = Counter()

            for r in range(nr):

                ctr[data.i2r[r]] = weights[r].norm()
                ctr['inv_'+ data.i2r[r]] = weights[r+nr].norm()

            print('relations with largest weight norms in layer 1.')
            for rel, w in ctr.most_common(printnorms):
                print(f'     norm {w:.4} for {rel} ')

            weights = rgcn.weights2 if bases is None else rgcn.comps2

            ctr = Counter()
            for r in range(nr):

                ctr[data.i2r[r]] = weights[r].norm()
                ctr['inv_'+ data.i2r[r]] = weights[r+nr].norm()

            print('relations with largest weight norms in layer 2.')
            for rel, w in ctr.most_common(printnorms):
                print(f'     norm {w:.4} for {rel} ')
        #torch.save(out[idxw, :], 'aifb_chk/prediction_aifb')

        if not os.path.exists(f'{name}_chk'):
            os.makedirs(f'{name}_chk')
        
        torch.save(out[idxw, :], f'{name}_chk/prediction_{name}_prune_{prune}')
        #torch.save(rgcn,'aifb_chk/model_aifb')
        torch.save(rgcn, f'{name}_chk/model_{name}_prune_{prune}')
        print(f'epoch {e:02}: loss {loss:.2}, train acc {training_acc:.2}, \t withheld acc {withheld_acc:.2} \t ({toc():.5}s)')



# print('arguments ', ' '.join(sys.argv))
# fire.Fire(go)

if __name__ == '__main__':
    fire.Fire(go)
