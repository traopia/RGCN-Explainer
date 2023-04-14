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


import numpy as np

def prune_f(data , n=2):
    """
    Prune a given dataset. That is, reduce the number of triples to an n-hop neighborhood around the labeled nodes. This
    can save a lot of memory if the model being used is known to look only to a certain depth in the graph.

    Note that switching between non-final and final mode will result in different pruned graphs.

    :param data:
    :return:
    """

    data_triples = data.triples
    data_training = data.training
    data_withheld = data.withheld

    if data.torch:
        data_triples = data_triples.numpy()
        data_training = data_training.numpy()
        data_withheld = data_withheld.numpy()

    assert n >= 1

    entities = set()

    for e in data_training[:, 0]:
        entities.add(e)
    for e in data_withheld[:, 0]:
        entities.add(e)

    entities_add = set()
    for _ in range(n):
        for s, p, o in data_triples:
            if s in entities:
                entities_add.add(o)
            if o in entities:
                entities_add.add(s)
        entities.update(entities_add)

    # new index to old index
    n2o = list(entities)
    o2n = {o: n for n, o in enumerate(entities)}

    nw = data

    nw.num_entities = len(n2o)
    nw.num_relations = data.num_relations

    nw.i2e = [data.i2e[n2o[i]] for i in range(len(n2o))]
    nw.e2i = {e: i for i, e in enumerate(nw.i2e)}

    # relations are unchanged, but copied for the sake of GC
    nw.i2r = list(data.i2r)
    nw.r2i = dict(data.r2i)

    # count the new number of triples
    num = 0
    for s, p, o in data_triples:
        if s in entities and o in entities:
            num += 1

    nw.triples = np.zeros((num, 3), dtype=int)

    row = 0
    for s, p, o in data_triples:
        if s in entities and o in entities:
            s, o =  o2n[s], o2n[o]
            nw.triples[row, :] = (s, p, o)
            row += 1

    nw.training = data_training.copy()
    for i in range(nw.training.shape[0]):
        nw.training[i, 0] = o2n[nw.training[i, 0]]

    nw.withheld = data_withheld.copy()
    for i in range(nw.withheld.shape[0]):
        nw.withheld[i, 0] = o2n[nw.withheld[i, 0]]

    nw.num_classes = data.num_classes

    nw.final = data.final
    nw.torch = data.torch
    if nw.torch:  # this should be constant-time/memory
        nw.triples = torch.from_numpy(nw.triples)
        nw.training = torch.from_numpy(nw.training)
        nw.withheld = torch.from_numpy(nw.withheld)

    return nw

def go(name='IMDb', lr=0.01, wd=0.0, l2=0.0, epochs=50, prune=True, optimizer='adam', final=False,  emb=16, bases=None, printnorms=True):

    include_val = name in ('aifb','mutag','bgs','am', 'IMDb')
    # -- For these datasets, the validation is added to the training for the final eval.

    
    if name == 'IMDb':
        data = torch.load('IMDb_typePeople_data.pt')
        data = prune_f(data, n=2)
    else:
        data = load(name, torch=True, prune_dist=2 if prune else None, final=final, include_val=include_val)    

    #print(f'{data.triples.size(0)} triples')
    print(f'{data.triples.shape[0]} triples ')
    print(f'{data.num_entities} entities')
    print(f'{data.num_relations} relations')
    data.triples = torch.tensor(data.triples, dtype=torch.int32)[:8285]

    tic()
    rgcn = RGCN(data.triples, n=data.num_entities, r=data.num_relations, numcls=data.num_classes, emb=emb, bases=bases)

    if torch.cuda.is_available():
        print('Using cuda.')
        rgcn.cuda()

        data.training = data.training.cuda()
        data.withheld = data.withheld.cuda()

    print(f'construct: {toc():.5}s')

    if optimizer == 'adam':
        opt = torch.optim.Adam(lr=lr, weight_decay=wd, params=rgcn.parameters())
    elif optimizer == 'adamw':
        opt = torch.optim.AdamW(lr=lr, weight_decay=wd, params=rgcn.parameters())
    else:
        raise Exception(f'Optimizer {optimizer} not known')

    for e in range(epochs):
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
        torch.save(out[idxw, :], f'{name}_chk/prediction_{name}')
        #torch.save(rgcn,'aifb_chk/model_aifb')
        torch.save(rgcn, f'{name}_chk/model_{name}')
        print(f'epoch {e:02}: loss {loss:.2}, train acc {training_acc:.2}, \t withheld acc {withheld_acc:.2} \t ({toc():.5}s)')



# print('arguments ', ' '.join(sys.argv))
# fire.Fire(go)

if __name__ == '__main__':
    fire.Fire(go)
