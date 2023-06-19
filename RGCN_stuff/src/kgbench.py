### In this file the functions defined in KGBench are used to load the datasets: 
## from the KGBench repository: https://github.com/pbloem/kgbench-loader
## The functions are slightly modified to be able to load the datasets in the RGCN_stuff repository.

import numpy as np

from os.path import join
from pathlib import Path

import pandas as pd
import gzip, base64, io, sys, warnings, wget, os, random

import torch
import gzip
import csv


#from kgbnech 

import rdflib as rdf
import time, zipfile, tarfile

"""
Dictionary  containing the download URLS for the datasets.
"""

URLS = {
    'aifb' :    ['https://www.dropbox.com/s/pig4tu771akgll0/aifb.tgz?dl=1'],
    'amplus':   ['https://www.dropbox.com/s/12yp5exeuujfkyp/amplus.tgz?dl=1'],
    'dblp' :    ['https://www.dropbox.com/s/zc3pdho1k7xr520/dblp.tgz?dl=1'],
    'mdgenre':  ['https://www.dropbox.com/s/uksmcbfv23cjsa8/mdgenre.tgz?dl=1'],
    'mdgender': ['https://www.dropbox.com/s/irb997j1yc6ohq3/mdgender.tgz?dl=1'],
    'dmgfull':  ['https://www.dropbox.com/s/59tfdawqdxvpvm8/dmgfull.tgz?dl=1'],
    'dmg777k':  ['https://www.dropbox.com/s/fal1nobf1etxtfu/dmg777k.tgz?dl=1'],
    'am':       ['https://www.dropbox.com/s/obo8q7doeg942m9/am.tgz?dl=1'],
    'mutag':    ['https://www.dropbox.com/s/o68n7kclovmbj9f/mutag.tgz?dl=1'],
    'bgs':      ['https://www.dropbox.com/s/54mo6i8nipad2e2/bgs.tgz?dl=1']
}

tics = []
def tic():
    tics.append(time.time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time.time()-tics.pop()

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if type(tensor) == bool:
        return 'cuda'if tensor else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'kgbench' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.dirname(__file__))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), subpath))

def load_rdf(rdf_file, name='', format='nt', store_file='./cache'):
    """
    Load an RDF file into a persistent store, creating the store if necessary.

    If the store exists, return the stored graph.

    :param file:
    :param store_name:
    :return:
    """
    if store_file is None:
        # use in-memory store
        graph = rdf.Graph()
        graph.parse(rdf_file, format=format)
        return graph

    graph = rdf.Graph(store='Sleepycat', identifier=f'kgbench-{name}')
    rt = graph.open(store_file + '-' + name, create=False)

    if rt == rdf.store.NO_STORE:
        print('Persistent store not found. Loading data.')
        rt = graph.open(store_file, create=True)
        graph.parse(rdf_file, format=format)

    else:
        assert rt == rdf.store.VALID_STORE, "The underlying store is corrupt"

        print('Persistent store exists. Loading.')

    return graph


def getfile(container : str, name : str):
    """
    Returns a file object pointing to the specified file. Container can be either a zip file or a directory.

    :param container: Directory or zip file
    :param name:
    :return:
    """

    if os.path.isdir(container):
        with open(os.path.join(container, name)) as file:
            return file

    elif os.path.isfile(container):
        try:
            tar = tarfile.open(container, 'r:gz')
            return tar.extractfile(name)
        except:
            print(f'File {container} does not appear to be a tgz file.')
            raise

# def getfile(dir, name):
#     """
#     Returns a file object pointing to the specified file.

#     :param dir:
#     :param name:
#     :return:
#     """

#     with open(os.path.join(dir, name)) as file:
#         return file




"""
Data loading utilities

TODO:
- create getters and setters for all object members
- rename entities to nodes
- rename datatype to annotation

"""

_XSD_NS = "http://www.w3.org/2001/XMLSchema#"


class Data:
    """
    Class representing a dataset.

    """

    def __init__(self, dir, personalized = False, final=False, use_torch=False, catval=False, name="unnamed_dataset"):

        self.name = name

        self.triples = None
        """ The edges of the knowledge graph (the triples), represented by their integer indices. A (m, 3) numpy 
            or pytorch array.
        """

        self.i2r, self.r2i = None, None

        self.i2e = None
        """ A mapping from an integer index to an entity representation. An entity is either a simple string indicating the label 
            of the entity (a url, blank node or literal), or it is a pair indicating the datatype and the label (in that order).
        """

        self.e2i = None
        """ A dictionary providing the inverse mappring of i2e
        """

        self.num_entities = None
        """ Total number of distinct entities (nodes) in the graph """

        self.num_relations = None
        """ Total number of distinct relation types in the graph """

        self.num_classes = None
        """ Total number of classes in the classification task """

        self.training = None
        """ Training data: a matrix with entity indices in column 0 and class indices in column 1.
            In non-final mode, this is the training part of the train/val/test split. In final mode, the training part, 
            possibly concatenated with the validation data.
        """

        self.withheld = None
        """ Validation/testing data: a matrix with entity indices in column 0 and class indices in column 1.
            In non-final mode this is the validation data. In final mode this is the testing data.
        """

        self._dt_l2g = {}
        self._dt_g2l = {}

        self._datatypes = None
        if dir is not None:

            self.torch = use_torch



            
            

            if personalized==True:
                self.triples = fastload((dir + '/triples_cleaned.int.csv'), personalized=True)
                self.i2r, self.r2i = load_indices((dir + '/relations.int.csv'))
                self.i2e, self.e2i = load_entities((dir + '/nodes.int.csv'))

            else:
                self.triples = fastload(getfile(dir, 'triples.int.csv.gz'))
                self.i2r, self.r2i = load_indices(getfile(dir, 'relations.int.csv'))
                self.i2e, self.e2i = load_entities(getfile(dir, 'nodes.int.csv'))

            self.num_entities  = len(self.i2e)
            self.num_relations = len(self.i2r)

            if personalized:

                train, val, test = \
                    np.loadtxt((dir + '/training.int.csv'),   dtype=np.int32, delimiter=',', skiprows=1), \
                    np.loadtxt((dir +  '/validation.int.csv'), dtype=np.int32, delimiter=',', skiprows=1), \
                    np.loadtxt((dir +  '/testing.int.csv'),    dtype=np.int32, delimiter=',', skiprows=1)
            else:
                train, val, test = \
                np.loadtxt(getfile(dir, 'training.int.csv'),   dtype=np.int32, delimiter=',', skiprows=1), \
                np.loadtxt(getfile(dir, 'validation.int.csv'), dtype=np.int32, delimiter=',', skiprows=1), \
                np.loadtxt(getfile(dir, 'testing.int.csv'),    dtype=np.int32, delimiter=',', skiprows=1)

            if final and catval:
                self.training = np.concatenate([train, val], axis=0)
                self.withheld = test

                if name not in ['aifb', 'mutag', 'bgs', 'am']:
                    warnings.warn('Adding the validation set to the training data. Note that this is not the correct '
                                  'way to load the KGBench data, and will lead to inflated performance. For AIFB, '
                                  'MUTAG, BGS and AM, this is the correct way to load the data.')
            elif final:
                if name in ['aifb', 'mutag', 'bgs', 'am']:
                    warnings.warn('The validation data is not added to the training data. For AIFB, MUTAG, BGS and AM, '
                                  'the correct evaluation is to combine train and validation for the final evaluation run.'
                                  'Set include_val to True when loading the data.')

                self.training = train
                self.withheld = test
            else:
                self.training = train
                self.withheld = val

            self.final = final

            self.num_classes = len(set(self.training[:, 1]))

            # print(f'   {len(self.triples)} triples')

            if use_torch: # this should be constant-time/memory
                self.triples = torch.from_numpy(self.triples)
                self.training = torch.from_numpy(self.training)
                self.withheld = torch.from_numpy(self.withheld)

    def to(self, device):
        # move all tensor attributes to the specified device
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.to(device))
        
        return self

    def get_images(self, dtype='http://kgbench.info/dt#base64Image'):
        """
        Retrieves the entities with the given datatype as PIL image objects.

        :param dtype:
        :return: A list of PIL image objects
        """
        from PIL import Image

        res = []
        # Take in base64 string and return cv image
        num_noparse = 0
        for b64 in self.get_strings(dtype):
            try:
                imgdata = base64.urlsafe_b64decode(b64)
            except:
                print(f'Could not decode b64 string {b64}')
                sys.exit()

            try:
                res.append(Image.open(io.BytesIO(imgdata)))
            except:
                num_noparse += 1
                res.append(Image.new('RGB', (1, 1)))
                # -- If the image can't be parsed, we insert a 1x1 black image

        if num_noparse > 0:
            warnings.warn(f'There were {num_noparse} images that couldn\'t be parsed. These have been replaced by black images.')

        # print(num_noparse, 'unparseable', len([r for r in res if r is not None]), 'parseable')

        return res

    def datatype_g2l(self, dtype, copy=True):
        """
        Returns a list mapping a global index of an entity (the indexing over all nodes) to its _local index_ the indexing
        over all nodes of the given datatype

        :param dtype:
        :param copy:
        :return: A dict d so that `d[global_index] = local_index`
        """
        if dtype not in self._dt_l2g:
            self._dt_l2g[dtype] = [i for i, (label, dt) in enumerate(self.i2e)
                                   if dt == dtype
                                   or (dtype == _XSD_NS+"string"
                                       and dt.startswith('@'))]
            self._dt_g2l[dtype] = {g: l for l, g in enumerate(self._dt_l2g[dtype])}

        return dict(self._dt_g2l[dtype]) if copy else self._dt_g2l[dtype]

    def datatype_l2g(self, dtype, copy=True):
        """
        Maps local to global indices.

        :param dtype:
        :param copy:
        :return: A list l so that `l[local index] = global_index`
        """
        self.datatype_g2l(dtype, copy=False) # init dicts

        return list(self._dt_l2g[dtype]) if copy else self._dt_l2g[dtype]

    def get_strings(self, dtype):
        """
        Retrieves a list of all string representations of a given datatype in order

        :return:
        """
        return [self.i2e[g][0] for g in self.datatype_l2g(dtype)]

    def datatypes(self, i = None):
        """
        :return: If i is None:a list containing all datatypes present in this dataset (including literals without datatype, URIs and
            blank nodes), in canonical order (dee `datatype_key()`).
            If `i` is a nonnegative integer, the i-th element in this list.
        """
        if self._datatypes is None:
            self._datatypes = {dt for _, dt in self.i2e}
            self._datatypes = list(self._datatypes)
            self._datatypes.sort(key=datatype_key)

        if i is None:
            return self._datatypes

        return self._datatypes[i]

    def pyg(self, add_inverse=True):
        """
        Returns a PyG data object KG
        :param add_inverse: If True, adds inverse edges for all edges in the graph, with a associated inverse relation types
                            (same as the default behaviour in PyG for RDF graphs)
        :return: PyG data object
        """
        assert self.torch, 'Data must be loaded with torch=True to generate PyG data object'

        try:
            # Optional import
            from torch_geometric.data import Data as PygData
        except:
            raise Exception('Pytorch geometric does not appear to be installed. Try `pip install pyg` to install it.')

        train_idx, train_y = self.training[:, 0], self.training[:, 1]
        test_idx, test_y = self.withheld[:, 0], self.withheld[:, 1]

        if add_inverse:
            edge_type = torch.hstack((2 * self.triples[:, 1].T, 2 * self.triples[:, 1].T + 1))
            edge_index = torch.hstack((self.triples[:, [0, 2]].T, self.triples[:, [2, 0]].T))
        else:
            edge_type = self.triples[:, 1].T
            edge_index = self.triples[:, [0, 2]].T

        data = PygData(edge_index=edge_index, edge_type=edge_type,
                train_idx=train_idx, train_y=train_y, test_idx=test_idx,
                test_y=test_y, num_nodes=self.num_entities)

        data.num_relations = 2 * self.num_relations if add_inverse else self.num_relations

        return data

    def get_strings_batch(self, idx, dtype):
        """
        Retrieves a list of all string representations of a given datatype in order in a batch by indices as idx

        :param: dtype:
        param: idx:
        :return:
        """
        keys = list(self.e2i.keys())
        key = [keys[i] for i in idx]

        return [key[i][0] for i in range(0, len(key)) if key[i][1] == dtype]

    def get_images_batch(self, idx, dtype='http://kgbench.info/dt#base64Image'):
        """
        Retrieves the entities in batched with the given datatype as PIL image objects and indices as idx.

        :param dtype:
        :param idx:
        :return: A list of PIL image objects
        """
        from PIL import Image
        res = []

        # Take in base64 string and return cv image
        num_noparse = 0
        for b64 in self.get_strings_batch(idx, dtype):
            try:
                imgdata = base64.urlsafe_b64decode(b64)
            except:
                print(f'Could not decode b64 string {b64}')
                sys.exit()
            try:
                res.append(Image.open(io.BytesIO(imgdata)))
            except:
                num_noparse += 1
                res.append(Image.new('RGB', (1, 1)))
                # -- If the image can't be parsed, we insert a 1x1 black image

        if num_noparse > 0:
            warnings.warn(
                f'There were {num_noparse} images that couldn\'t be parsed. These have been replaced by black images.')

        return res

    def dgl(self, training=True, verbose=False, to32=False, safestrings=False):
        """
        Returns a [DGL](http://dgl.ai) data object.

        The returned graph is a HeteroGraph object, with a uniform node type 'resource'. In RDF, type information is
        encoded in the graph with the `rdf:type` relation, not treated as aspecial node property.

        The `label` property extends to all nodes, with the label -1 for unlabeled nodes.

        The train/withheld split is indicated by the 'training_mask' and 'withheld_mask' properties on the nodes. The
        withheld_mask contains either the validation data or the test data, depending on how the original data object is
        loaded.

        :return: A DGL Dataset object representing this KGBench task.
        """
        RES = 'resource' # type for all nodes

        assert self.torch, 'Data must be loaded with torch=True to generate DGL data object'

        # Dynamically import DGL
        try:
            import dgl
            from dgl.data import DGLDataset
            from dgl import DGLGraph
        except:
            raise Exception('DGL does not appear to be installed. Try `pip install dgl` to install it. See http://dgl.ai for more information.')

        # Define a DGL dataset class locally to keep the DGL stuff contained to one function.
        class KGBDataset(DGLDataset):

            def __init__(self, data : Data, training = True):

                super().__init__(name="kgb_" + data.name)

                self.data = data
                self.training = training

                self.predict_category = RES
                self.num_classes = data.num_classes

                # Create heterogeneous graph
                hgdict = {}

                triples = data.triples.to(torch.int32) if to32 else data.triples

                for relid, relname in enumerate(data.i2r):

                    if safestrings:
                        relname = relname.replace('.', '')
                        relname = relname.replace('/', '')

                    # triples with this relation
                    rtriples = triples[triples[:, 1] == relid]
                    subjects, objects = rtriples[:, 0], rtriples[:, 2]

                    hgdict[(RES, relname, RES)] = ('coo', (subjects, objects))

                self.graph = dgl.heterograph(hgdict)

                if verbose:
                    print("Total #nodes:", self.graph.number_of_nodes())
                    print("Total #edges:", self.graph.number_of_edges())

                # Assign nodes to training and withheld
                n_nodes = self.graph.num_nodes()
                training_mask = torch.zeros(n_nodes, dtype=torch.bool)
                withheld_mask = torch.zeros(n_nodes, dtype=torch.bool)
                # -- withheld is either validation or test, depending on how the data is loaded

                training_mask[data.training[:, 0]] = True
                withheld_mask[data.withheld[:, 0]] = True

                self.graph.nodes[RES].data['training_mask'] = training_mask
                self.graph.nodes[RES].data['withheld_mask'] = withheld_mask

                # Assign labels to nodes
                labels = torch.full(fill_value=-1, size=(n_nodes,) , dtype=torch.long)

                labels[self.data.training[:, 0]] = self.data.training[:, 1]
                labels[self.data.withheld[:, 0]] = self.data.withheld[:, 1]

                self.graph.nodes[RES].data['label'] = labels.to(torch.int32) if to32 else labels

            def process(self):
                pass

            def __getitem__(self, i):

                return self.graph

            def __len__(self):
                return 1

        return KGBDataset(data=self, training=training)


SPECIAL = {'iri':'0', 'blank_node':'1', 'none':'2'}
def datatype_key(string):
    """
    A key that defines the canonical ordering for datatypes. The datatypes 'iri', 'blank_node' and 'none' are sorted to the front
    in that order, with any further datatypes following in lexicographic order.

    :param string:
    :return:
    """

    if string in SPECIAL:
        return SPECIAL[string] + string

    return '9' + string

def load(name, final=False, torch=False, prune_dist=None, include_val=False):
    """
    Returns the requested dataset.

    :param name: One of the available datasets
    :param final: Loads the test/train split instead of the validation train split. In this case the training data
    consists of both training and validation.
    :param prune_dist: Removes any nodes in the graph that are further away from any target node than this value. This is
    helpful for models like RGCNs that can only see a limited distance from the target nodes. It saves memory to prune
    the part of the graph that doesn't affect the predictions.
    :param torch: Load the dataset as pytorch tensors rather than numpy tensors.
    :param include_val: If `final == True`, this adds the validation data to the training data. this is not the correct
    way to load the kgbench datasets, but it is correct for older datasets like AIFB, MUTAG, BGS and AM.
    :return: A pair (triples, meta). `triples` is a numpy 2d array of datatype uint32 contianing integer-encoded
    triples. `meta` is an object of metadata containing the following fields:
     * e: The number of entities
     * r: The number of relations
     * i2r:
    """

    if name == 'micro':
        return micro(final, torch)
        # -- a miniature dataset for unit testing

    if name in URLS.keys():

        # ensure data dir exists
        Path(here('../datasets/')).mkdir(parents=True, exist_ok=True)

        # Download the data if necessary
        if not os.path.exists(here(f'../datasets/{name}.tgz')):
            print(f'Downloading {name} dataset.')

            tries = 0
            success = False
            while not success:

                try:
                    url = random.choice(URLS[name])
                    wget.download(url, out=here('../datasets/'))
                    success = True
                except Exception as e:
                    success = False
                    tries += 1
                    if tries > 10:
                        raise


        tic()
        data = Data(here(f'../datasets/{name}.tgz'), final=final, use_torch=torch, name=name, catval=include_val)
        print(f'loaded data {name} ({toc():.4}s).')

    elif os.path.isfile(name):

        tic()
        data = Data(here(name), final=final, use_torch=torch, name=name, catval=include_val)
        print(f'loaded data {name} ({toc():.4}s).')

    else:
        raise Exception(f'Argument {name} does refer to one of the included datasets and does not seem to be a file on the filesystem.')

    if prune_dist is not None:
        tic()
        data = prune(data, n=prune_dist)
        print(f'pruned ({toc():.4}s).')

    return data

def micro(final=True, use_torch=False):
    """
    Micro dataset for unit testing.

    :return:
    """

    data = Data(None)

    data.num_entities = 5
    data.num_relations = 2
    data.num_classes = 2

    data.i2e = [(str(i), 'none') for i in range(data.num_entities)]
    data.i2r = [str(i) for i in range(data.num_entities)]

    data.e2i = {e:i for i, e in enumerate(data.i2e)}
    data.r2i = {r:i for i, r in enumerate(data.i2e)}

    data.final = final
    data.triples = np.asarray(
        [[0, 0, 1], [1, 0, 2], [0, 0, 2], [2, 1, 3], [4, 1, 3], [4, 1, 0] ],
        dtype=np.int32
    )

    data.training = np.asarray(
        [[1, 0], [2, 0]],
        dtype=np.int32
    )

    data.withheld = np.asarray(
        [[3, 1], [3, 1]],
        dtype=np.int32
    )

    data.torch = use_torch
    if torch: # this should be constant-time/memory
        data.triples  = torch.from_numpy(data.triples)
        data.training = torch.from_numpy(data.training)
        data.withheld = torch.from_numpy(data.withheld)

    data.name="micro"

    return data

def load_indices(file):

    df = pd.read_csv(file, na_values=[], keep_default_na=False)

    assert len(df.columns) == 2, 'CSV file should have two columns (index and label)'
    assert not df.isnull().any().any(), f'CSV file {file} has missing values'

    idxs = df['index'].tolist()
    labels = df['label'].tolist()

    i2l = list(zip(idxs, labels))
    i2l.sort(key=lambda p: p[0])
    for i, (j, _) in enumerate(i2l):
        assert i == j, f'Indices in {file} are not contiguous'

    i2l = [l for i, l in i2l]

    l2i = {l:i for i, l in enumerate(i2l)}

    return i2l, l2i

def load_entities(file):

    df = pd.read_csv(file, na_values=[], keep_default_na=False)

    if df.isnull().any().any():
        lines = df.isnull().any(axis=1)
        print(df[lines])
        raise Exception('CSV has missing values.')

    assert len(df.columns) == 3, 'Entity file should have three columns (index, datatype and label)'
    assert not df.isnull().any().any(), f'CSV file {file} has missing values'

    idxs =   df['index']      .tolist()
    dtypes = df['annotation'] .tolist()
    labels = df['label']      .tolist()

    ents = zip(labels, dtypes)

    i2e = list(zip(idxs, ents))
    i2e.sort(key=lambda p: p[0])
    for i, (j, _) in enumerate(i2e):
        assert i == j, 'Indices in entities.int.csv are not contiguous'

    i2e = [l for i, l in i2e]

    e2i = {e: i for e in enumerate(i2e)}

    return i2e, e2i

def prune(data : Data, n=2):
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

    nw = Data(dir=None)

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

def group(data : Data):
    """
    Groups the dataset by datatype. That is, reorders the nodes so that all nodes of data.datatypes(0) come first,
    followed by the nodes of datatype(1), and so on.

    The datatypes 'iri', 'blank_node' and 'none' are guaranteed to be sorted to the front in that order.

    :param data:
    :return: A new Data object, not backed by the old.
    """

    data_triples = data.triples
    data_training = data.training
    data_withheld = data.withheld

    if data.torch:
        data_triples = data_triples.numpy()
        data_training = data_training.numpy()
        data_withheld = data_withheld.numpy()

    # new index to old index
    n2o = []
    for datatype in data.datatypes():
        n2o.extend(data.datatype_l2g(datatype))

    assert set(n2o) == set(range(len(n2o)))

    o2n = {o: n for n, o in enumerate(n2o)}

    # create the mapped data object
    nw = Data(dir=None)

    nw.num_entities = len(n2o)
    nw.num_relations = data.num_relations

    nw.i2e = [data.i2e[o] for o in n2o]
    nw.e2i = {e: i for i, e in enumerate(nw.i2e)}

    # relations are unchanged but copied for the sake of GC
    nw.i2r = list(data.i2r)
    nw.r2i = dict(data.r2i)

    nw.triples = np.zeros(data.triples.shape, dtype=int)

    row = 0
    for s, p, o in data_triples:
        s, o = o2n[s], o2n[o]
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

# def fastload(file):
#     """
#     Quickly load an (m, 3) matrix of integer triples
#     :param input:
#     :return:
#     """
#
#     # count the number of lines
#     with gzip.open(file, 'rt') as input:
#         lines = 0
#         for line in input:
#             print(line)
#             lines += 1
#
#     # prepare a zero metrix
#     result = np.zeros((lines, 3), dtype=np.int)
#
#     # fill the zero matrix with the values from the file
#     with gzip.open(file, 'rt') as input:
#         i = 0
#         for i, line in enumerate(input):
#
#             print(i, line)
#
#             s, p, o = str(line).split(',')
#             s, p, o = int(s), int(p), int(o)
#             result[i, :] = (s, p, o)
#
#             i += 1
#
#     return result

def fastload(file, personalized = False):
    """
    Quickly load an (m, 3) matrix of integer triples
    :param input:
    :return:
    """
    if personalized:
        triples = []
        with open(file, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f)
            for s, p, o in reader:
                triples.append((int(s.encode('utf-8').decode('utf-8')), 
                                int(p.encode('utf-8').decode('utf-8')), 
                                int(o.encode('utf-8').decode('utf-8'))))
        return triples

    triples = []

    with gzip.open(file, 'rt') as input:
        for line in input:

            s, p, o = str(line).split(',')
            s, p, o = s.encode('utf-8'), p.encode('utf-8'), o.encode('utf-8')
            s, p, o = s.decode('utf-8'), p.decode('utf-8'), o.decode('utf-8')
            s, p, o = int(s), int(p), int(o)
            triples.append( (s, p, o) )



    return np.asarray(triples)


