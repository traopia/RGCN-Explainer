import argparse
import os
import pickle
import random
import shutil
import time

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import configs
import utils.math_utils as math_utils
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
import utils.featgen as featgen
import utils.graph_utils as graph_utils

import models

def train(
    dataset,
    model,
    args,
    same_feat=True,
    val_dataset=None,
    test_dataset=None,
    writer=None,
    mask_nodes=True,
):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    #for epoch in range(args.num_epochs):
    for epoch in range(100):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            if batch_idx == 0:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = prev_adjs
                all_feats = prev_feats
                all_labels = prev_labels
            elif batch_idx < 20:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
                all_feats = torch.cat((all_feats, prev_feats), dim=0)
                all_labels = torch.cat((all_labels, prev_labels), dim=0)
            adj = Variable(data["adj"].float(), requires_grad=False).cuda()
            h0 = Variable(data["feats"].float(), requires_grad=False).cuda()
            label = Variable(data["label"].long()).cuda()
            batch_num_nodes = data["num_nodes"].int().numpy() if mask_nodes else None
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            ).cuda()

            ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if batch_idx < 5:
                predictions += ypred.cpu().detach().numpy().tolist()

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar("loss/linkpred_loss", model.link_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate(dataset, model, args, name="Train", max_num_examples=100)
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])

    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    plt.figure()
    plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
        plt.legend(["train", "val", "test"])
    else:
        plt.plot(best_val_epochs, best_val_accs, "bo")
        plt.legend(["train", "val"])
    plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")

    #print(all_adjs.shape, all_feats.shape, all_labels.shape)

    cg_data = {
        "adj": all_adjs,
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset))),
    }
    #io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    torch.save(
        {
            "epoch": 100,
            "model_type": args.method,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cg": cg_data,
        },
        'cora_100.pth',
    )
    return model, val_accs





def train_node_classifier(data,  model, args, writer=None):
    # train/test split only for nodes
    #num_nodes = G.number_of_nodes()
    num_nodes = data.num_nodes
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = preprocess_input_graph(data)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)
    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        #nn.utils.clip_grad_norm(model.parameters(), args.clip)
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    # computation graph
    model.eval()
    if args.gpu:
        ypred, _ = model(x.cuda(), adj.cuda())
    else:
        ypred, _ = model(x, adj)
    cg_data = {
        "adj": data["adj"],
        "feat": data["feat"],
        "label": data["labels"],
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }
    # import pdb
    # pdb.set_trace()
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    torch.save(
        {
            "epoch": 100,
            "model_type": args.method,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cg": cg_data,
        },
        'cora_100.pth',
    )


def evaluate(dataset, model, args, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False).cuda()
        h0 = Variable(data["feats"].float()).cuda()
        labels.append(data["label"].long().numpy())
        batch_num_nodes = data["num_nodes"].int().numpy()
        assign_input = Variable(
            data["assign_feats"].float(), requires_grad=False
        ).cuda()

        ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test

def arg_parse():
    parser = argparse.ArgumentParser(description="GraphPool arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument(
        "--assign-ratio",
        dest="assign_ratio",
        type=float,
        help="ratio of number of nodes in consecutive layers",
    )
    softpool_parser.add_argument(
        "--num-pool", dest="num_pool", type=int, help="number of pooling layers"
    )
    parser.add_argument(
        "--linkpred",
        dest="linkpred",
        action="store_const",
        const=True,
        default=False,
        help="Whether link prediction side objective is used",
    )

    parser_utils.parse_optimizer(parser)

    parser.add_argument(
        "--datadir", dest="datadir", help="Directory where benchmark is located"
    )
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--max-nodes",
        dest="max_nodes",
        type=int,
        help="Maximum number of nodes (ignore graghs with nodes exceeding the number.",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        type=float,
        help="Ratio of number of graphs training set to all graphs.",
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )
    parser.add_argument(
        "--input-dim", dest="input_dim", type=int, help="Input feature dimension"
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-classes", dest="num_classes", type=int, help="Number of label classes"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        help="Weight decay regularization constant.",
    )

    parser.add_argument(
        "--method", dest="method", help="Method. Possible values: base, "
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )

    parser.set_defaults(
        datadir=None, #"data",  # io_parser
        logdir="log",
        ckptdir="ckpt_cora",
        dataset= None,
        opt="adam",  # opt_parser
        opt_scheduler="none",
        max_nodes=100,
        cuda="1",
        feature_type="default",
        lr=0.001,
        clip=2.0,
        batch_size=20,
        num_epochs=100,
        train_ratio=0.8,
        test_ratio=0.1,
        num_workers=1,
        input_dim= 1433, #10,
        hidden_dim=20,
        output_dim=7, #20,
        num_classes=2,
        num_gc_layers=3,
        dropout=0.0,
        weight_decay=0.005,
        method="base",
        name_suffix="",
        assign_ratio=0.1,
    )
    return parser.parse_args()

def task(dataa, args, writer=None):
    # data
    # G, labels, name = gengraph.gen_syn1(
    #     feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    # )
    
    #num_classes = max(labels) + 1
    #print(dataa)
    num_classes = len(np.unique(dataa.y))


    print("Method:", args.method)
    model = models.GcnEncoderNode(
        1433, #args.input_dim,
        args.hidden_dim,
        7,#args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,
    )
    if args.gpu:
        model = model.cuda()

    train_node_classifier(dataa, model, args, writer=writer)

def get_adjacency(data):
    adj = torch.zeros(data.num_nodes, data.num_nodes)
    for edge in data.edge_index.t():
        adj[edge[0]][edge[1]] = 1
    return adj


def preprocess_input_graph(data,normalize_adj=False):
    """ Load an existing graph to be converted for the experiments.
    Args:
        G: Networkx graph to be loaded.
        labels: Associated node labels.
        normalize_adj: Should the method return a normalized adjacency matrix.
    Returns:
        A dictionary containing adjacency, node features and labels
    """
    #adj = np.array(nx.convert_matrix.to_numpy_array(G))

    adj = get_adjacency(data)
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    #existing_node = list(G.nodes)[-1]
    #feat_dim = dataset.num_features #G.nodes[existing_node]["feat"].shape[0]
    f = data.x
    labels = data.y
    #np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    # for i, u in enumerate(G.nodes()):
    #     f[i, :] = G.nodes[u]["feat"]

    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.expand_dims(labels, axis=0)
    return {"adj": adj, "feat": f, "labels": labels}    

def main():


    datasett = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    print(f'Number of graphs: {len(datasett)}')
    print(f'Number of features: {datasett.num_features}')
    print(f'Number of classes: {datasett.num_classes}')
    dataa = datasett[0]
    print('dataset under study')
    print(f'Number of nodes: {dataa.num_nodes}')
    print(f'Number of edges: {dataa.num_edges}')
    print('Node features:', dataa.x)
    print('Node labels:', dataa.y)
    prog_args = configs.arg_parse()
    print('args:',prog_args)

    path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
    writer = SummaryWriter(path)

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    task(dataa,prog_args, writer=writer)    

if __name__ == "__main__":   
    main() 