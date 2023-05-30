


#match triples 
# def match_to_triples(tensor1, tensor2,data, sparse=True):
#     """
#     tensor1: sub_edge tensor: edges of the neighborhood - transpose!!
#     tensor2: data.triples: all edges
#     """
#     matching = []
#     for i,i2 in zip(tensor1[:,0],tensor1[:,1]):
#         for j,j1,j2, index in zip(tensor2[:,0],tensor2[:,1],  tensor2[:,2], range(len(tensor2[:,0]))):
#             if i == j and i2 == j2:
#                 matching.append(tensor2[index])
                

#     result = torch.stack(matching)
#     print(result)
#     return result

# def match_to_triples(v, data):
#     p,_ = v.coalesce().indices()//data.num_entities
#     s,o = v.coalesce().indices()%data.num_entities
#     result = torch.stack([s,p,o], dim=1)
#     return result



# def match_to_triples(v, data, sparse=True):
#     if sparse:
#         # p,_ = torch.div(v.coalesce().indices(), data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
#         # s,o = v.coalesce().indices()%data.num_entities
#         # result = torch.stack([s,p,o], dim=1)
#         matching = []
#         indexes = v.coalesce().indices()%data.num_entities
#         for j in range(indexes.size()[1]):
#             for triple in data.triples:
#                 if triple[0] == indexes[0][j] and triple[2] == indexes[1][j]:
#                     matching.append(triple)
#         result = torch.stack(matching)

                    
#     else:
#         matching = []
#         for i,i2 in zip(v[:,0],v[:,1]):
#             for j,j1,j2, index in zip(data[:,0],data[:,1],  data[:,2], range(len(data[:,0]))):
#                 if i == j and i2 == j2:
#                     matching.append(data[index])
                    

#         result = torch.stack(matching)
    
#     return result


# def match_to_triples(v,h, data, sparse=True):
#     if sparse:
#         pv,_ = torch.div(v.coalesce().indices(), data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
#         sv,ov = v.coalesce().indices()%data.num_entities
#         result_v = torch.stack([sv,pv,ov], dim=1)
#         ph,_ = torch.div(h.coalesce().indices(), data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
#         sh,oh = h.coalesce().indices()%data.num_entities
#         result_h = torch.stack([sh,ph,oh], dim=1)
#         result = torch.cat((result_v, result_h), 0)


                    
#     else:
#         _,ph = torch.div(h, data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
#         sh,oh = h%data.num_entities
#         result_h = torch.stack([sh,ph,oh], dim=1)
#         pv, _ = torch.div(v, data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
#         sv,ov = v%data.num_entities
#         result_v = torch.stack([sv,pv,ov], dim=1)
#         result = torch.cat((result_v, result_h), 0)
#         print(pv, ph)

                    
    
#     return result



# def visualize(node_idx, n_hop, data, masked_ver,threshold,name, result_weights=True, low_threshold=False ):
#     """ 
#     Visualize important nodes for node idx prediction
#     """
#     dict_index = dict_index_classes(data,masked_ver)
    
#     #select only nodes with a certain threshold
#     sel_masked_ver = sub_sparse_tensor(masked_ver, threshold,data, low_threshold)
#     if len(sel_masked_ver)==0:
#         sel_masked_ver=sub_sparse_tensor(masked_ver, 0,data, low_threshold)
#     print('sel masked ver',sel_masked_ver)
#     indices_nodes = sel_masked_ver.coalesce().indices().detach().numpy()
#     new_index = np.transpose(np.stack((indices_nodes[0], indices_nodes[1]))) #original edge indexes

    
    
#     G = nx.Graph()
#     if result_weights:
#         values = sel_masked_ver.coalesce().values().detach().numpy()
#         for s,p,o in zip(indices_nodes[0],values , indices_nodes[1]):
#             G.add_edge(int(s), int(o), weight=np.round(p, 2))

#     else:
#         #get triples to get relations 
#         #triples_matched = match_to_triples(np.array(new_index), data.triples)
#         triples_matched = match_to_triples(sel_masked_ver, data)
#         l = []
#         for i in triples_matched[:,1]:
#             l.append(data.i2rel[int(i)][0])
#         print(Counter(l))
#         for s,p,o in triples_matched:
#             G.add_edge(int(s), int(o), weight=int(p))

#     edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())


#     pos = nx.circular_layout(G)

#     ordered_dict = {}
#     for item in list(G.nodes):
#         if item in ordered_dict:
#             ordered_dict[item].append(dict_index[item])
#         # else:
#         #     ordered_dict[item] =  dict_index[item]

#     dict_index = ordered_dict

#     labeldict = {}
#     for node in G.nodes:
#         labeldict[int(node)] = int(node)  

#     print('dict index:', dict_index)

#     dict = {}
#     for k,v in dict_index.items():
#         for k1,v1 in data.entities_classes.items():
#             if v==k1: 

#                 dict[k] = v1
#             else:
#                 if k not in dict:
#                     dict[k] = 0
                

#     color_list = list(dict.values())
#     color_list = list(encode_dict(dict_index).values())


    
#     if result_weights:
        
#         nx.draw(G, pos,labels = labeldict,  edgelist=edges, edge_color=weights, node_color =  color_list, cmap="Set2",edge_cmap=plt.cm.Reds,font_size=8)
#         nx.draw_networkx_edge_labels( G, pos,edge_labels=nx.get_edge_attributes(G,'weight'),font_size=8,font_color='red')
#         # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
#         # sm.set_array(weights)
#         # cbar = plt.colorbar(sm)
#         # cbar.ax.set_title('Weight')
#         plt.title("Node {}'s {}-hop neighborhood important nodes".format(node_idx, n_hop))
#     else:
#         rel = nx.get_edge_attributes(G,'weight')
#         for k,v in rel.items():
#             rel[k] = data.i2rel[v][0]
#         nx.draw(G, pos,labels = labeldict,  edgelist=edges, edge_color=weights,node_color =  color_list, cmap="Set2",font_size=8)
#         nx.draw_networkx_edge_labels( G, pos,edge_labels=rel,font_size=8,font_color='red')
#         res = Counter(rel.values())
#     if result_weights:
#         if not os.path.exists(f'chk/{name}_chk/graphs'):
#             os.makedirs(f'chk/{name}_chk/graphs')  
#         plt.savefig(f'chk/{name}_chk/graphs/Explanation_{node_idx}_{n_hop}_weights.png')
#         #plt.show()
#     else:
#         if not os.path.exists(f'chk/{name}_chk/graphs'):
#             os.makedirs(f'chk/{name}_chk/graphs')  
#         plt.savefig(f'chk/{name}_chk/graphs/Explanation_{node_idx}_{n_hop}_relations.png')    
#         #plt.show()
#         return res





# def visualize(node_idx, n_hop, data, masked_ver,masked_hor, threshold,name, result_weights=True, low_threshold=False,experiment_name=None ):
#     """ 
#     Visualize important nodes for node idx prediction
#     """
#     dict_index = dict_index_classes(data,masked_ver)
#     mask = torch.vstack((masked_ver, masked_hor.t()))
#     mask = sub(mask, threshold)
#     print(mask)
#     #select only nodes with a certain threshold
#     sel_masked_ver = sub(masked_ver, threshold)
#     sel_masked_hor = sub(masked_hor, threshold)
#     if len(sel_masked_ver)==0:
#         sel_masked_ver=sub_sparse_tensor(masked_ver, 0,data, low_threshold)
#     #mask = torch.vstack((sel_masked_ver, sel_masked_hor.t()))
#     print('sel masked ver',mask)
#     indices_nodes = mask.coalesce().indices().detach().numpy()
#     new_index = np.transpose(np.stack((indices_nodes[0], indices_nodes[1]))) #original edge indexes

    
    
#     G = nx.Graph()
#     if result_weights:
#         values = mask.coalesce().values().detach().numpy()
#         for s,p,o in zip(indices_nodes[0],values , indices_nodes[1]):
#             G.add_edge(int(s), int(o), weight=np.round(p, 2))

#     else:

#         triples_matched = match_to_triples(sel_masked_ver,sel_masked_hor, data)
#         l = []
#         for i in triples_matched[:,1]:
#             l.append(data.i2rel[int(i)][0])
#         triples_matched = find_repeating_sublists(triples_matched.numpy())
#         #print(triples_matched)
#         for s,p,o in triples_matched:
#             G.add_edge(int(s), int(o), weight=p)


#     edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    
#     weights = [[item] if not isinstance(item, list) else item for item in weights]


#     pos = nx.circular_layout(G)

#     ordered_dict = {}
#     for item in list(G.nodes):
#         if item in ordered_dict:
#             ordered_dict[item].append(dict_index[item])
#         # else:
#         #     ordered_dict[item] =  dict_index[item]

#     dict_index = ordered_dict

#     labeldict = {}
#     for node in G.nodes:
#         labeldict[int(node)] = int(node)  


#     dict = {}
#     for k,v in dict_index.items():
#         for k1,v1 in data.entities_classes.items():
#             if v==k1: 

#                 dict[k] = v1
#             else:
#                 if k not in dict:
#                     dict[k] = 0
                

#     color_list = list(dict.values())
#     color_list = list(encode_dict(dict_index).values())


#     col_weights = [weights[i][0] for i in range(len(weights))]
#     if result_weights:
        
#         nx.draw(G, pos,labels = labeldict,  edgelist=edges, edge_color=col_weights, node_color =  color_list, cmap="Set2",edge_cmap=plt.cm.Reds,font_size=8)
#         nx.draw_networkx_edge_labels( G, pos,edge_labels=nx.get_edge_attributes(G,'weight'),font_size=8,font_color='red')
#         sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
#         sm.set_array(weights)
#         cbar = plt.colorbar(sm)
#         cbar.ax.set_title('Weight')
#         plt.title("Node {}'s {}-hop neighborhood important nodes".format(node_idx, n_hop))
#     else:
#         rel = nx.get_edge_attributes(G,'weight')
#         rel = {k: [data.i2rel[i][0] for i in v] for k,v in rel.items()}
#         col_weights = [sum(weights[i], 3) if len(weights[i]) > 1 else weights[i][0] for i in range(len(weights))]
#         nx.draw(G, pos,labels = labeldict, edge_color=col_weights,edgelist=edges,node_color =  color_list, cmap="Set2",font_size=7, arrows = True)
#         nx.draw_networkx_edge_labels( G, pos,edge_labels=rel,font_size=8,font_color='red')
        
#         res = Counter(unnest_list(rel.values()))
#         print(res)
#     if result_weights:
#         if not os.path.exists(f'chk/{name}_chk/{experiment_name}⁄graphs'):
#             os.makedirs(f'chk/{name}_chk/{experiment_name}⁄graphs')  
#         plt.savefig(f'chk/{name}_chk/{experiment_name}⁄graphs/Explanation_{node_idx}_weights.png')

#         #plt.show()

#     else:
#         if not os.path.exists(f'chk/{name}_chk/{experiment_name}⁄graphs'):
#             os.makedirs(f'chk/{name}_chk/{experiment_name}⁄graphs')  
#         plt.savefig(f'chk/{name}_chk/{experiment_name}⁄graphs/Explanation_{node_idx}_relations.png')    
#         #plt.show()
#         return res#, weights  




#Visualize the result
def visualize_result(node_idx, masked_ver, neighbors, data, num_hops):
    """Visualizes the n-hop neighborhood of a given node."""
    indices_nodes = masked_ver.coalesce().indices().detach().numpy()
    new_ver = indices_nodes[0]%data.num_entities #get original indexes
    new_index = np.transpose(np.stack((new_ver, indices_nodes[1]))) #original edge indexes
    G = nx.from_edgelist(new_index)
    
    #create dict of index - node: to visualize index of the node
    labeldict = {}
    for node in G.nodes:
        labeldict[node] = node 
    #print(G.nodes)
    dict_index = dict_index_classes(data,masked_ver)
    #order dict index according to G nodes in networkx
    ordered_dict = {}
    for item in list(G.nodes):
        ordered_dict[item] = dict_index[item]

    dict_index = ordered_dict
    

    #get inverse of dict to allow mapping of different 'classes' of nodes to different nodes
    inv_map = {v: k for k, v in dict_index.items()}
    print(inv_map) #use inv:map to get a legend of node colors for later 
    color_list = list(dict_index.values())
    
    #make a list out of it 
    for i in range(len(color_list)):
        if color_list[i] in inv_map:
            color_list[i] = inv_map[color_list[i]]   

            
    #edge colors reflect the masked ver values - more important relations have darker color
    #to check why we have relations than expexted :')
    edge_colors = list(masked_ver.coalesce().values().detach().numpy())[:int(G.number_of_edges())]

    # draw graph with edge colors
    plt.figure()  
    plt.title("Node {}'s {}-hop neighborhood important nodes".format(node_idx, num_hops))
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True, edge_color = edge_colors, edge_cmap=plt.cm.Reds,node_color =  color_list  ,labels = labeldict, cmap="Set2" )

    #add colorbar legend


    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array(edge_colors)
    cbar = plt.colorbar(sm)
    cbar.ax.set_title('Weight')

    plt.show()  




#MAIN
# 
def main(name,node_idx, prune=True, explain_all = False, train=False):
    if explain_all == True:
        exp = 'all'
    else:
        exp = node_idx
    n_hops = 0 if prune else 2
    #n_hops = 2


    if name in ['aifb', 'mutag', 'bgs', 'am', 'mdgenre']:
        data = kg.load(name, torch=True, final=False)
    else:    
    #data = kg.load(name, torch=True)  
        data = torch.load(f'data/IMDB/finals/{name}.pt')
    if prune:
        data = prunee(data, 2)
        data.triples = torch.Tensor(data.triples).to(int)#data.triples.clone().detach()
        data.withheld = torch.Tensor(data.withheld).to(int)#data.withheld.clone().detach()
        data.training = torch.Tensor(data.training).to(int)#data.training.clone().detach()

          
    print(f'Number of entities: {data.num_entities}') #data.i2e
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') #data.i2r
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    get_relations(data)
    d = d_classes(data)




    # edge_index = edge_index_oneadj(data.triples)
    # _,neighbors,sub_edge = find_n_hop_neighbors(edge_index, n_hops, node_idx)
    # num_neighbors = len(sub_edge.t())
    # print('num_neighbors', num_neighbors)


    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    edge_index_h, edge_index_v = hor_graph.coalesce().indices(), ver_graph.coalesce().indices()
    sub_edges, neighbors_h, sub_edges_tensor_h  = find_n_hop_neighbors(edge_index_h, n_hops, node_idx)
    sub_edges, neighbors_v, sub_edges_tensor_v  = find_n_hop_neighbors(edge_index_v, n_hops, node_idx)
    num_neighbors = len(list(neighbors_h) + list(neighbors_v))
    print('num_neighbors', num_neighbors)




    sweep_config = {
        'method': 'grid',
        'parameters': {
            'lr': {'values': [0.1, 0.5]},
            'size': {'values': [0.005]},
            'ent': {'values': [10]},
            'size_std': {'values': [10]},
            'size_num': {'values': [ 0.1]},
            'epochs': {'values': [2]},
            'init_strategy': {'values': ['normal', 'const', 'overall_frequency', 'relative_frequency', 'inverse_relative_frequency', 'domain_frequency', 'range_frequency', 'owl']},
            'threshold': {'values': [0.5]},
        }
    }
    #sweep_id = wandb.sweep(sweep_config, project='RGCNExplainer_AIFB_5757')
    params={
                "pred": 1,
                "size":  0.005,#0.005,#0.005, #0.005,  
                "size_std": 10, #num_neighbors*0.1,#-10,
                "ent": 1,
                "size_num": 0.001,
                "lr": 0.1,
                "epochs": 30,
                "init_strategy": "normal", #[],
                "threshold": 0.5,
                "experiment": f"RGCNExplainer_AIFB_{exp}",
            }

    wandb.login()
    wandb.init(project=params['experiment'], config=params)
    
    config = wandb.config
    print(config)
    experiment_name = f'hops_{n_hops}_size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}'
    model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
    high = []
    high_floats = []
    low = []
    relations = [data.i2rel[i][0] for i in range(len(data.i2rel))]
    model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
    relations = ['label', 'node_idx','number_neighbors', 'prediction_explain', 'prediction_full', 'prediction_explain_binary'] + relations
    df = pd.DataFrame(columns=relations)
    df_floats = pd.DataFrame(columns=relations)
    if explain_all == True:
        for target_label in range(len(d.keys())):
            for node_idx in d[target_label]:
                print('node_idx', node_idx)
                _,neighbors,sub_edge = find_n_hop_neighbors(edge_index, n_hops, node_idx)
                num_neighbors = len(sub_edge.t())
                if num_neighbors > 100:
                    config.update({'size_std': num_neighbors*0.1}, allow_val_change=True)
                    config.update({'ent': num_neighbors*0.01}, allow_val_change=True)
                if num_neighbors < 50:
                    config.update({'size_std': 1}, allow_val_change=True)
                    config.update({'ent': 1}, allow_val_change=True)
                

                print('config size std:',config['size_std'])
                print('config ent:',config['ent'])
                
                explainer = Explainer(model, data,name,  node_idx, n_hops, prune,config)
                masked_hor, masked_ver = explainer.explain()
                h_0 ,v_0= select_on_relation_sparse(masked_hor,data, 34), select_on_relation_sparse(masked_ver,data, 34)
                h_0 ,v_0= select_on_relation_sparse(masked_hor,data, 38), select_on_relation_sparse(masked_ver,data, 38)
                h_0 ,v_0= select_on_relation_sparse(masked_hor,data, 39), select_on_relation_sparse(masked_ver,data, 39)
                #masked_hor, masked_ver = h_0, v_0

                #wandb.agent(sweep_id,  explainer.explain)
                # if not os.path.exists(f'/chk/{name}_chk/{experiment_name}_/masked_adj'):
                #         os.makedirs(f'chk/{name}_chk/{experiment_name}_/masked_adj') 
                # else:
                #     print('directory exists')
                directory = f'chk/{name}_chk/{experiment_name}/masked_adj'

                if not os.path.exists(directory):
                    os.makedirs(directory)
                else:
                    print(f"Directory '{directory}' already exists.")
                torch.save(masked_ver, f'chk/{name}_chk/{experiment_name}/masked_adj/masked_ver{node_idx}')
                torch.save(masked_hor, f'chk/{name}_chk/{experiment_name}/masked_adj/masked_hor{node_idx}') 
                #h = visualize(node_idx, n_hops, data, masked_ver,masked_hor,threshold=config['threshold'], name = name, result_weights=False, low_threshold=False)
                h = selected(masked_ver,masked_hor,  threshold=config['threshold'],data=data, low_threshold=False)
                res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
                masked_ver,masked_hor = convert_binary(masked_ver, config['threshold']), convert_binary(masked_hor, config['threshold'])
                res_binary = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
                

                hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
                y_full = model.forward2(hor_graph, ver_graph)
                node_pred_full = y_full[node_idx, :]
                res_full = nn.Softmax(dim=0)(node_pred_full)

                #high.append(h)
                h = dict(h)
                info = {'label': target_label, 'node_idx': str(node_idx),'number_neighbors': num_neighbors, 'prediction_explain': str(res.detach().numpy()), 'prediction_full': str(res_full.detach().numpy()), 'prediction_explain_binary': str(res_binary.detach().numpy())}
                h.update(info)
                print('info:',h)

                df.loc[str(node_idx)] = h




                # h_floats = selected(masked_ver, threshold=0.5,data=data, low_threshold=False,float=True)
                # high_floats.append(h_floats)
                # h_floats = dict(h_floats)
                # h_floats.update(info)
                # df_floats.loc[str(node_idx)] = h_floats
                #experiment_name = f'size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}'
                if not os.path.exists(f'Relation_Importance_{name}/{experiment_name}'):
                    os.makedirs(f'Relation_Importance_{name}/{experiment_name}')
                df.to_csv(f'Relation_Importance_{name}/{experiment_name}/Relations_Important_{name}_{node_idx}.csv', index=False)

                print('node_idx', node_idx, 
                    '\n node original label',[k for k, v in d.items() if node_idx in v],
                    '\n node predicted label explain', torch.argmax(res).item(), 'explain binary', torch.argmax(res_binary).item(),
                    '\n node prediction probability explain', res,
                        '\n node predicted label full', torch.argmax(res_full).item(),
                        'most important relations ', h,
                        '\n final masks and lenght', masked_ver, len(masked_ver.coalesce().values()[masked_ver.coalesce().values()>config['threshold'] ]),
                        '\n ---------------------------------------------------------------')
        
        if not os.path.exists(f'Relation_Importance_{name}/{experiment_name}'):
            os.makedirs(f'Relation_Importance_{name}/{experiment_name}')
        df.to_csv(f'Relation_Importance_{name}/{experiment_name}/Relations_Important_all_{name}.csv', index=False) 
        #df_floats.to_csv(f'Relation_Importance_{name}/Relations_Important_all_{name}_{node_idx}_floats.csv', index=False) 
                
    else:
        if name != 'aifb':
            node_idx = d[0][0]
        if train:
            # if num_neighbors > 100:

            #     config.update({'size_std': num_neighbors*0.1}, allow_val_change=True)
            #     config.update({'ent': 1}, allow_val_change=True)
            # if num_neighbors < 50:
            #     config.update({'size_std': 1}, allow_val_change=True)
            #     config.update({'ent': 1}, allow_val_change=True)
            # if num_neighbors > 50 and num_neighbors < 100:
            #     config.update({'size_std': 10}, allow_val_change=True)
            #     config.update({'ent': 10}, allow_val_change=True)


            #config.update({'size_std': num_neighbors}, allow_val_change=True)
            print('config size std:',config['size_std'])

            print('config size:',config['size'])
            explainer = Explainer(model, data,name,  node_idx, n_hops, prune, config)
            masked_hor, masked_ver = explainer.explain()



            #wandb.agent(sweep_id, explainer.explain)
            directory = f'chk/{name}_chk/{experiment_name}/masked_adj'

            if not os.path.exists(directory):
                os.makedirs(directory)
            else:
                print(f"Directory '{directory}' already exists.")
            torch.save(masked_ver, f'chk/{name}_chk/{experiment_name}/masked_adj/masked_ver{node_idx}')
            torch.save(masked_hor, f'chk/{name}_chk/{experiment_name}/masked_adj/masked_hor{node_idx}') 
        else:
            masked_ver = torch.load(f'chk/{name}_chk/{experiment_name}/masked_adj/masked_ver{node_idx}')
            masked_hor = torch.load(f'chk/{name}_chk/{experiment_name}/masked_adj/masked_ver{node_idx}')
        #h = visualize(node_idx, n_hops, data, masked_ver,masked_hor, threshold=config['threshold'] , name = name, result_weights=False, low_threshold=False, experiment_name=experiment_name)
        h = visualize(node_idx, n_hops, data, masked_ver, threshold=config['threshold'] , name = name, result_weights=False, low_threshold=False, experiment_name=experiment_name)
       
        #h = selected(masked_ver,masked_hor,  threshold=config['threshold'],data=data, low_threshold=False)
        res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])


        masked_ver,masked_hor = convert_binary(masked_ver, config['threshold']), convert_binary(masked_hor, config['threshold'])
        res_binary = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])

        masked_ver, masked_hor = sub(masked_ver, 0.5), sub(masked_hor,0.5)
        m = match_to_triples(masked_ver, masked_hor, data, node_idx)
        print(Counter(m[:,1].tolist()))


        hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        hor_graph, ver_graph = convert_binary(hor_graph, config['threshold']), convert_binary(ver_graph, config['threshold'])
        y_full = model.forward2(hor_graph, ver_graph)
        node_pred_full = y_full[node_idx, :]
        res_full = nn.Softmax(dim=0)(node_pred_full)



        high.append(h)
        h = dict(h)
        print('Important relations', h)



        target_label = str([k for k, v in d.items() if node_idx in v])
        info = {'label': str(target_label), 'node_idx': str(node_idx)}
        h.update(info)
        
        df.loc[str(node_idx)] = h
        df['number_neighbors'] = num_neighbors
        df['prediction_explain_binary'] = str(res_binary.detach().numpy())
        df['prediction_full'] = str(res_full.detach().numpy())
        df['prediction_explain'] = str(res.detach().numpy())

        

        # h_floats = selected(masked_ver, threshold=0.5,data=data, low_threshold=False,float=True)
        # high_floats.append(h_floats)
        # h_floats = dict(h_floats)
        # h_floats.update(info)
        # df_floats.loc[str(node_idx)] = h_floats


        print('node_idx', node_idx, 
            '\n node original label',target_label,
            '\n node predicted label explain', torch.argmax(res).item(),
            '\n node prediction probability explain', res, 'explain binary', torch.argmax(res_binary).item(),
            '\n node prediction probability explain binary', res_binary,
            '\n node predicted label full', torch.argmax(res_full).item(),
            '\n node prediction probability full', res_full,
            '\n final masks and lenght', masked_ver, len(masked_ver.coalesce().values()[masked_ver.coalesce().values()>config['threshold'] ]))
        experiment_name = f'size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}'
        if not os.path.exists(f'Relation_Importance_{name}/{experiment_name}'):
            os.makedirs(f'Relation_Importance_{name}/{experiment_name}') 
        df.to_csv(f'Relation_Importance_{name}/{experiment_name}/Relations_Important_{name}_{node_idx}.csv', index=False)
        #df_floats.to_csv(f'Relation_Importance_{name}/Relations_Important_{name}_{node_idx}_floats.csv', index=False)
        return h    