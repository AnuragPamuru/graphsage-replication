from ogb.nodeproppred import NodePropPredDataset
from scipy.sparse import identity
from scipy.sparse import coo_matrix
import numpy as np
import networkx as nx
import pandas as pd
import torch

def encode_onehot(labels):
    #ordinal
    nrows = len(labels)
    unique = set(labels)
    classes_dict = {c: np.identity(len(unique))[i, :] for i, c in
                    enumerate(unique)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def encode(features, encoding_config):
    encoded_data = []
    for encoding_instruction in encoding_config:
        encoder_col = features[encoding_instruction["column"]]
        if encoding_instruction["encoding_types"] == "one_hot":
            encoder_col = encode_onehot(encoder_col)
            features.drop(encoding_instruction["column"], axis = 1)
        encoded_data.append(encoder_col)
    return encoded_data if len(encoded_data) > 1 else encoded_data[0]

def get_adj(edges, directed):
    rows = edges[0]
    cols = edges[1]
    
    nodes = list(set(edges[0]).union(set(edges[1])))
    n_nodes = len(nodes)
    
    node_index = {}
    for i in np.arange(len(nodes)):
        node_index[nodes[i]] = i
        i += 1
    
    adj = np.zeros((n_nodes, n_nodes), dtype='float32')

    for i in range(len(edges)):
        adj[node_index[rows[i]], node_index[cols[i]]]  = 1.0
        if not directed: 
            adj[node_index[cols[i]], node_index[rows[i]]]  = 1.0 
            
    return adj

def get_data(feature_address, edges_address, encoding_config = None, directed = False):
    if feature_address == 'arxiv':
        d = NodePropPredDataset('ogbn-arxiv', root='/datasets/ogb/ogbn-arxiv')
        graph, labels = d[0]
        labels = np.ravel(labels)
        edges = list(zip(graph["edge_index"][0], graph["edge_index"][1]))
        G = nx.DiGraph(edges)
        adj = nx.adjacency_matrix(G)
    else:
        features = pd.read_csv(feature_address, sep ='\t', header=None)
        edges = pd.read_csv(edges_address, sep ='\t', header=None)

        #adjacency matrix
        adj = get_adj(edges, directed)
        
        #encoding
        encoded_labels = features if encoding_config == None else encode(features, encoding_config)
    
    #put numpy arrays to tensors
    if feature_address == 'arxiv' and torch.cuda.is_available():
        device = torch.device('cuda')
        features = torch.FloatTensor(graph["node_feat"]).cuda().to(device)
        labels = torch.LongTensor(labels).cuda().to(device)
        
        adj_added = coo_matrix(adj + identity(adj.shape[0]))
        
        #add identity matrix to adjacency matrix
        values = adj_added.data
        indices = np.vstack((adj_added.row, adj_added.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj_added.shape

        A = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    else:
        features = np.array(features.iloc[:, 1:features.shape[1]-1])
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(encoded_labels)[1])
        
        #add identity matrix to adjacency matrix
        adj_added = adj + np.eye(adj.shape[0])

        A = torch.from_numpy(adj_added).float()
    
    return features, labels, A
