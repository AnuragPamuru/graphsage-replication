import numpy as np
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

def get_data(feature_address, edges_address, encoding_config, directed):
    features = pd.read_csv(feature_address, sep ='\t', header=None)
    edges = pd.read_csv(edges_address, sep ='\t', header=None)

    #adjacency matrix
    adj = get_adj(edges, directed)
    
    #encoding
    encoded_labels = encode(features, encoding_config)
    
    #add identity matrix to adjacency matrix
    adj_added = adj + np.eye(adj.shape[0])

    A = torch.from_numpy(adj_added).float()

    # Normalization as per (Kipf & Welling, ICLR 2017)
    D = A.sum(1)  # nodes degree (N,)
    D_hat = (D + 1e-5) ** (-0.5)
    adj_hat = D_hat.view(-1, 1) * A * D_hat.view(1, -1)  # N,N

    # Some additional trick I found to be useful
    adj_hat[adj_hat > 0.0001] = adj_hat[adj_hat > 0.0001] - 0.2

    #put numpy arrays to tensors
    features = np.array(features.iloc[:, 1:features.shape[1]-1])
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(encoded_labels)[1])

    return features, labels, adj_hat
