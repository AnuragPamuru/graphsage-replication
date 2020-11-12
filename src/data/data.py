import numpy as np
import pandas as pd
import torch

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def get_data(d1_address, d2_address, keys_address):
    
    d1 = (pd.read_csv(d1_address, sep ='\t', header=None))
    d2 = pd.read_csv(d2_address, sep ='\t', header=None)
    keys = pd.read_csv(keys_address, sep ='\t', header=None)[[0,1434]]

    index = d1[0]

    id2index = {}
    for i in np.arange(len(index)):
        id2index[index[i]] = i
        i += 1

    n_papers = len(id2index)

    labels = d1[1434]
    label2index = {
            'Case_Based': 0,
            'Genetic_Algorithms': 1,
            'Neural_Networks': 2,
            'Probabilistic_Methods': 3,
            'Reinforcement_Learning': 4,
            'Rule_Learning': 5,
            'Theory': 6
    }

    label_ind = np.zeros(n_papers)
    for i in np.arange(n_papers):
        label_ind[i] = label2index[labels[i]]

    adj = np.zeros((n_papers, n_papers), dtype='float32')

    rows = d2[0]
    cols = d2[1]

    for i in np.arange(len(d2)):
        adj[id2index[rows[i]], id2index[cols[i]]]  = 1.0 # directed graph

    features = np.array(d1.iloc[:, 1:1434])

    encoded_labels = np.array(encode_onehot(label_ind))
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
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(encoded_labels)[1])

    return features, labels, adj_hat
