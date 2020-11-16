from src.models.models import *
from src.data.data import *
import argparse
import json

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_GCN_LPA(model, op, epoch, idx_train, adj, features, labels):
    model.train()
    op.zero_grad()
    output, y_hat = model(features, adj, labels)
    loss_gcn = F.cross_entropy(output[idx_train], labels[idx_train])
    loss_lpa = F.cross_entropy(y_hat, labels)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train = loss_gcn + 10 * loss_lpa
    loss_train.backward(retain_graph=True)
    op.step()
    if (epoch + 1) % 5 == 0:
        print('Epoch: {}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()))

def train_model(model, op, epoch, idx_train, adj_hat, features, labels):
    model.train()
    op.zero_grad()
    output = model(features, adj_hat)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    op.step()
    if (epoch + 1) % 5 == 0:
        print('Epoch: {}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()))

def test_GCN_LPA(model, idx_test, adj_hat, features, labels):
    model.eval()
    output, y_hat = model(features, adj_hat, labels)
    loss_gcn = F.cross_entropy(output[idx_test], labels[idx_test])
    loss_lpa = F.cross_entropy(y_hat, labels)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    loss_train = loss_gcn + 10 * loss_lpa
    
    print("\nTest set results:",
          "loss_test: {:.4f}".format(loss_test.item()),
          "accuracy_test: {:.4f}".format(acc_test.item()))
        
def test_model(model, idx_test, adj_hat, features, labels):
    model.eval()
    output = model(features, adj_hat)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("\nTest set results:",
          "loss_test: {:.4f}".format(loss_test.item()),
          "accuracy_test: {:.4f}".format(acc_test.item()))


def main():
    #load config data
    data_configs = json.load(open("config/data-params.json"))
    print(data_configs)
    
    model_configs = json.load(open("config/model-params.json"))
    print(model_configs)
    
    #load cora data
    features, labels, adj = get_data(data_configs["feature_address"],
                                     data_configs["edges_address"],
                                     data_configs["encoding"],
                                     data_configs["directed"])

    #train and test all the models and report losses and accuracy
    num_epochs = model_configs["num_epochs"]
    learning_rate = model_configs["learning_rate"]
    num_hidden = model_configs["num_hidden"]
    
    #initialize models
    in_features = list(features.size())[0]
    num_classes = len(set(labels))
    models = [Fully1Net(in_features, num_classes), 
              Fully2Net(in_features, num_hidden, num_classes), 
              Graph1Net(in_features, num_classes), 
              Graph2Net(in_features, num_hidden, num_classes)]
    

    #split for train and test sets
    idx_train = torch.LongTensor(range(1000))
    idx_test = torch.LongTensor(range(1000, 1433))
    
    #initialize optimizers
    ops = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]
    
    LPA_model = GCNLPA(features.shape[1],
                       num_hidden,
                       num_classes,
                       adj)
    optimizer = optim.Adam(LPA_model.parameters(), lr=learning_rate)
    
    features = torch.transpose(features, 0, 1)
    model_names = ["Fully1Net", "Fully2Net", "Graph1Net", "Graph2Net"]
    for i in range(len(models)):
        print("Running {} Model...".format(model_names[i]))
        for epoch in range(num_epochs):
            train_model(models[i], ops[i], epoch, idx_train, adj, features, labels)
        test_model(models[i], idx_test, adj, features, labels)
        
    for epoch in range(num_epochs):   
        train_GCN_LPA(LPA_model, optimizer, epoch, idx_train, adj, features, labels)
    test_model(LPA_model, idx_test, adj, features, labels)
if __name__ == '__main__':
    main()
