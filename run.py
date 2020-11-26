from src.models.models import *
from src.data.data import *
import sys
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
    if(len(sys.argv)==2 and sys.argv[1] == "test"):
        print("Testing mode...")
        
        #load test data
        features, labels, adj = get_data("test/testdata/test.content",
                                         "test/testdata/test.cites")
    else:
        #load config data
        data_configs = json.load(open("config/data-params.json"))
        print(data_configs)

        #load cora data
        features, labels, adj = get_data(data_configs["feature_address"],
                                         data_configs["edges_address"],
                                         data_configs["encoding"],
                                         data_configs["directed"])

    model_configs = json.load(open("config/model-params.json"))
    print(model_configs)

    #train and test all the models and report losses and accuracy
    num_epochs = model_configs["num_epochs"]
    learning_rate = model_configs["learning_rate"]
    num_hidden = model_configs["num_hidden"]

    #initialize models
    in_features = list(features.size())[0]
    in_features_1 = list(features.size())[1]
    num_classes = len(set(labels))
    models = [Fully1Net(in_features, num_classes),
              Fully2Net(in_features, num_hidden, num_classes),
              Graph1Net(in_features_1, num_hidden, num_classes),
              Graph2Net(in_features_1, num_hidden, num_classes)]

    #split for train and test sets
    idx_train = torch.LongTensor(range(int(in_features_1 * 0.75)))
    idx_test = torch.LongTensor(range(int(in_features_1 * 0.75), in_features_1))

    #initialize optimizers
    ops = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]

    model_names = ["Fully1Net", "Fully2Net", "Graph1Net", "Graph2Net"]
    for i in range(len(models)):
        print("\nRunning {} Model...".format(model_names[i]))
        for epoch in range(num_epochs):
            train_model(models[i], ops[i], epoch, idx_train, adj, features, labels)
        test_model(models[i], idx_test, adj, features, labels)

    #GCN-LPA
    LPA_model = GCNLPA(features.shape[1],
                       num_hidden,
                       num_classes,
                       adj)

    optimizer = optim.Adam(LPA_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_GCN_LPA(LPA_model, optimizer, epoch, idx_train, adj, features, labels)
    test_model(LPA_model, idx_test, adj, features, labels)
if __name__ == '__main__':
    main()
