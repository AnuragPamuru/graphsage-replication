from src.models.models import *
from src.data.data import *
import argparse
import json

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_model(model, op, epoch, idx_train, adj_hat, features, labels):
    model.train()
    op.zero_grad()
    output = model(features, adj_hat)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    op.step()
    if (epoch + 1) % 50 == 0:
        print('Epoch: {}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()))


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
    configs = json.load(open("config/data-params.json"))
    print(configs)
    
    #load cora data
    features, labels, adj = get_data(configs["d1_address"],
                                     configs["d2_address"],
                                     configs["keys_address"])

    #initialize models
    models = [Fully1Net(), Fully2Net(), Graph1Net(), Graph2Net()]
    #initialize optimizers
    ops = [optim.SGD(model.parameters(), lr=.05) for model in models]

    #split for train and test sets
    idx_train = torch.LongTensor(range(1000))
    idx_test = torch.LongTensor(range(1000, 1433))

    #train and test all the models and report losses and accuracy
    num_epochs = 500
    for i in range(len(models)):
        for epoch in range(configs["num_epochs"]):
            train_model(models[i], ops[i], epoch, idx_train, adj, features, labels)
        test_model(models[i], idx_test, adj, features, labels)
if __name__ == '__main__':
    main()
