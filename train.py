import Define
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import ReadDatasets
from tqdm import *
import numpy as np
from metric import *


def train(model, metric, dataloader, criterion, optimizer, epoch, device):
    print("\nepoch: %d" % epoch)

    model.train()
    metric.train()
    train_loss = 0
    train_acc = 0
    data_num = 0
    batch_num = len(dataloader)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, 5, gamma=0.8, last_epoch=-1)

    p0 = []
    p1 = []

    for batch_id, data in enumerate(tqdm(dataloader)):
        print("第{0}个epoch，第{1}个batch".format(epoch, batch_id))
        inputs0 = data['image_0'].to(device, dtype=torch.float)
        print(inputs0.shape)
        inputs1 = data['image_1'].to(device, dtype=torch.float)
        # targets = data['label'].to(device, dtype=torch.float)
        label0 = data['label_0'].to(device, dtype=torch.float)
        label1 = data['label_1'].to(device, dtype=torch.float)
        targets = torch.eq(label0,label1).float()
        targets = targets.to(device, dtype=torch.float)
        print(targets)
        pbatch_num = targets.shape[0]
        data_num += pbatch_num

        optimizer.zero_grad()
        output1, output2 = model(inputs0, inputs1)
        metrics = metric(output1, output2)  # .squeeze()

        print(metrics.shape, targets.shape)

        # loss = criterion(metrics, targets)
        loss = criterion(metrics, label0, label1)

        loss.backward()

        optimizer.step()
        # scheduler.step()

        train_loss += float(loss.item())

    # for i,l in enumerate(data['label_0']):
    #     if l == 0:
    #         p0.append(emb1.cpu().detach().numpy())
    #     else:
    #         p1.append(emb1.cpu().detach().numpy())
    # for i,l in enumerate(data['label_1']):
    #     if l == 0:
    #         p0.append(emb2.cpu().detach().numpy())
    #     else:
    #         p1.append(emb2.cpu().detach().numpy())

        print('!!!', np.array(metrics.cpu().detach().numpy()),
              targets.cpu().detach().numpy(), '!!!\n')
        t_acc = np.sum(np.round(metrics.cpu().detach().numpy())
                       == targets.cpu().detach().numpy())
        # t_acc = np.sum(np.array(list(map(fun,metrics.cpu().detach().numpy()))) == targets.cpu().detach().numpy())
        train_acc += t_acc
    train_loss = train_loss / batch_num
    train_acc = train_acc / data_num
    return train_loss, train_acc * 100  # p0, p1


def fun(l):
    if l > 0.6:
        return 1.
    else:
        return 0.


def test(model, metric, dataloader, static, criterion, n_way,device):
    # print("\nepoch: %d" % epoch)

    model.eval()
    metric.eval()
    # test_loss = 0
    # test_acc = 0
    # data_num = 0
    # batch_num = len(dataloader)

    with torch.no_grad():
        for index, (images, labels) in enumerate(static):
            if index == 0:
                images.to(device)
                feature_train = model(images)
                label_train = labels
            else:
                images.to(device)
                outputs = model(images)
                feature_train = torch.cat([feature_train, outputs], dim=0)
                label_train = torch.cat([label_train, labels], dim=0)

            # get test set features
        for index, (images, labels) in enumerate(dataloader):
            if index == 0:
                images.to(device)
                feature_test = model(images)
                label_test = labels
            else:
                images.to(device)
                outputs = model(images)
                feature_test = torch.cat([feature_test, outputs], dim=0)
                label_test = torch.cat([label_test, labels], dim=0)
            len_train = feature_train.shape[0]
            len_test = feature_test.shape[0]
            label_train = label_train.reshape([-1]).long()
            label_test = label_test.reshape([-1]).long()

            # acc - test
            distances_test = torch.zeros([feature_test.shape[0], feature_train.shape[0]], dtype=torch.float)
            for i in range(len_test):
                for j in range(len_train):
                    dis = metric(feature_test[i].reshape([1, -1]), feature_train[j].reshape([1, -1]))
                    distances_test[i][j] = dis.reshape([1])

            test_acc = my_knn(distances_test, label_train, label_test, same_set=False, k=Define.knn_num, class_num=n_way,
                              )

            # acc - train
            distances_train = torch.zeros([feature_train.shape[0], feature_train.shape[0]], dtype=torch.float)
            for i in range(len_train):
                for j in range(len_train):
                    dis = metric(feature_train[i].reshape([1, -1]), feature_train[j].reshape([1, -1]))
                    distances_train[i][j] = dis.reshape([1])

            train_acc = my_knn(distances_train, label_train, label_train, same_set=True, k=Define.knn_num, class_num=10,
                               )

            # loss - test
            test_loss = 0
            for i in range(len_test):
                test_loss += criterion(distances_test[i], label_train, label_test[i].repeat(len_train))
            test_loss /= len_test

            # loss - train
            train_loss = 0
            for i in range(len_train):
                train_loss += criterion(distances_train[i], label_train, label_train[i].repeat(len_train))
            train_loss /= len_train

        return train_loss, train_acc, test_loss, test_acc
    # return test_loss, test_acc * 100
