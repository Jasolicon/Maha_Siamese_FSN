import Define
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import ReadDatasets
from tqdm import *
import numpy as np


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
        targets = (data['label_1'] + data['label_0']) % 2
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


def test(model, metric, dataloader, criterion, device):
    # print("\nepoch: %d" % epoch)

    model.eval()
    metric.eval()
    test_loss = 0
    test_acc = 0
    data_num = 0
    batch_num = len(dataloader)

    with torch.no_grad():
        for batch_id, data in enumerate(tqdm(dataloader)):
            print("第%d个batch" % batch_id)
            inputs0 = data['image_0'].to(device, dtype=torch.float)
            # print(inputs0.shape)
            inputs1 = data['image_1'].to(device, dtype=torch.float)
            # targets = data['label'].to(device, dtype=torch.float)
            # print(targets)
            label0 = data['label_0'].to(device, dtype=torch.float)
            label1 = data['label_1'].to(device, dtype=torch.float)
            targets = (data['label_1'] + data['label_0']) % 2
            targets = targets.to(device, dtype=torch.float)
            pbatch_num = targets.shape[0]
            data_num += pbatch_num

            # optimizer.zero_grad()
            output1, output2 = model(inputs0, inputs1)
            metrics = metric(output1, output2)  # .squeeze()

            # loss = criterion(metrics, targets)
            loss = criterion(metrics, label0, label1)

            # loss.backward()
            # optimizer.step()

            test_loss += float(loss.item())
            print('!!!', metrics.cpu().detach().numpy(),
                  targets.cpu().detach().numpy(), '!!!\n')
            # t_acc = np.sum(np.array(list(map(fun, metrics.cpu().detach().numpy()))) == targets.cpu().detach().numpy())
            print('!!!', np.array(metrics.cpu().detach().numpy()),
                  targets.cpu().detach().numpy(), '!!!\n')
            t_acc = np.sum(np.round(metrics.cpu().detach().numpy())
                           == targets.cpu().detach().numpy())
            test_acc += t_acc
    test_loss = test_loss / batch_num
    test_acc = test_acc / data_num
    return test_loss, test_acc * 100
