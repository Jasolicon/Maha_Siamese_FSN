import copy
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from data_omni import *
from Define import *
from metric import *
from models.networks import *
from models.res18 import resnet18
from models.networks import *
from paint import *
from torch.utils.data import DataLoader, Dataset
from train import *


def experiment(embedding_model, n_way=2, k_shot=5, test_shots=15, dataset_name='miniimagenet', device_num=0):
    device = train_by_GPU(device_num)
    embednet = embedding_model
    siamese = SiameseNet(embednet).to(device)
    mnet = MNet(64).to(device)
    # mnet = L2Metric().to(device)
    optimizer = optim.Adam([{'params': siamese.parameters()}, {
        'params': mnet.parameters()}], lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, 5, gamma=0.8, last_epoch=-1)

    loss = MyCriterion()

    avg_train_loss = []
    avg_train_acc = []
    test_loss = []
    test_acc = []
    last_acc = 0

    # + str(time.asctime(time.localtime(time.time())))
    name = str(n_way)+" way "+str(k_shot)+" shot "
    if not os.path.exists(name):
        os.mkdir(name)

    dataset=FewShotClass(dataset_name,PATH[dataset_name],k_shot,n_way,test_shots,1)
    meta_train_dataloader = dataset.train_dataloader
    meta_test_dataloader = dataset.test_dataloader
    for i, batch in enumerate(meta_train_dataloader):
        # mnet.reset_parameters()
        # siamese.reset()
        tp = TaskPaired(batch["train"])
        tpdl = DataLoader(tp, batch_size=20, shuffle=True)
        # tpv = TaskPaired(batch["test"])
        # tpvdl = DataLoader(tpv, batch_size=20, shuffle=True)

        train_loss_per_task = []
        train_acc_per_task = []
        #模型开始训练
        for epoch in range(epoch_num):
            print("第{0}个任务".format(i))
            train_loss, train_acc = train(
                siamese, mnet, tpdl, loss, optimizer, epoch, device)
            train_loss_per_task.append(train_loss)
            train_acc_per_task.append(train_acc)

        train_loss, train_acc, ftest_loss, ftest_acc = test(
            embednet, mnet, batch['test'], batch['train'], loss, 5, device)
        print('任务{0}测试准确率{1}'.format(i, ftest_acc))
        Define.save_model(ftest_acc, last_acc, i, siamese,
                          './{0}/siamese.pkl'.format(name), mnet, './{0}/metric.pkl'.format(name))

        test_loss.append(ftest_loss)
        test_acc.append(ftest_acc)
        avg_train_loss.append(sum(train_loss_per_task) /
                              len(train_acc_per_task))
        avg_train_acc.append(sum(train_acc_per_task) / len(train_acc_per_task))
        if i>=meta_train_num:break

        # fig = plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.plot(train_loss_per_task)
        # plt.subplot(1, 2, 2)
        # plt.plot(train_acc_per_task)

        # plt.savefig('task_mini%d'% cls)
    fig2 = plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("avg trn loss:" +
              str(np.sum(np.array(avg_train_loss))/len(avg_train_loss)))
    plt.plot(avg_train_loss)

    plt.subplot(2, 2, 2)
    plt.title("avg trn acc" + str(np.sum(
        np.array(avg_train_acc))/len(avg_train_acc)))
    plt.plot(avg_train_acc)
    plt.subplot(2, 2, 3)
    plt.title("avg val loss"+str(np.sum(
        np.array(test_loss))/len(test_loss)))
    plt.plot(test_loss)
    plt.subplot(2, 2, 4)
    plt.plot(test_acc)
    plt.title("avg val acc:"+str(np.sum(
        np.array(test_acc))/len(test_acc)))
    plt.savefig("./{0}/test_mini".format(name))

    ##meta test
    avg_train_loss = []
    avg_train_acc = []
    test_loss = []
    test_acc = []
    last_acc = 0


    best_model = SiameseNet(embedding_model).to(device)
    best_model.load_state_dict(torch.load("./{0}/siamese.pkl".format(name)))
    best_metric = MNet(64).to(device)
    best_metric.load_state_dict(torch.load("./{0}/metric.pkl".format(name)))
    for i,batch in enumerate(meta_test_dataloader):
        # mnet.reset_parameters()
        # siamese.reset()
        mttp = TaskPaired(batch["train"])
        mttpdl=DataLoader(mttp,shuffle=True)
        # mttpv = TaskPaired(batch["test"])
        # mttpvdl=DataLoader(mttpv,shuffle=True)
        train_loss_per_task = []
        train_acc_per_task = []
        #模型开始训练
        for epoch in range(20):
            print("第{0}个任务".format(i))
            train_loss, train_acc = train(
                best_model, best_metric, mttpdl, loss, optimizer, epoch, device)
            # train_loss_per_task.append(train_loss)
            # train_acc_per_task.append(train_acc)

        train_loss, train_acc, ftest_loss, ftest_acc = test(
            best_model, best_metric, batch['test'],batch['train'], loss,5, device)
        train_loss_per_task.append(train_loss)
        train_acc_per_task.append(train_acc)
        print('任务{0}测试准确率{1}'.format(i, ftest_acc))
        test_loss.append(ftest_loss)
        test_acc.append(ftest_acc)
        avg_train_loss.append(sum(train_loss_per_task) /
                              len(train_acc_per_task))
        avg_train_acc.append(sum(train_acc_per_task) / len(train_acc_per_task))

    fig2 = plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("avg trn loss:" +
              str(np.sum(np.array(avg_train_loss))/len(avg_train_loss)))
    plt.plot(avg_train_loss)

    plt.subplot(2, 2, 2)
    plt.title("avg trn acc" + str(np.sum(
        np.array(avg_train_acc))/len(avg_train_acc)))
    plt.plot(avg_train_acc)
    plt.subplot(2, 2, 3)
    plt.title("avg val loss"+str(np.sum(
        np.array(test_loss))/len(test_loss)))
    plt.plot(test_loss)
    plt.subplot(2, 2, 4)
    plt.plot(test_acc)
    plt.title("avg val acc:"+str(np.sum(
        np.array(test_acc))/len(test_acc)))

    plt.savefig("./{0}/meta_test_mini".format(name))
