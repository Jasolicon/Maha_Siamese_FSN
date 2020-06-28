import torch
import torch.nn as nn
import torch.optim as optim
from networks import *
from metric import *
from Define import *
from train import *
from paint import *
from torch.utils.data import Dataset, DataLoader
from datasets import *
from models.res18 import resnet18
from models.networks import *
import copy
if __name__ == "__main__":
    device = train_by_GPU(0)
    embednet = EmbeddingNet()
    # embednet = ResNet18()
    siamese = SiameseNet(embednet).to(device)
    mnet = MNet(64).to(device)
    # mnet = L2Metric().to(device)
    optimizer = optim.Adam([{'params': siamese.parameters()}, {
        'params': mnet.parameters()}], lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 5, gamma=0.8, last_epoch=-1)

    loss = nn.MSELoss()

    # data = ReadDatasets.ReadDatasets(Define.PATH['omniglot'], 'omni')
    # f, l = data.get_whole_datas

    avg_train_loss = []
    avg_train_acc = []
    test_loss = []
    test_acc = []
    last_acc = 0

    points0 = []
    points1 = []

    dataset = FewShotClass(
        'miniimagenet', PATH['miniimagenet'], 5, 5, 5, 1)

    for cls in range(meta_train_num):
        mnet.reset_parameters()
        siamese.reset()
        FSDataloader = FewShotDataLoader(dataset, True)
        # print(FSDataloader.__getitem__(90))
        trainloadero = DataLoader(
            FSDataloader, batch_size=10, shuffle=True, num_workers=0)

        trainloader = copy.deepcopy(trainloadero)

        FSDataloader.get_Classify()
        testloadero = DataLoader(FSDataloader, batch_size=10)

        testloader = copy.deepcopy(testloadero)
        train_loss_per_task = []
        train_acc_per_task = []
        for epoch in range(epoch_num):
            print("第{0}个任务".format(cls))
            # scheduler.step()
            train_loss, train_acc, emb = train(siamese, mnet, trainloader, loss, optimizer, epoch, device)
            # print('train_loss %f, train_acc %f' % (train_loss,train_acc))
            train_loss_per_task.append(train_loss)
            train_acc_per_task.append(train_acc)
            points0.append(emb[0].cpu().detach().numpy())
            points1.append(emb[1].cpu().detach().numpy())

        # paint_scatters('points0', p0)
        # paint_scatters('points1', p1)
        paint_scatters('points', points0,points1)
        ftest_loss,ftest_acc = test(siamese, mnet, testloader, loss,device)
        print('任务{0}测试准确率{1}'.format(cls,ftest_acc))
        Define.save_model(ftest_acc, last_acc, cls, siamese, 'siamese.pkl', mnet, 'metric.pkl')
        # print('test_acc %f' % ftest_acc)
        test_loss.append(ftest_loss)
        test_acc.append(ftest_acc)
        avg_train_loss.append(sum(train_loss_per_task) / len(train_acc_per_task))
        avg_train_acc.append(sum(train_acc_per_task) / len(train_acc_per_task))

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_per_task)
        plt.subplot(1, 2, 2)
        plt.plot(train_acc_per_task)

        plt.savefig('task_mini%d'% cls)
    # plt.savefig('task %d.png' % i)
    # if i >= 20:
    #     break
    fig2 = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(avg_train_loss)
    plt.subplot(2, 2, 2)
    plt.plot(avg_train_acc)
    plt.subplot(2, 2, 3)
    plt.plot(test_loss)
    plt.subplot(2, 2, 4)
    plt.plot(test_acc)
    plt.savefig("test_mini")
# plt.savefig("test %d.png" % n)

    # paint_pics(FSDataloader.to_Pair_f,
    #            ROOT+'/FSDataloader_to_Pair_f')
    # paint_pics(FSDataloader.to_Classify_f,
    #            ROOT+'/FSDataloader_to_Classify_f')