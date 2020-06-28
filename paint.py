import networks
import copy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from math import *
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from datasets import *
def paint_pics(data,name):
    fig=plt.figure()
    l=len(data)
    up=ceil(sqrt(l))
    down = floor(sqrt(l))
    print('up',up,'down',down)
    for i in range(l):
        plt.subplot(up,down,i+1)
        plt.imshow(data[i].permute(1,2,0))
    plt.savefig(name+'.png')

def paint_pics_1(name,*data ):
    fig = plt.figure()
    for d in data:
        l = len(d)
        up = ceil(sqrt(l))
        down = floor(sqrt(l))
        print('up', up, 'down', down)
        for i in range(l):
            plt.subplot(up, down, i + 1)
            plt.imshow(np.squeeze(d[i]))
    plt.savefig(name + '.png')
    # plt.show()

def paint_scatters(name: object, *args: object) -> object:
    fig=plt.figure()
    # print(len(args))
    # clr={'r','b'}
    for i,data in enumerate(args):
        data_x=[d[0] for d in data]
        data_y=[d[1] for d in data]
        plt.scatter(data_x,data_y)
        # print('i',i)
    # plt.scatter(data1[0],data1[1],c='green')
    plt.savefig(name+'.png')
    # plt.show()


    

if __name__ == "__main__":
    plt.plot([1,2,3,4,5,6],[7,6,5,4,8,2,])
    plt.show()
    # a = [torch.Tensor([1, 2]).numpy(), torch.Tensor([3, 4]).numpy()]
    # b = [torch.Tensor([5, 6]).numpy(), torch.Tensor([7, 8]).numpy()]
    # c = [torch.Tensor([11, 12]).numpy(), torch.Tensor([9,10]).numpy()]
    # d = [torch.Tensor([2, 12]).numpy(), torch.Tensor(
    #     [9, 6]).numpy(), torch.Tensor([9, 16]).numpy()]
    # e=[np.array([-0.03363481,  0.20244735]), np.array([-0.03450943,  0.20331284]), np.array([-0.03397714,  0.20455754]), np.array([-0.03522148,  0.20401022]), np.array([-0.0343244,  0.20338371]), np.array([-0.03400132,  0.20285928]), np.array([-0.03350436,  0.2049334]), np.array([-0.03511086,  0.20448884]), np.array([-0.03388133,  0.20510146]), np.array([-0.03367164,  0.20261914]), np.array([-0.03547236,  0.20472041]), np.array([-0.03422207,  0.20484173]), np.array([-0.03422299,  0.20523202]), np.array([-0.03439202,  0.20428663]), np.array([-0.03425793,  0.20420262]), np.array([-0.03337302,  0.2038296])
    #  ]
    # f=[np.array([-0.03411731,  0.20381817]), np.array([-0.03417028,  0.2035674]), np.array([-0.03395596,  0.20367175]), np.array([-0.03405499,  0.20454752]),]
    # # print(list(map(list, e)))
    # paint_scatters('eg',list(map(list, e)), list(map(list, e)))
    # paint_scatters('f', f)

    # dataset = FewShotClass(
    #     'miniimagenet', '/home/ljy/DataSets/miniimagenet/', 20, 2, 20, 1)
    # FSDataloader = FewShotDataLoader(dataset, True,False)
    # paint_pics(FSDataloader.to_Pair_f,
    #            '/home/ljy/Codes/LMMNet-v3.01-AllPairSampling/FSDataloader_to_Pair_f')
    # # print(FSDataloader.__getitem__(90))
    # trainloadero = DataLoader(
    #     FSDataloader, batch_size=10, shuffle=True, num_workers=0)
    # trainloader = copy.deepcopy(trainloadero)
    # PNNDataset = PosAndNegDataset(
    #     FSDataloader.to_Pair_f, FSDataloader.to_Pair_l)
    # stdloader_0o = DataLoader(PNNDataset, batch_size=1,
    #                           shuffle=False, num_workers=0)
    # stdloader_0 = copy.deepcopy(stdloader_0o)
    # stdloader_1o = DataLoader(PNNDataset, batch_size=1,
    #                           shuffle=False, num_workers=0)
    # stdloader_1 = copy.deepcopy(stdloader_1o)
    # FSDataloader.get_Classify()
    # testloadero = DataLoader(FSDataloader, shuffle=False)
    # testloader = copy.deepcopy(testloadero)
    
    # paint_pics(FSDataloader.to_Classify_f,
    #            '/home/ljy/Codes/LMMNet-v3.01-AllPairSampling/FSDataloader_to_Classify_f')
    # paint_pics(PNNDataset.f_0,
    #            '/home/ljy/Codes/LMMNet-v3.01-AllPairSampling/PNNDataset_f_0')
    # paint_pics(PNNDataset.f_1,
    #            '/home/ljy/Codes/LMMNet-v3.01-AllPairSampling/PNNDataset_f_1')
