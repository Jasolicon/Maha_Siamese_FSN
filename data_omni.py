import torch
import torchmeta
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchmeta.datasets.helpers import omniglot,miniimagenet,tieredimagenet
from torchmeta.utils.data import BatchMetaDataLoader
import math
import os

class FewShotClass:
    def __init__(self, dataset_name, folder, num_shots=5, num_ways=5, test_shots=15, batch_size=20, download=False, dataloader_shuffle=False, num_workers=0,**kwargs):
        datasetnamedict = {'omniglot': omniglot,
                           'miniimagenet': miniimagenet, 'tieredimagenet': tieredimagenet}
        dataset=datasetnamedict[dataset_name]

        #输入[metatrain,metatest]
        test_shots=[test_shots,test_shots] if isinstance(test_shots,int) else test_shots
        batch_size=[batch_size,batch_size] if isinstance(batch_size,int) else batch_size

        #meta-train
        self.train_dataset = dataset(folder, shots=num_shots, ways=num_ways,
                                     shuffle=dataloader_shuffle, test_shots=test_shots[0], meta_train=True, download=download)
        self.train_dataloader = BatchMetaDataLoader(self.train_dataset, batch_size=batch_size[0],
                                         shuffle=dataloader_shuffle, num_workers=num_workers)

        #meta-test
        self.test_dataset = dataset(folder, shots=num_shots, ways=num_ways,
                                    shuffle=dataloader_shuffle, test_shots=test_shots[1], meta_test=True, download=download)
        self.test_dataloader = BatchMetaDataLoader(self.test_dataset, batch_size=batch_size[1],
                                                   shuffle=dataloader_shuffle, num_workers=num_workers)


class TaskPaired(Dataset):
    #获得[task_batch_size, N*K, channel, weight, height][task_batch_size, N*K]
    def __init__(self,batch_task,):
        self.task_feature,self.task_label = batch_task
        self.task_feature=self.task_feature.view(-1,*self.task_feature.shape[2:])
        self.task_label=self.task_label.view(1,-1)
        print(self.task_feature.shape,self.task_label.shape)
        self.length=self.task_label.shape[1]

    def __len__(self):
        return self.length*(self.length-1)//2

    def __getitem__(self, item):
        n = self.length
        index1 = math.floor((2 * n - 1 - ((2 * n - 1) ** 2 - 8 * item) ** 0.5) / 2)
        index2 = item - (((2 * n - 1) * index1 - index1 ** 2) // 2) + index1 + 1
        # print(n,index1,index2)

        data0=self.task_feature[index1]
        label0=self.task_label[0,index1]
        data1=self.task_feature[index2]
        label1=self.task_label[0,index2]

        data_dict = {
            'image_0': data0,
            'label_0': label0,
            'image_1': data1,
            'label_1': label1,
        }
        return data_dict


if __name__ == "__main__":
    # a=torch.Tensor([1,2,3,4,5])
    # b=torch.Tensor([5,4,3,2,1])
    # c = torch.eq(a,b).float()
    # print(c)
    print("start")
    import Define
    import paint
    # dataset=FewShotClass("omniglot",Define.PATH["omniglot"],5,2,5,1)
    dataset=FewShotClass("miniimagenet",Define.PATH["miniimagenet"],5,5,5,1)
    dl = dataset.train_dataloader
    for i,batch in enumerate(dl):
        f,l=batch['train']
        # tp = TaskPaired(batch["train"])
        # tpdl = DataLoader(tp, batch_size=20, shuffle=True)
        # a=torch.Tensor()
        # for j,b in enumerate(tpdl):
        #     ff=b['image_0']#.view(-1,*a.shape[2:])
        #     fd=b['image_1']#.view(-1,*a.shape[2:])
        #     print(ff.shape)
        #     paint.paint_pics(ff,'task0{0}{1}'.format(i,j))
        #     paint.paint_pics(fd,'task1{0}{1}'.format(i,j))
        print(l)
        # for j,b in enumerate(tpdl):
        #     print(b['label_0'],b['label_1'])
        #
        #     break

        if i >=2:break

