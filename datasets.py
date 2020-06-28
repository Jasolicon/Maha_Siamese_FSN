"""
torchmeta数据库直接调用
    - omniglot
        The dataset is downloaded from the original [Omniglot repository]
    (https://github.com/brendenlake/omniglot). The meta train/validation/test 
    splits used in [3] are taken from [this repository]
    (https://github.com/jakesnell/prototypical-networks). These splits are 
    over 1028/172/423 classes (characters).

    - miniimagenet它包含100类，每类600张图片，其中80个类用来训练，20类用来测试。
        The dataset is downloaded from [this repository]
    (https://github.com/renmengye/few-shot-ssl-public/). The meta train/
    validation/test splits are over 64/16/20 classes.

    - tieredimagenet
        The dataset is downloaded from [this repository]
    (https://github.com/renmengye/few-shot-ssl-public/). The dataset contains 
    images from 34 categories. The meta train/validation/test splits are over 
    20/6/8 categories. Each category contains between 10 and 30 classes. The 
    splits over categories (instead of over classes) ensures that all the training 
    classes are sufficiently distinct from the test classes (unlike Mini-Imagenet).
"""

import torch
import torchmeta 
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from torchmeta.datasets.helpers import omniglot,miniimagenet,tieredimagenet
from torchmeta.utils.data import BatchMetaDataLoader
import math
import os


###############################################
class FewShotClass:
    def __init__(self, dataset_name, folder, num_shots=5, num_ways=5, test_shots=15, batch_size=20, download=False, dataloader_shuffle=True, num_workers=0,**kwargs):
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


class FewShotDataLoader(Dataset):
    def __init__(self,dataloader,train=True,meta_test=False):
        # torch.Size([20, 25, 3, 84, 84])
        # torch.Size([20, 40, 3, 84, 84])
        if not meta_test:
            self.train_data = dataloader.train_dataloader.__iter__().__next__()
        else:
            self.train_data = dataloader.test_dataloader.__iter__().__next__()
        self.to_Pair_f,self.to_Pair_l = self.train_data['train']
        self.to_Pair_f, self.to_Pair_l = self.to_Pair_f.squeeze(), self.to_Pair_l.squeeze()
        # torch.Size([80, 3, 84, 84])
        self.to_Classify_f,self.to_Classify_l = self.train_data['test']
        self.to_Classify_f, self.to_Classify_l = self.to_Classify_f.squeeze(), self.to_Classify_l.squeeze()
        # self.test_data = dataloader.test_dataloader.__iter__().__next__()

        self._train=train

    def get_Classify(self,train=False):
        self._train=train
    def __len__(self):
        if self._train:
            return len(self.to_Pair_l)*(len(self.to_Pair_l)-1)
        else:
            return len(self.to_Classify_l)#*(len(self.to_Classify_l)-1)

    def __getitem__(self, idx):
        # print('f1',idx % (len(self.to_Pair_l) - 1) , 'f0', idx // (len(self.to_Pair_l) - 1))
        if self._train:
            img_1 = self.to_Pair_f[idx // (len(self.to_Pair_l) - 1)]
            label_1 = self.to_Pair_l[idx // (len(self.to_Pair_l) - 1)]

            if idx % (len(self.to_Pair_l) - 1) < idx // (len(self.to_Pair_l) - 1):
                img_0 = self.to_Pair_f[idx % (len(self.to_Pair_l) - 1)]
                label_0 = self.to_Pair_l[idx % (len(self.to_Pair_l) - 1)]
            else:
                img_0 = self.to_Pair_f[idx % (len(self.to_Pair_l) - 1)+1]
                label_0 = self.to_Pair_l[idx % (len(self.to_Pair_l) - 1)+1]

            data_dict = {
                'image_1': img_1,
                'label_1': label_1,
                'image_0': img_0,
                'label_0': label_0,
            }
        else:
            img_1 = self.to_Classify_f[idx // (len(self.to_Classify_f) - 1)]
            label_1 = self.to_Classify_l[idx // (len(self.to_Classify_l) - 1)]

            if idx % (len(self.to_Classify_l) - 1) < idx // (len(self.to_Classify_l) - 1):
                img_0 = self.to_Classify_f[idx % (len(self.to_Classify_l) - 1)]
                label_0 = self.to_Classify_l[idx % (len(self.to_Classify_l) - 1)]
            else:
                img_0 = self.to_Classify_f[idx % (len(self.to_Classify_f) - 1) + 1]
                label_0 = self.to_Classify_l[idx % (len(self.to_Classify_l) - 1) + 1]

            data_dict = {
                'image_1': img_1,
                'label_1': label_1,
                'image_0': img_0,
                'label_0': label_0,
            }
            # label=self.to_Classify_l[idx]
            # img = self.to_Classify_f[idx]
            # # img = torch.from_numpy(img[np.newaxis, :])
            #
            # data_dict = {
            #     'image': img,
            #     'label': label,
            # }

        return data_dict
    def check(self):
        pass
        # f=f[0,0,:].view(84,84,-1)
        # plt.imshow(f)
        # plt.title(str(l))
        # plt.show()

class PosAndNegDataset(Dataset):
    def __init__(self, feature, label, std_0=True):
        self.feature=feature
        self.label = label
        self.seperate()
        self._std_0 = std_0

    def __len__(self):
        return math.ceil(len(self.label)//2)

    def seperate(self):
        f_0=[]
        l_0=[]
        f_1=[]
        l_1=[]
        for i in range(len(self.label)):
            if self.label[i]==0:
                l_0.append(self.label[i])
                f_0.append(self.feature[i])
            else:
                l_1.append(self.label[i])
                f_1.append(self.feature[i])

        self.f_0=f_0
        self.l_0=l_0
        self.f_1=f_1
        self.l_1=l_1

    def get_std_1(self):
        self._std_0=False

    def __getitem__(self,idx):
        if self._std_0:
            img = self.f_0[idx]
            label = self.l_0[idx]

        else:
            img = self.f_1[idx]
            label=self.l_1[idx]
            
        data_dict = {
            'image': img,
            'label': label,
        }

        return data_dict



if __name__ == "__main__":
    from Define import *
    dataset = FewShotClass(
        'miniimagenet', PATH['miniimagenet'], 5, 5, 5, 1)
    FSDataloader = FewShotDataLoader(dataset, True)
    FSDataloader.get_Classify()

    print(len(FSDataloader))
    plt.imshow(FSDataloader.__getitem__(0)['image_0'].permute(1,2,0))# torch.Size([3, 84, 84]),图片84，84，3
    plt.imshow(FSDataloader.__getitem__(0)['image_1'].permute(1,2,0))# torch.Size([3, 84, 84]),图片84，84，3
    plt.show()
    # PNNDataset = PosAndNegDataset(
    #     FSDataloader.to_Pair_f, FSDataloader.to_Pair_l)
    # print(PNNDataset.__getitem__(0)['label'].item(),
    #       type(PNNDataset.__getitem__(0)['label']))
    
