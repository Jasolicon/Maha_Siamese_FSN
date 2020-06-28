from torchmeta.datasets import omniglot, miniimagenet, tieredimagenet
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import json
from PIL import Image
import numpy as np
from collections import Counter
import random
# TODO: 按类分；总数据；数据对


class ReadDatasets(Dataset):
    def __init__(self, root, dataset_name, meta_split='train', **kwargs):
        # kwargs:

        self.meta_split = meta_split
        self.root = root
        if dataset_name == 'omniglot' or 'omni':
            self._dataset = self._read_omniglot(**kwargs)
        elif dataset_name == 'miniimagenet' or 'mini':
            self._dataset = self._read_miniimagenet()
        elif dataset_name == 'tieredimagenet' or 'tiered':
            self._dataset = self._read_tieredimagenet()

    @property
    def get_whole_datas(self):
        return self._dataset

    def _read_omniglot(self, **kwargs):
        path = os.path.join(self.root, 'omniglot')
        data_path = os.path.join(path, 'data.hdf5')
        print(data_path)
        _data = h5py.File(data_path, 'r')
        filename_labels = '{0}{1}_labels.json'
        split_filename_labels = os.path.join(
            path,
            filename_labels.format(
                'vinyals_' if (
                    'use_vinyals_split' in kwargs and kwargs['use_vinyals_split']) else '',
                self.meta_split))
        print(split_filename_labels)
        with open(split_filename_labels, 'r') as f:
            _labels = json.load(f)
        return (_data, _labels)

    def _read_miniimagenet(self, root):
        path = os.path.join(root, 'miniimagenet')
        # data_path = os.path.join(path, 'data.hdf5')
        # data = h5py.File(data_path, 'r')
        # return data

    def _read_tieredimagenet(self, root):
        pass

    def __len__(self):
        return len(self._dataset[1])

    def __getitem__(self, item):
        return self._dataset[0][item], self._dataset[1][item]

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None


class Get_Category_Pairs(Dataset):
    def __init__(self, dataset, evaluation=False, same_letter=False, **kwargs):
        # :dataset (feature.label)
        # :kwargs: categories:选择前多少个类
        self.features = dataset[0]
        self.labels = dataset[1]
        self.same_letter = same_letter
        self._keys = self.features.keys()
        # image_background; image_evaluation
        if evaluation:
            self._key = list(self._keys)[1]
        else:
            self._key = list(self._keys)[0]

        self._categories = self.features[self._key].keys()
        self.need_categories = kwargs['categories'] if 'categories' in kwargs else len(
            self._categories)
        # ['Alphabet_of_the_Magi', 'Anglo-Saxon_Futhorc', 'Arcadian', 'Armenian', 'Asomtavruli_(Georgian)', 'Balinese', 'Bengali', 'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Braille', 'Burmese_(Myanmar)', 'Cyrillic', 'Early_Aramaic', 'Futurama', 'Grantha', 'Greek', 'Gujarati', 'Hebrew', 'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Japanese_(hiragana)', 'Japanese_(katakana)', 'Korean', 'Latin', 'Malay_(Jawi_-_Arabic)', 'Mkhedruli_(Georgian)', 'N_Ko', 'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Sanskrit', 'Syriac_(Estrangelo)', 'Tagalog', 'Tifinagh']
        self._category_num = np.zeros(len(self._categories))
        for n, m in enumerate(self._categories):
            for i in range(len(self.labels)):
                if self.labels[i][1] == m:
                    self._category_num[n] += 1

        # print('really?')

    @property
    def get_features(self):
        return self.features

    @property
    def get_categories_num(self):
        return self._category_num

    def __len__(self):
        return int(np.sum(self._category_num))

    def __getitem__(self, item):
        if self.need_categories < self.__len__():
            length = np.sum(self._category_num[:self.need_categories - 1])
            print('length:', length)
        else:
            length = self.__len__()

        length = int(length)
        i = item % length
        if self.same_letter:
            j = i
        else:
            j = np.random.randint(length)
        # j= np.random.randint(1) % self.__len__()
        anchor0 = self.labels[i]
        anchor1 = self.labels[j]
        pairs_f = [anchor0, anchor1]
        if i == j:
            pairs_l = [0, 0]
        else:
            pairs_l = [0, 1]
        pairs = [pairs_f, pairs_l]
        return pairs


class Get_Letter_Pairs(Dataset):
    def __init__(self, features, catogories_pairs, shots=5, val_shots=5):
        self.anchor = catogories_pairs[0]
        self.labels = catogories_pairs[1]

        # self._label = self.labels[0] ^ self.labels[1]

        self.features = features
        self.shots = shots
        self.val_shots = val_shots

        self._read_characters()
        self._sample_in_class()

    def _read_characters(self):
        self.character0 = self.features[self.anchor[0][
            0]][self.anchor[0][1]][self.anchor[0][2]]
        self.character1 = self.features[self.anchor[1][
            0]][self.anchor[1][1]][self.anchor[1][2]]
        self.len_character0 = len(self.character0)
        self.len_character1 = len(self.character1)

    def _sample_in_class(self):
        self._sample0 = random.sample(
            list(
                self.character0),
            self.shots +
            self.val_shots)
        self._sample1 = random.sample(
            list(
                self.character1),
            self.shots +
            self.val_shots)
        self._sample_spt = self._sample0[:self.shots] + \
            self._sample1[:self.shots]
        self._sample_val = self._sample0[self.shots:] + \
            self._sample1[self.shots:]
        self._sample_np_spt = np.expand_dims(
            np.array(self._sample_spt), axis=1)
        self._sample_np_val = np.expand_dims(
            np.array(self._sample_val), axis=1)

        self._label_spt = np.zeros(self.shots * 2)
        self._label_spt[self.shots:] = 1
        self._label_val = np.zeros(self.val_shots * 2)
        self._label_val[self.val_shots:] = 1

    def get_pics(self):
        return self._sample_np_spt, self._sample_np_val

    @property
    def task(self):
        return {
            'train': (
                self._sample_np_spt,
                self._label_spt),
            'val': (
                self._sample_np_val,
                self._label_val)}


class Meta(Dataset):
    def __init__(self, dataset):
        self.f = dataset[0]
        self.l = dataset[1]

    def __len__(self):
        return len(self.l) * (len(self.l) - 1)

    def __getitem__(self, item):
        img_1 = self.f[item // (len(self.l) - 1)]
        label_1 = self.l[item // (len(self.l) - 1)]

        if item % (len(self.l) - 1) < item // (len(self.l) - 1):
            img_0 = self.f[item % (len(self.l) - 1)]
            label_0 = self.l[item % (len(self.l) - 1)]
        else:
            img_0 = self.f[item % (len(self.l) - 1) + 1]
            label_0 = self.l[item % (len(self.l) - 1) + 1]
        label = (label_1 + label_0) % 2
        data_dict = {
            'image_1': img_1,
            'label_1': label_1,
            'image_0': img_0,
            'label_0': label_0,
            'label': label
        }
        return data_dict

# class ReturnMeta(Dataset):
#     def __init__(self,root,dataset_name,meta_split='train',shots=5,val_shots=15, same_letter=False, **kwargs):
#         rd=ReadDatasets(root,dataset_name,meta_split)
#         f, l = data.get_whole_datas
#         cp=Get_Category_Pairs((f,l),same_letter)
#         lp=Get_Letter_Pairs(f,)


if __name__ == '__main__':
    import Define
    import matplotlib.pyplot as plt
    import paint
    data = ReadDatasets(Define.PATH['omniglot'], 'omni')
    f, l = data.get_whole_datas
    for n in range(10):
        print('第{0}对'.format(n))
        pair = Get_Category_Pairs((f, l))
        # pair_dataloader=DataLoader(pair)

        for i, p in enumerate(pair):
            print('第{0}个任务'.format(i))
            ls = Get_Letter_Pairs(f, p,)
            print(ls.anchor)
            train_data = ls.task['train']
            val_data = ls.task['val']
            meta = Meta(train_data)
            vmeta = Meta(val_data)
            meta_dataloader = DataLoader(meta, batch_size= 20,shuffle=True)
            val_meta_dataloader = DataLoader(vmeta, shuffle=True)
            for j, m in enumerate(meta_dataloader):
                print('第{0}对孪生数据, label:{1}'.format(j, m['label']))
                paint.paint_pics_1(m['image_0'],'img0{0}.png'.format(j))
                paint.paint_pics_1(m['image_1'],'img1{0}.png'.format(j))
                break

                print("m['label_0'],m['label_1']", m['label_0'], m['label_1'])
                # plt.show()

            # for k, v in enumerate(val_meta_dataloader):
            #     print('第{0}对孪生验证数据, label:{1}'.format(k, v['label']))
            #     print("m['label_0'],m['label_1']", v['label_0'], v['label_1'])
