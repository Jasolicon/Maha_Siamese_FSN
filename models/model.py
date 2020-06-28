""" FNet 2D with input size of 128 * 128

"""

import torch
from torch import nn

import os


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FNet(nn.Module):
    def __init__(self, get_feature=False):
        super(FNet, self).__init__()

        self.get_feature = get_feature
        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]

        for i in range(len(num_blocks_forw)):  # 4
            blocks = []
            for j in range(num_blocks_forw[i]):  # {2,2,3,3}
                if j == 0:  # conv
                    # plus source connection
                    blocks.append(PostRes(self.featureNum_forw[i] + 1, self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.postBlock = None
        self.classifier = None

    def forward(self, x):
        out = self.preBlock(x)                                      # b*24*128*128
        out_pool = self.maxpool(out)                                # b*24*64*64

        source0 = self.avgpool(x)                                   # b*1*64*64
        out_pool_s = torch.cat((out_pool, source0), 1)              # b*25*64*64

        out1 = self.forw1(out_pool_s)                               # b*32*64*64
        out1_pool = self.maxpool(out1)                              # b*32*32*32

        source1 = self.avgpool(source0)                             # b*1*32*32
        out1_pool_s = torch.cat((out1_pool, source1), 1)            # b*33*32*32

        out2 = self.forw2(out1_pool_s)                              # b*64*32*32
        out2_pool = self.maxpool(out2)                              # b*64*16*16

        source2 = self.avgpool(source1)                             # b*1*16*16
        out2_pool_s = torch.cat((out2_pool, source2), 1)            # b*65*16*16

        out3 = self.forw3(out2_pool_s)                              # b*64*16*16
        out3_pool = self.maxpool(out3)                              # b*64*8*8

        source3 = self.avgpool(source2)                             # b*1*8*8
        out3_pool_s = torch.cat((out3_pool, source3), 1)            # b*64*8*8

        out4 = self.forw4(out3_pool_s)                              # b*64*8*8

        rev = self.postBlock(out4)                                  # b*8*2*2
        features = torch.squeeze(rev)
        features = features.view(-1, 32)                            # b*32
        out = self.classifier(features)

        if self.get_feature is False:
            return out
        return out, features


def get_FNet(class_num=2, get_feature=True):
    net = FNet(get_feature=get_feature)

    net.postBlock = nn.Sequential(
        nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True)
    )

    net.classifier = nn.Sequential(
        # nn.Dropout(0.5),
        nn.Linear(32, class_num),
        nn.Softmax()
    )
    return net


def test():
    net = get_FNet(class_num=10)

    gpu = "6"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = net.to(device)

    from torchsummary import summary
    summary(net, (1, 128, 128))

    inputs = torch.randn(7, 1, 128, 128)
    inputs = inputs.to(device)
    _, outputs = net(inputs)
    print()
    print(outputs.shape)


if __name__ == '__main__':
    test()

