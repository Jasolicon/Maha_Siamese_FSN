""" Metric Learning related
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class MNet(nn.Module):
    def __init__(self, dim):
        super(MNet, self).__init__()
        self.dimension = dim
        self.w_matrix = nn.Parameter(torch.zeros(dim, dim))         # dimension of features
        self.reset_parameters()                                     # initialize parameters

    def reset_parameters(self):                                     # Identity matrix
        nn.init.zeros_(self.w_matrix)
        with torch.no_grad():
            for i in range(self.dimension):
                self.w_matrix[i][i] = 1

    def forward(self, features1, features2):
        diff = features1 - features2                                # (x - y)
        weighted_diff = diff.mm(self.w_matrix)                      # W^T(x - y)
        out = torch.tanh(torch.norm(weighted_diff, 2, dim=1))
        return out


def my_knn(distances, train_labels, query_labels, same_set, k, class_num, log_path, write_log):
    """                     My KNN Algorithm

    :param distances:       [q * t] train-query distances
    :param train_labels:    [t]
    :param query_labels:    [q]
    :param same_set:        True if training set and query set are the same.
    :param k:
    :param class_num:
    :param log_path:
    :param write_log:
    :return:
    """

    ct_t = train_labels.shape[0]
    ct_q = query_labels.shape[0]
    train_labels = train_labels.float()
    query_labels = query_labels.float()
    result = torch.zeros([ct_q])

    with torch.no_grad():

        # KNN
        if write_log:
            with open(logpath, mode='a', encoding='utf-8') as f:
                print('Distances:', file=f)

        for i in range(ct_q):
            distances_cur = {}
            for j in range(ct_t):
                distances_cur[j] = distances[i][j]
                if same_set and (i == j):
                    distances_cur[j] = 10

            # sort
            distances_cur = sorted(distances_cur.items(), key=lambda x: x[1])

            # count neighbors
            neighbors = torch.zeros([class_num], dtype=torch.int)
            for j in range(k):
                neighbors[train_labels[distances_cur[j][0]].long().item()] += 1

            # find the nearest neighbor
            nearest_ct = 0
            nearest = 0
            for j in range(class_num):
                if neighbors[j] > nearest_ct:
                    nearest_ct = neighbors[j]
                    nearest = j
            result[i] = nearest

        acc_record = torch.eq(result.long(), query_labels.long())
        acc = float(acc_record.sum()) / ct_q

        if write_log:
            with open(logpath, mode='a', encoding='utf-8') as f:
                print('Prediction result:\n', result, '\n', file=f)

    return acc


class MyCriterion(nn.Module):
    def __init__(self, m=0.8, t=0.05):
        """
        :param m:   margin
        :param t:   threshold
        """
        super(MyCriterion, self).__init__()
        self.m = m
        self.t = t

    def forward(self, distances, labels1, labels2):
        """                 Loss = y * max(d, t) + (1 - y) * max(0, m - d)
                            y = 1 if same labels; y = 0 if different labels
        :param distances:   shape: [b]
        :param labels1:     shape: [b * 1]
        :param labels2:     shape: [b * 1]
        :return:            loss
        """

        distances = distances.reshape(-1)
        labels1 = labels1.reshape(-1)
        labels2 = labels2.reshape(-1)
        size = distances.shape[0]

        y = torch.eq(labels1, labels2).long()
        a = distances.clamp_min(self.t)
        b = (-distances + self.m).clamp_min(0)
        loss = y * a + (-y + 1) * b
        loss = loss.sum() / size

        return loss


if __name__ == '__main__':

    dim = 5
    batch_size = 4

    m_net = MNet(dim)
    m_net.train()
    feature1 = torch.randn((batch_size, dim))
    feature2 = torch.zeros((batch_size, dim))
    label1 = torch.randint(0, 3, [batch_size])
    label2 = torch.randint(0, 3, [batch_size])

    optimizer = optim.Adam(m_net.parameters(), lr=0.0001)

    # for p in m_net.parameters():
    #     if p.requires_grad:
    #         print(p.name, p.data)

    out = m_net(feature1, feature2)
    # out = torch.tensor([0.7, 0.03, 0.07, 0.9], dtype=torch.float)

    criterion = MyCriterion()
    loss = criterion(out, label1, label2)
    loss.backward()
    optimizer.step()

    # for p in m_net.parameters():
    #     if p.requires_grad:
    #         print(p.name, p.data)
