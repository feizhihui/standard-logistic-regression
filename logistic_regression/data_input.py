# encoding=utf-8
import numpy as np


class DataMaster(object):
    def __init__(self, train_mode=True):
        if train_mode:
            filename = '../Data/modifications.csv'
        else:
            filename = '../Data/test.modifications.csv'
        with open(filename, 'r') as file:
            train_x, train_y = [], []
            for row in file.readlines()[1:]:
                cols = row.split(',')
                x1, x2, x3 = float(cols[5]), float(cols[6]), float(cols[7])
                y = 1 if int(cols[4]) > 20 else 0
                train_x.append([x1, x2, x3])
                train_y.append([y])
        print('Data input completed filename=', filename)
        self.datasets = np.array(train_x, dtype=np.float32)
        self.datalabels = np.array(train_y, dtype=np.int32)
        if train_mode:
            self.pos_idx = (self.datalabels == 1).reshape(-1)
            self.neg_idx = (self.datalabels == 0).reshape(-1)
            self.datasize = len(self.datalabels[self.pos_idx]) * 2
        else:
            self.datasize = len(self.datalabels)

    def shuffle(self):
        mark = list(range(self.datasize // 2))
        np.random.shuffle(mark)
        self.train_x = np.concatenate([self.datasets[self.pos_idx], self.datasets[self.neg_idx][mark]])
        self.train_y = np.concatenate([self.datalabels[self.pos_idx], self.datalabels[self.neg_idx][mark]])
        mark = list(range(self.datasize))
        np.random.shuffle(mark)
        self.train_x = self.train_x[mark]
        self.train_y = self.train_y[mark]


if __name__ == '__main__':
    DataMaster()
