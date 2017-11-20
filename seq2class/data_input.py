# encoding=utf-8
import numpy as np


class DataMaster(object):
    # ==============
    def __init__(self, train_mode=True):
        if train_mode:
            filename = '../Data/ecoli_modifications.gff'
        else:
            filename = '../Data/lambda_modifications.gff'

        with open(filename, 'r') as file:
            train_x, train_y, train_c = [], [], []
            for row in file.readlines()[4:]:
                cols = row.split()
                cat, seq = cols[4], cols[10].split(";")[1][-41:]
                # print(seq, cat)
                # assert seq[20] == "A" or seq[20] == "C", "Error:" + seq[20]
                train_x.append(self.seq2matrix(seq))
                if cat == "modified_base":
                    train_y.append(0)
                else:
                    train_y.append(1)
                train_c.append(cat)

        print('Data input completed filename=', filename)
        self.datasets = np.array(train_x, dtype=np.float32)
        self.datalabels = np.array(train_y, dtype=np.int32)
        self.datacat = np.array(train_y, dtype=np.str)
        print("availabel data numbers", str(len(self.datalabels)))
        if train_mode:
            self.pos_idx = (self.datalabels == 1).reshape(-1)
            self.neg_idx = (self.datalabels == 0).reshape(-1)
            self.datasize = len(self.datalabels[self.pos_idx]) * 2
            print("positive data numbers", str(self.datasize // 2))
        else:
            self.datasize = len(self.datalabels)

    # AGCT=>0123
    def seq2matrix(self, line):
        seq_arr = np.zeros([41])
        for j, c in enumerate(line):
            if c == 'A':
                seq_arr[j] = 0
            elif c == 'G':
                seq_arr[j] = 1
            elif c == 'C':
                seq_arr[j] = 2
            elif c == 'T':
                seq_arr[j] = 3
        return seq_arr

    def shuffle(self):
        mark = list(range(self.datasize // 2))
        np.random.shuffle(mark)
        self.train_x = np.concatenate([self.datasets[self.pos_idx], self.datasets[self.neg_idx][mark]])
        self.train_y = np.concatenate([self.datalabels[self.pos_idx], self.datalabels[self.neg_idx][mark]])
        self.train_c = np.concatenate([self.datacat[self.pos_idx], self.datacat[self.neg_idx][mark]])
        mark = list(range(self.datasize))
        np.random.shuffle(mark)
        self.train_x = self.train_x[mark]
        self.train_y = self.train_y[mark]
        self.train_c = self.train_c[mark]


if __name__ == '__main__':
    DataMaster()
