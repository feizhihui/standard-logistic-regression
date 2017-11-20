# encoding=utf-8
import numpy as np


class DataMaster(object):
    def __init__(self):
        train_x = []
        train_y = []
        with open("../Data/ecoli_modifications.gff") as file:
            for rowid, line in enumerate(file.readlines()[4:]):
                columns = line.split()
                line = columns[10]
                line = line.split(";")[1]
                line = line[-40:]
                category = columns[4]
                if category == "m6A":
                    label = 0
                elif category == "m4C":
                    label = 1
                elif category == "modified_base":
                    label = 2
                else:
                    raise BaseException("exception in category", category)
                print(label)
                train_x.append(line)
                train_y.append(label)
        self.datasize = len(train_y)
        print("trainset size:", self.datasize)

    def shuffle(self):
        mark = list(range(self.datasize))
        np.random.shuffle(mark)
        self.train_x = self.train_x[mark]
        self.train_y = self.train_y[mark]

    # AGCT=>0123
    def seq2onehot(self, seqs):
        seq_arr = np.zeros([len(seqs, 40)])
        for i, seq in enumerate(seqs):
            for j, c in enumerate(seq):
                if c == 'A':
                    seq_arr[i, j] = 0
                elif c == 'G':
                    seq_arr[i, j] = 1
                elif c == 'C':
                    seq_arr[i, j] = 2
                elif c == 'T':
                    seq_arr[i, j] = 3


if __name__ == '__main__':
    DataMaster()
