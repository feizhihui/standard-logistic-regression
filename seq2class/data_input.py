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
                train_x.append(self.seq2onehot(line))
                train_y.append(label)
        self.train_x = np.array(train_x, dtype=np.int32)
        self.train_y = np.array(train_y, dtype=np.int32)
        self.datasize = len(self.train_y)
        print("trainset size:", self.datasize)

    def shuffle(self):
        mark = list(range(self.datasize))
        np.random.shuffle(mark)
        self.train_x = self.train_x[mark]
        self.train_y = self.train_y[mark]

    # AGCT=>0123
    def seq2onehot(self, line):
        seq_arr = np.zeros([40])
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


if __name__ == '__main__':
    DataMaster()
