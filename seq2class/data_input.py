# encoding=utf-8


class DataMaster(object):
    def __init__(self):
        trainset = []
        trainlabels = []
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
                trainset.append(line)
                trainlabels.append(label)
        print("trainset size:", len(trainset))

    pass


if __name__ == '__main__':
    DataMaster()
