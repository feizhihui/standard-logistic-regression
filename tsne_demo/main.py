# encoding=utf-8

import sys

for filename in sys.argv[1:]:
    with open(filename, 'r') as fileReader, open(filename + ".edgelist", 'w') as fileWriter:
        xymap = dict()
        count = 0
        for line in fileReader.readlines():
            x, y = line.split()
            if x in xymap:
                k1 = xymap[x]
            else:
                count += 1
                xymap[x] = count
                k1 = count
            if y in xymap:
                k2 = xymap[y]
            else:
                count += 1
                xymap[y] = count
                k2 = count
            fileWriter.write(str(k1) + "\t" + str(k2) + "\n")
