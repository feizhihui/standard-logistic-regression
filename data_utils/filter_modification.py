# encoding=utf-8
import numpy as np

# filename = '../Data/ecoli_modifications.gff'
# cat_idx, seq_idx = 4, 10

filename = '../Data/lambda_modifications.gff'
cat_idx, seq_idx = 2, 8

with open(filename, 'r') as file, open(filename + ".txt", 'w') as fileWriter:
    for rowid, row in enumerate(file.readlines()):

        if rowid < 4:
            fileWriter.write(row)
            continue

        cols = row.split()
        cat, seq = cols[cat_idx], cols[seq_idx].split(";")[1][-41:]
        if seq[20] == "A" or seq[20] == "C":
            fileWriter.write(row)
        else:
            continue
