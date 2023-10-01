import numpy as np
from scipy.stats import pearsonr
import scanpy as sc
import anndata as ad
import pandas as pd
from scipy.stats import kruskal
from scipy.sparse import lil_matrix, csr_matrix, vstack
from sys import argv
import os
import matplotlib.pyplot as plt

script, bin_file, startx, starty, patchsize, nucl_seg, cell_seg1, cell_seg2 = argv

def bin2exp():
    #find xmin ymin
    xall = []
    yall = []
    df = pd.read_csv(bin_file, sep='\t')
    xmin = df['x'].min()
    ymin = df['y'].min()
    print(xmin, ymin)

    #find all the genes in the range
    geneid = {}
    genecnt = 0
    allgenes = []
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            if int(x) - xmin >= int(startx) and int(x) - xmin < int(startx) + int(patchsize) and int(y) - ymin >= int(starty) and int(y) - ymin < int(starty) + int(patchsize):
                if gene not in geneid:
                    geneid[gene] = genecnt
                    allgenes.append(gene)
                    genecnt += 1
    print(genecnt)

    posexp = lil_matrix((int(patchsize) * int(patchsize), genecnt), dtype=np.int8)
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            if gene not in geneid:
                continue
            if int(x) - xmin >= int(startx) and int(x) - xmin < int(startx) + int(patchsize) and int(y) - ymin >= int(starty) and int(y) - ymin < int(starty) + int(patchsize):
                idx = int((int(x) - xmin - int(startx)) * int(patchsize) + (int(y) - ymin - int(starty)))
                posexp[idx, geneid[gene]] = int(count)

    posexp = posexp.tocsr()

    return posexp, genecnt, allgenes

watershed_nucleus = {}
with open(nucl_seg) as fr:
    for line in fr:
        coord, cell = line.split()
        if cell not in watershed_nucleus:
            watershed_nucleus[cell] = [coord]
        else:
            watershed_nucleus[cell].append(coord)

seg2_cell = {}
with open(cell_seg2) as fr:
    for line in fr:
        coord, cell = line.split()
        if cell not in seg2_cell:
            seg2_cell[cell] = [coord]
        else:
            seg2_cell[cell].append(coord)

seg1_cell = {}
with open(cell_seg1) as fr:
    for line in fr:
        coord, cell = line.split()
        if cell not in seg1_cell:
            seg1_cell[cell] = [coord]
        else:
            seg1_cell[cell].append(coord)

nucleus2cell1= {}
nucleus2cell2 = {}

for nu in watershed_nucleus:
    mapcell = 0
    intsc_ze = 0
    for cell in seg1_cell:
        intsc = set(watershed_nucleus[nu]).intersection(seg1_cell[cell])
        if len(intsc) > intsc_ze:
            intsc_ze = len(intsc)
            mapcell = cell
    if mapcell:
        nucleus2cell1[nu] = mapcell

for nu in watershed_nucleus:
    mapcell = 0
    intsc_ze = 0
    for cell in seg2_cell:
        intsc = set(watershed_nucleus[nu]).intersection(seg2_cell[cell])
        if len(intsc) > intsc_ze:
            intsc_ze = len(intsc)
            mapcell = cell
    if mapcell:
        nucleus2cell2[nu] = mapcell

posexp, genecnt, allgenes = bin2exp()
allgenes = np.array(allgenes)
corr_seg1 = []
corr_seg2 = []
for nu in nucleus2cell1:
    if nu in nucleus2cell2:
        int_part = set(seg1_cell[nucleus2cell1[nu]]).intersection(seg2_cell[nucleus2cell2[nu]])
        diff_seg1 = set(seg1_cell[nucleus2cell1[nu]]).difference(int_part)
        diff_seg2 = set(seg2_cell[nucleus2cell2[nu]]).difference(int_part)
        exp_int = np.zeros(genecnt)
        exp_seg1 = np.zeros(genecnt)
        exp_seg2 = np.zeros(genecnt)
        for bin in int_part:
            idx = int(bin.split(':')[0]) * int(patchsize) + int(bin.split(':')[1])
            exp_int += np.squeeze(posexp[idx].toarray())
        for bin in diff_seg1:
            idx = int(bin.split(':')[0]) * int(patchsize) + int(bin.split(':')[1])
            exp_seg1 += np.squeeze(posexp[idx].toarray())
        for bin in diff_seg2:
            idx = int(bin.split(':')[0]) * int(patchsize) + int(bin.split(':')[1])
            exp_seg2 += np.squeeze(posexp[idx].toarray())

        if np.sum(exp_int) >= 100 and np.sum(exp_seg1) >= 100 and np.sum(exp_seg2) >= 100:
            r, p = pearsonr(exp_int, exp_seg1)
            if np.isnan(r):
                corr_seg1.append(0)
            else:
                corr_seg1.append(r)
            r, p = pearsonr(exp_int, exp_seg2)
            if np.isnan(r):
                corr_seg2.append(0)
            else:
                corr_seg2.append(r)
        print(nu, len(int_part), len(diff_seg1), len(diff_seg2), (nucleus2cell1[nu], len(seg1_cell[nucleus2cell1[nu]])), (nucleus2cell2[nu], len(seg2_cell[nucleus2cell2[nu]])))
        print('mean corr_seg1:', np.mean(corr_seg1), 'mean corr_seg2:', np.mean(corr_seg2), 'median corr_seg1:', np.median(corr_seg1), 'median corr_seg2:', np.median(corr_seg2))

import numpy as np
import matplotlib.pyplot as plt

data = [corr_seg1, corr_seg2]
_, p = kruskal(corr_seg1, corr_seg2)
print(p, len(corr_seg1))
fig, ax = plt.subplots()
ax.boxplot(data)

plt.savefig('results/r_boxplt_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.png')
