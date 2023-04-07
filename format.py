import sys
import os
import csv
import scipy
import math
import numpy as np
import pandas as pd

geoName = "GSM5212844_Liver"
parent_folder = "data/"
NameSeq1st = "MiSeq"

# Read in coordinates for barcodes
filename = "MiSeq-DraI-100pM-mbcore-RD2-revHDMIs-pos-uniq.txt"
miseq_pos = pd.read_csv(os.path.join(parent_folder, filename), delim_whitespace=True, header=None)
miseq_pos.columns = ['HDMI', 'tile','x','y']
print(miseq_pos.shape)

#Read featrures.tsv
gene_names = [row[1] for row in csv.reader(open(os.path.join(parent_folder, geoName +  "_SeqScope_features.tsv")), delimiter="\t")]
gene_names = np.array(gene_names)

#Read barcodes.tsv
bc = []
with open(os.path.join(parent_folder, geoName + "_SeqScope_barcode.tsv")) as f:
    reader = csv.reader(f, delimiter='\t', quotechar='"')
    for row in reader:
        if row:
            bc.append(row[0])
bc2idx = {c:i for i,c in enumerate(bc)}

exp = scipy.io.mmread(os.path.join(parent_folder, geoName + "_SeqScope_matrix.mtx.gz"))
print(exp.shape)
exp = exp.tocsr()

for i in [2104, 2105, 2106, 2107]:
    fw = open('data/Mouse_liver_bin_' + str(i) + '_ori.tsv', 'w')
    fw.write('geneID\tx\ty\tMIDCounts\n')
    print('tile', i)
    selected = miseq_pos[miseq_pos.tile.eq(i)]
    print(len(selected))
    print(selected['x'])
    x = list(selected['x'])
    y = list(selected['y'])
    hdmi = list(selected['HDMI'])
    bcidx = []
    xidx = []
    yidx = []
    for j,c in enumerate(hdmi):
        if j % 10000 == 0:
            print(j, c, len(hdmi))
        if c not in bc2idx:
            continue
        bcidx.append(bc2idx[c])
        xidx.append(x[j])
        yidx.append(y[j])

    bcidx = np.array(bcidx)
    genenzidx, bcnzidx = exp[:, bcidx].nonzero()
    for j in range(len(bcnzidx)):
        if j % 10000 == 0:
            print(j, len(bcnzidx))
        geneID = gene_names[genenzidx[j]]
        count = exp[genenzidx[j], bcidx[bcnzidx[j]]]
        fw.write(geneID + '\t' + str(xidx[bcnzidx[j]]) + '\t' + str(yidx[bcnzidx[j]]) + '\t' + str(count) + '\n')
    fw.close()

    maxx = 0
    with open('data/Mouse_liver_bin_' + str(i) + '_ori.tsv') as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, cnt = line.split()
            if int(x) > maxx:
                maxx = int(x)

    fw = open('data/Mouse_liver_bin_' + str(i) + '.tsv', 'w')
    xyg2c = {}
    with open('data/Mouse_liver_bin_' + str(i) + '_ori.tsv') as fr:
        header = fr.readline()
        fw.write(header)
        for line in fr:
            gene, x, y, cnt = line.split()
            x = str(math.floor((maxx - int(x)) / 15))
            y = str(math.floor(int(y) / 15))
            if x+':'+y+':'+gene not in xyg2c:
                xyg2c[x+':'+y+':'+gene] = int(cnt)
            else:
                xyg2c[x+':'+y+':'+gene] += int(cnt)

    for key in xyg2c:
        x, y, gene = key.split(':')
        fw.write('\t'.join([gene, x, y, str(xyg2c[key])]) + '\n')
    fw.close()
