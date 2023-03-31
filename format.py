import sys
import os
import csv
import scipy
import numpy as np
import pandas as pd

geoName = "GSM5212844_Liver"
parent_folder = "data/"
NameSeq1st = "MiSeq"

# Read in coordinates for barcodes
filename = "coors_liver.txt"
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

exp = scipy.io.mmread(os.path.join(parent_folder, geoName + "_SeqScope_matrix.mtx"))
print(exp.shape)
exp = exp.tocsr()

for i in [2104, 2105, 2106, 2107]:
    fw = open('data/Mouse_liver_bin_' + str(i) + '.tsv', 'w')
    fw.write('geneID\trow\tcolumn\tCount\n')
    print(i)
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
        print(j, c, len(hdmi))
        if c not in bc2idx:
            continue
        bcidx.append(bc2idx[c])
        xidx.append(x[j])
        yidx.append(y[j])

    bcidx = np.array(bcidx)
    genenzidx, bcnzidx = exp[:, bcidx].nonzero()
    for j in range(len(bcnzidx)):
        print(j, len(bcnzidx))
        geneID = gene_names[genenzidx[j]]
        count = exp[genenzidx[j], bcidx[bcnzidx[j]]]
        fw.write(geneID + '\t' + str(xidx[bcnzidx[j]]) + '\t' + str(yidx[bcnzidx[j]]) + '\t' + str(count) + '\n')
    fw.close()
