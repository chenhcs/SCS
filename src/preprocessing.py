import spateo as st
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import math
import os
from scipy.sparse import lil_matrix, csr_matrix, vstack


def preprocess(bin_file, image_file):
    #read data
    adatasub = st.io.read_bgi_agg(bin_file, image_file)
    print(adatasub.shape)
    adatasub.layers['unspliced'] = adatasub.X
    patchsizex = adatasub.X.shape[0]
    patchsizey = adatasub.X.shape[1]

    #align staining image with bins
    before = adatasub.layers['stain'].copy()
    st.cs.refine_alignment(adatasub, mode='rigid', transform_layers=['stain'])

    #nucleus segmentation from staining image
    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    st.cs.mask_nuclei_from_stain(adatasub, otsu_classes = 4, otsu_index=1)
    st.pl.imshow(adatasub, 'stain_mask', ax = ax)

    try:
        os.mkdir('fig/')
    except FileExistsError:
        print('fig folder exists')
    plt.savefig('fig/stain_mask.png')

    st.cs.find_peaks_from_mask(adatasub, 'stain', 7)
    st.cs.watershed(adatasub, 'stain', 5, out_layer='watershed_labels')

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    st.pl.imshow(adatasub, 'stain', save_show_or_return='return', ax=ax)
    st.pl.imshow(adatasub, 'watershed_labels', labels=True, alpha=0.5, ax=ax)
    plt.savefig('fig/watershed_labels.png')
    #adatasub.write('data/Mouse_brain_Adult_5800:8000:900:900.h5ad')
    print(adatasub)

    adatasub.write('data/bins.h5ad')

    #prepare data for neural network
    print(adatasub.layers['watershed_labels'])
    print(np.max(set(adatasub.layers['watershed_labels'].reshape((1, -1))[0])))
    print(adatasub.uns['spatial'])

    watershed2x = {}
    watershed2y = {}
    for i in range(adatasub.layers['watershed_labels'].shape[0]):
        for j in range(adatasub.layers['watershed_labels'].shape[1]):
            if adatasub.layers['watershed_labels'][i, j] == 0:
                continue
            if adatasub.layers['watershed_labels'][i, j] in watershed2x:
                watershed2x[adatasub.layers['watershed_labels'][i, j]].append(i)
                watershed2y[adatasub.layers['watershed_labels'][i, j]].append(j)
            else:
                watershed2x[adatasub.layers['watershed_labels'][i, j]] = [i]
                watershed2y[adatasub.layers['watershed_labels'][i, j]] = [j]

    watershed2center = {}
    sizes = []
    for nucleus in watershed2x:
        watershed2center[nucleus] = [np.mean(watershed2x[nucleus]), np.mean(watershed2y[nucleus])]
        sizes.append(len(watershed2x[nucleus]))
    print(np.min(sizes), np.max(sizes), np.mean(sizes))
    print('#nucleus', len(watershed2center))

    #find xmin ymin
    xall = []
    yall = []
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            xall.append(int(x))
            yall.append(int(y))
    xmin = np.min(xall)
    ymin = np.min(yall)
    print(np.min(xall), np.min(yall), np.max(xall), np.max(yall))

    #find all the genes in the range
    geneid = {}
    genecnt = 0
    id2gene = {}
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            if gene not in geneid:
                geneid[gene] = genecnt
                id2gene[genecnt] = gene
                genecnt += 1

    idx2exp = {}
    down = 3
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            if gene not in geneid:
                continue
            idx = int(math.floor((int(x) - xmin) / down) * math.ceil(patchsizey / down) + math.floor((int(y) - ymin) / down))
            if idx not in idx2exp:
                idx2exp[idx] = {}
                idx2exp[idx][geneid[gene]] = int(count)
            elif geneid[gene] not in idx2exp[idx]:
                idx2exp[idx][geneid[gene]] = int(count)
            else:
                idx2exp[idx][geneid[gene]] += int(count)

    all_exp_merged_bins = lil_matrix((int(math.ceil(patchsizex / down) * math.ceil(patchsizey / down)), genecnt), dtype=np.int8)
    for idx in idx2exp:
        for gid in idx2exp[idx]:
            all_exp_merged_bins[idx, gid] = idx2exp[idx][gid]
            print(idx, gid, idx2exp[idx][gid])
    all_exp_merged_bins = all_exp_merged_bins.tocsr()
    print(all_exp_merged_bins.shape)

    all_exp_merged_bins_ad = ad.AnnData(
        all_exp_merged_bins,
        obs=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[0])]),
        var=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[1])]),
    )
    sc.pp.highly_variable_genes(all_exp_merged_bins_ad, n_top_genes=2000, flavor='seurat_v3', span=1.0)
    selected_index = all_exp_merged_bins_ad.var[all_exp_merged_bins_ad.var.highly_variable].index
    selected_index = list(selected_index)
    selected_index = [int(i) for i in selected_index]
    #selected_index = geneidx[selected_index]
    print(selected_index, len(selected_index))

    with open('../data/variable_genes.txt', 'w') as fw:
        for id in selected_index:
            fw.write(id2gene[id] + '\n')

    #check total gene counts
    all_exp_merged_bins = all_exp_merged_bins.toarray()[:, selected_index]
    print(all_exp_merged_bins.shape)
    cell_bins = {}
    x_train_tmp = []
    x_train = []
    x_train_pos = []
    y_train = []
    y_binary_train = []
    x_train_bg_tmp = []
    x_train_bg = []
    x_train_pos_bg = []
    y_train_bg = []
    y_binary_train_bg = []
    x_validate_tmp = []
    x_validate= []
    x_validate_pos = []
    for i in range(adatasub.layers['watershed_labels'].shape[0]):
        for j in range(adatasub.layers['watershed_labels'].shape[1]):
            if (not i % down == 0) or (not j % down == 0):
                continue
            idx = int(math.floor(i / down) * math.ceil(patchsize / down) + math.floor(j / down))
            if adatasub.layers['watershed_labels'][i, j] > 0:
                if idx >= 0 and idx < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx, :]) > 0:
                    adatasum = 0
                    for r in range(i, i + down):
                        for c in range(j, j + down):
                            if r < adatasub.X.shape[0] and c < adatasub.X.shape[1]:
                                adatasum += adatasub.X[r,c]
                    print(np.sum(all_exp_merged_bins[idx, :]), adatasum)
                    x_train_sample = [all_exp_merged_bins[idx, :]]
                    x_train_pos_sample = [[i, j]]
                    y_train_sample = [watershed2center[adatasub.layers['watershed_labels'][i, j]]]
                    for dis in range(1, 11):
                        for dx in range(-45, 46):
                            for dy in range(-45, 46):
                                if len(x_train_sample) == 50:
                                    break
                                if not ((np.abs(dx) == dis * down and np.abs(dy) <= dis * down) or (np.abs(dx) <= dis * down and np.abs(dy) == dis * down)):
                                    continue
                                x = i + dx
                                y = j + dy
                                if (not x % down == 0) or (not y % down == 0):
                                    continue
                                if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= adatasub.layers['watershed_labels'].shape[1]:
                                    continue
                                idx_nb = int(math.floor(x / down) * math.ceil(patchsize / down) + math.floor(y / down))
                                if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx_nb, :]) > 0:
                                    x_train_sample.append(all_exp_merged_bins[idx_nb, :])
                                    x_train_pos_sample.append([x, y])
                    if len(x_train_sample) < 50:
                        continue
                    x_train_tmp.append(x_train_sample)
                    if len(x_train_tmp) > 500:
                        x_train.extend(x_train_tmp)
                        x_train_tmp = []
                        print(np.array(x_train).shape)
                    #print(x_train)
                    print(len(x_train_tmp))
                    x_train_pos.append(x_train_pos_sample)
                    y_train.append(y_train_sample)
                    y_binary_train.append(1)
            else:
                if idx >= 0 and idx < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx, :]) > 0:
                    backgroud = True
                    for nucleus in watershed2center:
                        if (i - watershed2center[nucleus][0]) ** 2 + (j - watershed2center[nucleus][1]) ** 2 <= 900 or adatasub.layers['stain'][i, j] > 10:
                            backgroud = False
                            break
                    if backgroud:
                        if len(x_train_bg) + len(x_train_bg_tmp) >= len(x_train) + len(x_train_tmp):
                            continue
                        x_train_sample = [all_exp_merged_bins[idx, :]]
                        x_train_pos_sample = [[i, j]]
                        y_train_sample = [[-1, -1]]
                        for dis in range(1, 11):
                            for dx in range(-45, 46):
                                for dy in range(-45, 46):
                                    if len(x_train_sample) == 50:
                                        break
                                    if not ((np.abs(dx) == dis * down and np.abs(dy) <= dis * down) or (np.abs(dx) <= dis * down and np.abs(dy) == dis * down)):
                                        continue
                                    x = i + dx
                                    y = j + dy
                                    if (not x % down == 0) or (not y % down == 0):
                                        continue
                                    if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= adatasub.layers['watershed_labels'].shape[1]:
                                        continue
                                    idx_nb = int(math.floor(x / down) * math.ceil(patchsize / down) + math.floor(y / down))
                                    if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx_nb, :]) > 0:
                                        x_train_sample.append(all_exp_merged_bins[idx_nb, :])
                                        x_train_pos_sample.append([x, y])
                        if len(x_train_sample) < 50:
                            continue
                        x_train_bg_tmp.append(x_train_sample)
                        if len(x_train_bg_tmp) > 500:
                            x_train_bg.extend(x_train_bg_tmp)
                            x_train_bg_tmp = []
                        print(len(x_train_bg_tmp))
                        x_train_pos_bg.append(x_train_pos_sample)
                        y_train_bg.append(y_train_sample)
                        y_binary_train_bg.append(0)
                        #print(np.sum(posexp[str(i) + ':' + str(j)]), adatasub.X[i,j])
                    else:
                        x_validate_sample = [all_exp_merged_bins[idx, :]]
                        x_validate_pos_sample = [[i, j]]
                        for dis in range(1, 11):
                            for dx in range(-45, 46):
                                for dy in range(-45, 46):
                                    if len(x_validate_sample) == 50:
                                        break
                                    if not ((np.abs(dx) == dis * down and np.abs(dy) <= dis * down) or (np.abs(dx) <= dis * down and np.abs(dy) == dis * down)):
                                        continue
                                    x = i + dx
                                    y = j + dy
                                    if (not x % down == 0) or (not y % down == 0):
                                        continue
                                    exp_merge = np.zeros(len(selected_index))
                                    if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= adatasub.layers['watershed_labels'].shape[1]:
                                        continue
                                    idx_nb = int(math.floor(x / down) * math.ceil(patchsize / down) + math.floor(y / down))
                                    if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx_nb, :]) > 0:
                                        x_validate_sample.append(all_exp_merged_bins[idx_nb, :])
                                        x_validate_pos_sample.append([x, y])
                        if len(x_validate_sample) < 50:
                            continue
                        x_validate_tmp.append(x_validate_sample)
                        if len(x_validate_tmp) > 500:
                            x_validate.extend(x_validate_tmp)
                            x_validate_tmp = []
                        x_validate_pos.append(x_validate_pos_sample)#
    x_train.extend(x_train_tmp)
    x_train_bg.extend(x_train_bg_tmp)
    x_validate.extend(x_validate_tmp)

    x_train = np.array(x_train)
    x_train_pos = np.array(x_train_pos)
    y_train = np.vstack(y_train)
    y_binary_train = np.array(y_binary_train)
    x_train_bg = np.array(x_train_bg)
    x_train_pos_bg = np.array(x_train_pos_bg)
    y_train_bg = np.vstack(y_train_bg)
    y_binary_train_bg = np.array(y_binary_train_bg)
    print(x_train.shape, x_train_pos.shape, y_train.shape, y_binary_train.shape, x_train_bg.shape, x_train_pos_bg.shape, y_train_bg.shape, y_binary_train_bg.shape)

    bg_index = np.arange(len(x_train_bg))
    np.random.shuffle(bg_index)
    x_train = np.vstack((x_train, x_train_bg[bg_index[:len(x_train)]]))
    x_train_pos = np.vstack((x_train_pos, x_train_pos_bg[bg_index[:len(x_train_pos)]]))
    y_train = np.vstack((y_train, y_train_bg[bg_index[:len(y_train)]]))
    y_binary_train = np.hstack((y_binary_train, y_binary_train_bg[bg_index[:len(y_binary_train)]]))
    print(x_train.shape, x_train_pos.shape, y_train.shape, y_binary_train.shape)

    x_validate= np.array(x_validate)
    x_validate_pos = np.array(x_validate_pos)
    print(x_validate.shape, x_validate_pos.shape)

    np.savez_compressed('../data/x_train.npz', x_train=x_train)
    np.savez_compressed('../data/x_train_pos.npz', x_train_pos=x_train_pos)
    np.savez_compressed('../data/y_train.npz', y_train=y_train)
    np.savez_compressed('../data/y_binary_train.npz', y_binary_train=y_binary_train)
    np.savez_compressed('../data/x_validate_brain.npz', x_validate=x_validate)
    np.savez_compressed('../data/x_validate_pos.npz', x_validate_pos=x_validate_pos)
