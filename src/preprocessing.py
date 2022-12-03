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
    #print(adatasub.shape)
    adatasub.layers['unspliced'] = adatasub.X
    patchsizex = adatasub.X.shape[0]
    patchsizey = adatasub.X.shape[1]

    #align staining image with bins
    before = adatasub.layers['stain'].copy()
    st.cs.refine_alignment(adatasub, mode='rigid', transform_layers=['stain'])

    fig, axes = plt.subplots(ncols=2, figsize=(16, 8), tight_layout=True)
    axes[0].imshow(before)
    st.pl.imshow(adatasub, 'unspliced', ax=axes[0], alpha=0.6, cmap='Reds', vmax=10, use_scale=False, save_show_or_return='return')
    axes[0].set_title('before alignment')
    st.pl.imshow(adatasub, 'stain', ax=axes[1], use_scale=False, save_show_or_return='return')
    st.pl.imshow(adatasub, 'unspliced', ax=axes[1], alpha=0.6, cmap='Reds', vmax=10, use_scale=False, save_show_or_return='return')
    axes[1].set_title('after alignment')

    try:
        os.mkdir('fig/')
    except FileExistsError:
        print('fig folder exists')

    plt.savefig('fig/alignment.png')

    #nucleus segmentation from staining image
    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    st.cs.mask_nuclei_from_stain(adatasub, otsu_classes = 4, otsu_index=1)
    st.pl.imshow(adatasub, 'stain_mask', ax = ax)

    plt.savefig('fig/stain_mask.png')

    st.cs.find_peaks_from_mask(adatasub, 'stain', 7)
    st.cs.watershed(adatasub, 'stain', 5, out_layer='watershed_labels')

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    st.pl.imshow(adatasub, 'stain', save_show_or_return='return', ax=ax)
    st.pl.imshow(adatasub, 'watershed_labels', labels=True, alpha=0.5, ax=ax)
    plt.savefig('fig/watershed_labels.png')
    #adatasub.write('data/Mouse_brain_Adult_5800:8000:900:900.h5ad')
    #print(adatasub)

    adatasub.write('data/spots.h5ad')

    print('Prepare data for neural network...')
    #print(adatasub.layers['watershed_labels'])
    #print(np.max(set(adatasub.layers['watershed_labels'].reshape((1, -1))[0])))
    #print(adatasub.uns['spatial'])

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
    #print(np.min(sizes), np.max(sizes), np.mean(sizes))
    #print('#nucleus', len(watershed2center))

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
    #print(np.min(xall), np.min(yall), np.max(xall), np.max(yall))

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
    downrs = 3
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            if gene not in geneid:
                continue
            idx = int(math.floor((int(x) - xmin) / downrs) * math.ceil(patchsizey / downrs) + math.floor((int(y) - ymin) / downrs))
            if idx not in idx2exp:
                idx2exp[idx] = {}
                idx2exp[idx][geneid[gene]] = int(count)
            elif geneid[gene] not in idx2exp[idx]:
                idx2exp[idx][geneid[gene]] = int(count)
            else:
                idx2exp[idx][geneid[gene]] += int(count)

    all_exp_merged_bins = lil_matrix((int(math.ceil(patchsizex / downrs) * math.ceil(patchsizey / downrs)), genecnt), dtype=np.int8)
    for idx in idx2exp:
        for gid in idx2exp[idx]:
            all_exp_merged_bins[idx, gid] = idx2exp[idx][gid]
            #print(idx, gid, idx2exp[idx][gid])
    all_exp_merged_bins = all_exp_merged_bins.tocsr()
    #print(all_exp_merged_bins.shape)

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
    #print(selected_index, len(selected_index))

    with open('data/variable_genes.txt', 'w') as fw:
        for id in selected_index:
            fw.write(id2gene[id] + '\n')

    #check total gene counts
    all_exp_merged_bins = all_exp_merged_bins.toarray()[:, selected_index]
    #print(all_exp_merged_bins.shape)
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
    x_test_tmp = []
    x_test= []
    x_test_pos = []
    offsets = []
    for dis in range(1, 11):
        for dy in range(-dis, dis + 1):
            offsets.append([-dis * downrs, dy * downrs])
        for dy in range(-dis, dis + 1):
            offsets.append([dis * downrs, dy * downrs])
        for dx in range(-dis + 1, dis):
            offsets.append([dx * downrs, -dis * downrs])
        for dx in range(-dis + 1, dis):
            offsets.append([dx * downrs, dis * downrs])
    for i in range(adatasub.layers['watershed_labels'].shape[0]):
        if (i + 1) % 100 == 0:
            print("finished {0:.0%}".format(i / adatasub.layers['watershed_labels'].shape[0]))
        for j in range(adatasub.layers['watershed_labels'].shape[1]):
            if (not i % downrs == 0) or (not j % downrs == 0):
                continue
            idx = int(math.floor(i / downrs) * math.ceil(patchsizey / downrs) + math.floor(j / downrs))
            if adatasub.layers['watershed_labels'][i, j] > 0:
                if idx >= 0 and idx < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx, :]) > 0:
                    x_train_sample = [all_exp_merged_bins[idx, :]]
                    x_train_pos_sample = [[i, j]]
                    y_train_sample = [watershed2center[adatasub.layers['watershed_labels'][i, j]]]
                    for dx, dy in offsets:
                        if len(x_train_sample) == 50:
                            break
                        x = i + dx
                        y = j + dy
                        if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= adatasub.layers['watershed_labels'].shape[1]:
                            continue
                        idx_nb = int(math.floor(x / downrs) * math.ceil(patchsizey / downrs) + math.floor(y / downrs))
                        if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx_nb, :]) > 0:
                            x_train_sample.append(all_exp_merged_bins[idx_nb, :])
                            x_train_pos_sample.append([x, y])
                    if len(x_train_sample) < 50:
                        continue
                    x_train_tmp.append(x_train_sample)
                    if len(x_train_tmp) > 500:
                        x_train.extend(x_train_tmp)
                        x_train_tmp = []
                        #print(np.array(x_train).shape)
                    #print(x_train)
                    #print(len(x_train_tmp))
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
                        for dx, dy in offsets:
                            if len(x_train_sample) == 50:
                                break
                            x = i + dx
                            y = j + dy
                            if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= adatasub.layers['watershed_labels'].shape[1]:
                                continue
                            idx_nb = int(math.floor(x / downrs) * math.ceil(patchsizey / downrs) + math.floor(y / downrs))
                            if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx_nb, :]) > 0:
                                x_train_sample.append(all_exp_merged_bins[idx_nb, :])
                                x_train_pos_sample.append([x, y])
                        if len(x_train_sample) < 50:
                            continue
                        x_train_bg_tmp.append(x_train_sample)
                        if len(x_train_bg_tmp) > 500:
                            x_train_bg.extend(x_train_bg_tmp)
                            x_train_bg_tmp = []
                        #print(len(x_train_bg_tmp))
                        x_train_pos_bg.append(x_train_pos_sample)
                        y_train_bg.append(y_train_sample)
                        y_binary_train_bg.append(0)
                        #print(np.sum(posexp[str(i) + ':' + str(j)]), adatasub.X[i,j])
                    else:
                        x_test_sample = [all_exp_merged_bins[idx, :]]
                        x_test_pos_sample = [[i, j]]
                        for dx, dy in offsets:
                            if len(x_test_sample) == 50:
                                break
                            x = i + dx
                            y = j + dy
                            exp_merge = np.zeros(len(selected_index))
                            if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= adatasub.layers['watershed_labels'].shape[1]:
                                continue
                            idx_nb = int(math.floor(x / downrs) * math.ceil(patchsizey / downrs) + math.floor(y / downrs))
                            if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx_nb, :]) > 0:
                                x_test_sample.append(all_exp_merged_bins[idx_nb, :])
                                x_test_pos_sample.append([x, y])
                        if len(x_test_sample) < 50:
                            continue
                        x_test_tmp.append(x_test_sample)
                        if len(x_test_tmp) > 500:
                            x_test.extend(x_test_tmp)
                            x_test_tmp = []
                        x_test_pos.append(x_test_pos_sample)#
    x_train.extend(x_train_tmp)
    x_train_bg.extend(x_train_bg_tmp)
    x_test.extend(x_test_tmp)

    x_train = np.array(x_train)
    x_train_pos = np.array(x_train_pos)
    y_train = np.vstack(y_train)
    y_binary_train = np.array(y_binary_train)
    x_train_bg = np.array(x_train_bg)
    x_train_pos_bg = np.array(x_train_pos_bg)
    y_train_bg = np.vstack(y_train_bg)
    y_binary_train_bg = np.array(y_binary_train_bg)
    #print(x_train.shape, x_train_pos.shape, y_train.shape, y_binary_train.shape, x_train_bg.shape, x_train_pos_bg.shape, y_train_bg.shape, y_binary_train_bg.shape)

    bg_index = np.arange(len(x_train_bg))
    np.random.shuffle(bg_index)
    x_train = np.vstack((x_train, x_train_bg[bg_index[:len(x_train)]]))
    x_train_pos = np.vstack((x_train_pos, x_train_pos_bg[bg_index[:len(x_train_pos)]]))
    y_train = np.vstack((y_train, y_train_bg[bg_index[:len(y_train)]]))
    y_binary_train = np.hstack((y_binary_train, y_binary_train_bg[bg_index[:len(y_binary_train)]]))
    #print(x_train.shape, x_train_pos.shape, y_train.shape, y_binary_train.shape)

    x_test= np.array(x_test)
    x_test_pos = np.array(x_test_pos)
    #print(x_test.shape, x_test_pos.shape)

    np.savez_compressed('data/x_train.npz', x_train=x_train)
    np.savez_compressed('data/x_train_pos.npz', x_train_pos=x_train_pos)
    np.savez_compressed('data/y_train.npz', y_train=y_train)
    np.savez_compressed('data/y_binary_train.npz', y_binary_train=y_binary_train)
    np.savez_compressed('data/x_test.npz', x_test=x_test)
    np.savez_compressed('data/x_test_pos.npz', x_test_pos=x_test_pos)
