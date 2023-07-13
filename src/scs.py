import math
import numpy as np
from src import preprocessing, transformer, postprocessing

def segment_cells(bin_file, image_file, prealigned=False, align=None, patch_size=0, bin_size=3, n_neighbor=50, epochs=100, r_estimate=15, val_ratio=0.0625):
    """
    Parameters:
        bin_file - string, tsv file for detected RNAs
        image_file - string, staining image of the tissue (.tiff)
        prealigned - boolean, if the staining image is prealigned with the sequencing spots coordinates, default False
        align - string, alignment method used to align the input staining image and sequencing spots ('rigid', 'non-rigid' or 'None'), default None
        patch_size - int, length and width (spots) of patches, if greater than zero, the input section will be cut into patches and processed patch by patch, if zero, the section will be process as a whole, default 0
        bin size - int, the length and width (spots) of regions that will be merged as a bin, default 3
        n_neighbor - int, the number of nearest neighbors who will be considered when make predictions for one spot in the transformer model, default 50
        epochs - int, the training epochs of the transformer model, default 100
        r_estimate - int, the estimated radius (spots) of cells, used to calculate the priors for transformer predictions, default 15
        val_ratio - float, the fraction of the patch set aside for validation, default 0.0625 (1/4 height x 1/4 width)
    """
    if patch_size == 0:
        preprocessing.preprocess(bin_file, image_file, prealigned, align, 0, 0, patch_size, bin_size, n_neighbor)
        transformer.train(0, 0, patch_size, epochs, val_ratio)
        postprocessing.postprocess(0, 0, patch_size, bin_size, r_estimate)
    else:
        r_all = []
        c_all = []
        with open(bin_file) as fr:
            header = fr.readline()
            for line in fr:
                _, r, c, _ = line.split()
                r_all.append(int(r))
                c_all.append(int(c))
        rmax = np.max(r_all) - np.min(r_all)
        cmax = np.max(c_all) - np.min(c_all)
        n_patches = math.ceil(rmax / patch_size) * math.ceil(cmax / patch_size)
        print(str(n_patches) + ' patches will be processed.')
        for startr in range(0, rmax, patch_size):
            for startc in range(0, cmax, patch_size):
                try:
                    print('Processing the patch ' + str(startr) + ':' + str(startc) + '...')
                    preprocessing.preprocess(bin_file, image_file, prealigned, align, startr, startc, patch_size, bin_size, n_neighbor)
                    transformer.train(startr, startc, patch_size, epochs, val_ratio)
                    postprocessing.postprocess(startr, startc, patch_size, bin_size, r_estimate)
                except Exception as e:
                    print(e)
                    print('Patch ' + str(startr) + ':' + str(startc) + ' failed. This could be due to no nuclei detected by Watershed or too few RNAs in the patch.')
