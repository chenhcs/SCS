from src import preprocessing, transformer, postprocessing

def segment_cells(bin_file, image_file, align=None, patch_size=0, bin_size=3, n_neighbor=50, epochs=100, dia_estimate=15):
    if patch_size == 0:
        preprocessing.preprocess(bin_file, image_file, align, 0, 0, patch_size, bin_size, n_neighbor)
        transformer.train(0, 0, patch_size, epochs)
        postprocessing.postprocess(0, 0, patch_size, bin_size, dia_estimate)
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
        for startr in range(0, rmax, patch_size):
            for startc in range(0, cmax, patch_size):
                preprocessing.preprocess(bin_file, image_file, align, startr, startc, patch_size, bin_size, n_neighbor)
                transformer.train(startr, startc, patch_size, epochs)
                postprocessing.postprocess(startr, startc, patch_size, bin_size, dia_estimate)