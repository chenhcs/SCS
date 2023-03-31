from src import scs

bin_file = 'data/Mouse_brain_Adult_GEM_bin1.tsv'
image_file = 'data/Mouse_brain_Adult.tif'
scs.segment_cells(bin_file, image_file, align='rigid', patch_size=1200)
