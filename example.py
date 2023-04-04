from src import scs

bin_file = 'data/Mouse_brain_Adult_GEM_bin1_sub.tsv'
image_file = 'data/Mouse_brain_Adult_sub.tif'
scs.segment_cells(bin_file, image_file, align='rigid')
