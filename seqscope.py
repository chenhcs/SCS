from src import scs

for tile in ['2104', '2105', '2106', '2107']:
    bin_file = 'data/Mouse_liver_bin_' + tile + '.tsv'
    image_file = 'data/tile_' + tile + '.tiff'
    scs.segment_cells(bin_file, image_file, prealigned=True, r_estimate=20)
