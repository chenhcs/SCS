from src import scs

for tile in ['2104', '2105', '2106', '2107']:
    bin_file = 'data/Mouse_liver_bin_' + tile + '.tsv'
    image_file = 'data/tile_' + tile + '.tif'
    scs.segment_cells(bin_file, image_file, dia_estimate=20)