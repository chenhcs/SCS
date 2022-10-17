from src import preprocessing, transformer, postprocessing

fw = open('data/Mouse_brain_Adult_GEM_bin1_sub.tsv', 'w')
# Take one patch from the whole mouse brain section
with open('data/Mouse_brain_Adult_GEM_bin1.tsv') as fr:
    xmin = 3225
    ymin = 6175
    header = fr.readline()
    fw.write(header)
    for line in fr:
        gene, x, y, count = line.split()
        if int(x) - xmin >= 5700 and int(x) - xmin < 6900 and int(y) - ymin >= 5700 and int(y) - ymin < 6900:
            fw.write(gene + '\t' + str(int(x) - 5700 - xmin) + '\t' + str(int(y) - 5700 - ymin) + '\t' + count + '\n')
fw.close()

bin_file = 'data/Mouse_brain_Adult_GEM_bin1_sub.tsv'
image_file = 'data/Mouse_brain_Adult_sub.tif'
preprocessing.preprocess(bin_file, image_file)
transformer.train()
postprocessing.postprocess()
