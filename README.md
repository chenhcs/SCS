# SCS
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SCS (Sub-cellular spatial transcriptomics Cell Segmentation) is a method that combines sequencing and staining data to accurately identify cell boundaries from high-resolution spatial transcriptomics.

## System requirements
### Operating system
The software has been tested on the CentOS Linux 7 system.

### Software requirements
- python 3.9</br>
- anndata</br>
- matplotlib</br>
- numpy</br>
- pandas</br>
- scanpy</br>
- scikit-learn</br>
- scipy</br>
- tensorflow</br>
- tensorflow_addons</br>
- imagecodecs </br>
- scikit-misc </br>
- [spateo](https://spateo-release.readthedocs.io/en/latest/installation.html)


### Installation
It is recommended to create a virtual environment using [Conda](https://conda.io/projects/conda/en/latest/index.html). After successfully installing Anaconda/Miniconda, create an environment using the provided `environment.yml` file, then manually install the spateo package:
```
conda env create -f environment.yml
conda activate SCS
pip install spateo-release
```

## Usage
### Example
This section describes an example of how to use SCS to perform cell segmentation on the high-resolution spatial transcriptomics data.

An example is provided for one mouse adult brain section generated from the Stereo-seq platform. To run the example, download the [Mouse_brain_Adult_GEM_bin1.tsv.gz](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/Bin1_matrix/Mouse_brain_Adult_GEM_bin1.tsv.gz) file from the [MOSTA](https://db.cngb.org/stomics/mosta/download/) data portal and save it to the `data` folder under this project directory, then unzip the file by running the following command.
```
gunzip data/Mouse_brain_Adult_GEM_bin1.tsv.gz
```
The file of detected RNAs should follow the following format in a tab-delimited file:
```
geneID  row  column  counts
```
The corresponding staining image data is already in the `data` folder. Run the following script from the project home directory to take one patch from the whole section as an example:
```
python patch_cut.py
```
Then use the following python code to run SCS on the example patch or run the `example.py` script:
```
from src import scs

bin_file = 'data/Mouse_brain_Adult_GEM_bin1_sub.tsv'
image_file = 'data/Mouse_brain_Adult_sub.tif'
scs.segment_cells(bin_file, image_file, align='rigid')
```
Use `help(scs.segment_cells)` in python to see more instructions on the usages.

The `segment_cells` function will run three steps to segment the provided patch: (*i*) preprocessing, *i.e.*, identifying nuclei and preparing data for the transformer, (*ii*) training the transformer and inference on all the spots in the patch, (*iii*), postprocessing, *i.e.*, gradient flow tracking. The preprocessing time on the demo patch will be about 10 minutes, transformer training will take roughly 1 hour with an Nvidia GeForce 10 series graphics card, and the postprocessing will take about 5 minutes.

### Processing large-scale data
SCS can process large-scale spatial data by splitting the provided section into patches, and process the data patch by patch. This makes the prediction on very large datasets feasible on normal computers.

The example of running SCS on the whole mouse brain section of Stereo-seq is as follows. Before running the example, the transcriptomics data [Mouse_brain_Adult_GEM_bin1.tsv.gz](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/Bin1_matrix/Mouse_brain_Adult_GEM_bin1.tsv.gz) should be downloaded and saved to the `data` folder under this project directory and uncompressed. The corresponding image data [Mouse_brain_Adult.tif](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/Image/Mouse_brain_Adult.tif) should be downloaded and saved to the same `data` folder as well.

Next, run the following code or the `large_scale.py` script from the project home directory to run SCS on the whole mouse brain section, in which SCS will split the section into patches of size (patch_size) 1200 spots x 1200 spots, and make predictions patch by patch.
```
from src import scs

bin_file = 'data/Mouse_brain_Adult_GEM_bin1.tsv'
image_file = 'data/Mouse_brain_Adult.tif'
scs.segment_cells(bin_file, image_file, align='rigid', patch_size=1200)
```
The `patch_size` parameter controls how large one patch will be.

We also advise the users to save patches into separate files as done in the "Example" section and run SCS on patches parallelly on different CPUs/GPUs.

### Reproducing cell segmentations for the Stereo-seq and Seq-scope datasets
The cell segmentations for the whole Stereo-seq section can be generated following the instruction in the "Processing large-scale data" section.

Follow the instruction below to generate cell segmentations for the Seq-Scope mouse liver dataset. The Seq-Scope transcriptomics data can be downloaded from [GEO](https://www-ncbi-nlm-nih-gov.cmu.idm.oclc.org/geo/query/acc.cgi?acc=GSM5212844). Save the three files in the link to the `data` folder and unzip the `tsv.gz` files. The [file](https://deepblue.lib.umich.edu/data/downloads/g158bh60f) for coordinates of sequencing spots can be downloaded from [Deep Blue Data](https://deepblue.lib.umich.edu/data/concern/file_sets/g158bh60f). Save this file to the `data` folder as well. Then run the following script to convert data format. Or directly use the processed `.tsv` files saved in the `data` folder.
```
python format.py
```

The paired H&E images can be found at [Deep Blue Data](https://doi.org/10.7302/cjfe-wa35), the processed images corresponding to tiles 2104-2107 have already been saved to the `data` folder. Run the following script to make predictions for the four tiles (2104-2107) of the Seq-Scope data:
```
python seqscope.py
```

## Output
Results will be saved to `results` directory.

The output file `cell_masks.png` visualizes cell boundaries in the sequencing section.

The output file `spot2cell.txt` contains the mapping from spot coordinates to cell indexes.

Each line has the following format:
```
row:column  cell_id
```
where `row:column` is the coordinate of one spot indicating which row and column the spot is located in from the upper left corner, and `cell_id` is the index of the cell to which the spot belongs.

A statistical summary for the segmented cells `cell_stats.txt`, including the number of cells identified and cell size statistics, will be saved to the `results` directory.

## Credits
The software is an implementation of the method SCS, jointly developed by Hao Chen, Dongshunyi Li, and Ziv Bar-Joseph from the [System Biology Group @ Carnegie Mellon University](http://sb.cs.cmu.edu/).

## Contact
Contact us if you have any questions:</br>
Hao Chen: hchen4 at andrew.cmu.edu</br>
Ziv Bar-Joseph: zivbj at andrew.cmu.edu</br>

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you find SCS is useful for your research, please cite the following paper:

```
Chen, H., Li, D. & Bar-Joseph, Z.
SCS: cell segmentation for high-resolution spatial transcriptomics.
Nat Methods (2023). https://doi.org/10.1038/s41592-023-01939-3
```
