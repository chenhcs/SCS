# SCS
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SCS (Sub-cellular spatial transcriptomics Cell Segmentation) is a method that combines sequencing and staining data to accurately identify cell boundaries from high-resolution spatial transcriptomics.

## System requirements
### Operating system
The software has been tested on the CentOS Linux 7 system.

### Software requirements
- python 3.9.7</br>
- anndata 0.7.5</br>
- matplotlib 3.5.0</br>
- numpy 1.22.4</br>
- pandas 1.3.4</br>
- scanpy 1.8.2</br>
- scikit-learn 1.0.1</br>
- scipy 1.7.2</br>
- [spateo](https://spateo-release.readthedocs.io/en/latest/installation.html)
- tensorflow 2.8.2</br>
- tensorflow_addons 0.16.1</br>

## Usage
This section describes how to use SCS to perform cell segmentation on the high-resolution spatial transcriptomics data.

An example is provided for one mouse adult brain section generated from the Stereo-seq platform. To run the example, download the [Mouse_brain_Adult_GEM_bin1.tsv.gz](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/Bin1_matrix/Mouse_brain_Adult_GEM_bin1.tsv.gz) file from the [MOSTA](https://db.cngb.org/stomics/mosta/download.html) data portal and save it to the `data` folder, the corresponding staining image data is already in the folder. Then run the following command from the project home directory:
```
python main.py
```

The script run three steps to segment the provided patch: (i) preprocessing, i.e., identifying nuclei and preparing data for the transformer, (ii) training the transformer and inference on all the spots in the patch, (iii), postprocessing, i.e., gradient flow tracking. The preprocessing time on the demo patch will be about 10 minutes, transformer training will take roughly 1.5 hour with an Nvidia GeForce 10 series graphics card, and the postprocessing will take about 5 minutes.

## Output
Results will be saved to `results` directory.

The output file `stain_mask.png` visualizes cell boundaries in the sequencing section.

The output file `spot2cell.txt` contains the mapping from spot coordinates to cell indexes.

Each line has the following format:
```
row:column  cell_id
```
where `row:column` is the coordinate of one spot indicating which row and column the spot is located in from the upper left corner, and `cell_id` is the index of the cell to which the spot belongs.

## Credits
The software is an implementation of the method SCS, jointly developed by Hao Chen, Dongshunyi Li, and Ziv Bar-Joseph from the [System Biology Group @ Carnegie Mellon University](http://sb.cs.cmu.edu/).

## Contact
Contact us if you have any questions:</br>
Hao Chen: hchen4 at andrew.cmu.edu</br>
Ziv Bar-Joseph: zivbj at andrew.cmu.edu</br>

## License
This project is licensed under the MIT License - see the LICENSE file for details.
