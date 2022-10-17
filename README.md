# SCS
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SCS (Sub-cellular spatial transcriptomics Cell Segmentation) is a method that combines sequencing and staining data to accurately identify cell boundaries from high-resolution spatial transcriptomics.

## System requirements
### Operating system
The software has been tested on the CentOS Linux 7 system.

### Software requirements
- python 3.9.7</br>
- anndata 0.7.5</br>
- numpy 1.22.4</br>
- scanpy 1.8.2</br>
-

## Usage
To run SCS on one mouse adult brain section generated from the Stereo-seq platform, download the [Mouse_brain_Adult_GEM_bin1.tsv.gz](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/Bin1_matrix/Mouse_brain_Adult_GEM_bin1.tsv.gz) file from the [MOSTA](https://db.cngb.org/stomics/mosta/download.html) data portal and save it to the data folder, then run the following command from the project home directory:
```
python main.py
```

## Credits
The software is an implementation of the method SCS, jointly developed by Hao Chen, Dongshunyi Li, and Ziv Bar-Joseph from the [System Biology Group @ Carnegie Mellon University](http://sb.cs.cmu.edu/).

## Contact
Contact us if you have any questions:</br>
Hao Chen: hchen4 at andrew.cmu.edu</br>
Ziv Bar-Joseph: zivbj at andrew.cmu.edu</br>

## License
This project is licensed under the MIT License - see the LICENSE file for details.
