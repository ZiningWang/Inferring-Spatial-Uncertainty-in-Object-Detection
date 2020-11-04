
# Inferring Spatial Uncertainty in Object Detection

Dirty version of the code for the paper [**Inferring Spatial Uncertainty in Object Detection**](https://arxiv.org/pdf/2003.03644.pdf).

## Usage
### C++ evaluation first
It contains a cpp file used for kitti offline evaluation, compile it with

`g++ -o evaluate_object_3d_offline  evaluate_object_3d_offline.cpp -lboost_system -lboost_filesystem`

and use it same as the original kitti evaluation

`./evaluate_object_3d_offline ~/Kitti/object/training/label_2/ [your-path-to-predictions-in-KITTI-format]`

It will generate files hack files at the result folder

### Python visualization after evaluation

Use the python file `write_JIOU.py` to print results. Change the paths in the file before usage.

There are several implicit parameters:
* If "waymo" exists in the prediction data path, it will read the data in a format different from KITTI.
* If "unc" exists in the prediction data path, it will try to read the probabilistic prediction in the format of the paper [**Hujie20**](https://arxiv.org/pdf/2006.12015.pdf)
* If "uncertainty"  exists in the prediction data path, it will try to read the probabilistic prediction in the format of ProbPIXOR proposed in paper [**Di19**](https://arxiv.org/pdf/1909.12358.pdf)

## Hints
The IO of probabilistic prediction result is a disaster. The data format are not unified and vary from paper to paper. We have two kinds of  