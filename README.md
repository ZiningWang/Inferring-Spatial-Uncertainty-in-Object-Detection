
# Inferring Spatial Uncertainty in Object Detection

A teaser version of the code for the paper [**Labels Are Not Perfect: Inferring Spatial Uncertainty in Object Detection**](https://arxiv.org/pdf/2012.12195.pdf).

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

## LICENSE
### BSD License – Berkeley

Copyright (c) 2020 [Zining Wang](https://github.com/ZiningWang)

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the <organization> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
