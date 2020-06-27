import sys
import os
import math
import numpy as np
sys.path.append(os.path.abspath('../'))
from utils.metric_utils import label_inference_BEV, label_inference_3D

import glob
filelist = []

test_frame = 4214 
file_lists = ["00{:04d}".format(test_frame)]
file_lists.sort()

def main():
	dr = 0
	gs = 0.2
	pscale = 0
	inference = label_inference_3D(gen_std=gs, degree_register=dr, prob_outlier=0.03)
	box3D = np.array([0,0,0,1.5,2,4,math.pi*0.05])
	points_clip_3D = np.array([[0,0,-1.1],[2.2,0,0],[2.2,0,-1.1],[0,-1.5,-1.1],[2.2,-1.5,0],[2.2,-1.5,-1.1]])
	uncertain_label = inference.infer(points_clip_3D, box3D, prior_scaler=pscale)
	covs = uncertain_label.calc_uncertainty_corners()
	print(covs)
	'''
	inference = label_inference_BEV(gen_std=gs, degree_register=dr, prob_outlier=0.03)
	boxBEV = box3D[[0,1,3,5,6]]
	points_clip_BEV = points_clip_3D[0:3,[0,2]]
	uncertain_label = inference.infer(points_clip_BEV, boxBEV, prior_scaler=pscale)
	centers, covs = uncertain_label.calc_uncertainty_corners()
	print(covs)
	'''
if __name__ == '__main__':
	main()
