import sys
import os
import math
import numpy as np
sys.path.append(os.path.abspath('../'))
from utils.metric_utils import label_inference_BEV, label_inference_3D
import glob

def main():
	dr = 0        #registration degree (0 or 1 or 2)
	gs = 0.2      #LiDAR measurement noise standard deviation
	pscale = 0    #=0 means don't add prior distribution
	inference = label_inference_3D(gen_std=gs, degree_register=dr, prob_outlier=0.03) #initialize the inference algorithm
	box3D = np.array([0,0,0,1.5,2,4,math.pi*0.05]) #3D bounding box, [x,y_top,z,h,w,l,ry]
	points_clip_3D = np.array([[0,0,-1.1],[2.2,0,0],[2.2,0,-1.1],[0,-1.5,-1.1],[2.2,-1.5,0],[2.2,-1.5,-1.1]]) #3D points in the bbox
	#run inference and and covariance of 8 corners
	uncertain_label = inference.infer(points_clip_3D, box3D, prior_scaler=pscale)
	covs = uncertain_label.calc_uncertainty_corners()
	print(covs)
	''' inference in BEV
	inference = label_inference_BEV(gen_std=gs, degree_register=dr, prob_outlier=0.03)
	boxBEV = box3D[[0,1,3,5,6]]
	points_clip_BEV = points_clip_3D[0:3,[0,2]]
	uncertain_label = inference.infer(points_clip_BEV, boxBEV, prior_scaler=pscale)
	centers, covs = uncertain_label.calc_uncertainty_corners()
	print(covs)
	'''
if __name__ == '__main__':
	main()
