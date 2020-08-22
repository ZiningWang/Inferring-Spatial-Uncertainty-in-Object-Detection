import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../'))
import utils.metric_utils#probability_utils
from utils.probability_utils import Hujie_uncertainty_reader as unc_reader
from utils.probability_utils import cov_interp_3D, W_Vs_3D_to_BEV_full
from utils.metric_utils import uncertain_prediction_BEV_interp
from utils.kitti_utils import box3d_to_boxBEV

imageSet_dir = '/mnt/d/Berkeley/Kitti_data/object/kitti/ImageSets/'
list_dir = imageSet_dir+'{}.txt'.format('val_Hujie')

#3D corner uncertainty
pred_dir = '/mnt/d/Berkeley/Kitti_data/predictions/Hujie_unc_full_val50/'

#write the new imageSet file as Hujie has been using a different one
import glob
filelist = []
'''
with open(os.path.join(imageSet_dir,'val_Hujie.txt'),'a') as fo:
	for file in glob.glob(unc_dir+'*.pickle'):
		filenum = file[-17:-11]
		filelist.append(filenum)
		fo.write(filenum+'\n')
'''

test_frame = 4214 

def main():
	#with open(list_dir) as fi:
	#	file_lists = fi.read().splitlines() 
	file_lists = ["00{:04d}".format(test_frame)]
	file_lists.sort()
	unc_data = unc_reader(pred_dir,file_lists)
	#test one frame
	Vs_diag = unc_data['points_unc'][test_frame][0]
	Vs = np.array([np.diag(Vs_diag[i]) for i in range(Vs_diag.shape[0])])
	cov_interpolation = cov_interp_3D(unc_data['homogeneous_w'][test_frame],Vs)
	boxBEV = box3d_to_boxBEV(unc_data['box3D'][test_frame], ry=unc_data['ry'][test_frame])
	# TODO: get W and Vs for BEV
	W_BEV, Vs_BEV = W_Vs_3D_to_BEV_full(unc_data['homogeneous_w'][test_frame], unc_data['points_unc'][test_frame])
	for i in range(unc_data['box3D'][test_frame].shape[0]):
		assert np.all(np.array(Vs_BEV[i].shape == np.array([6, 2, 2]))), Vs_BEV.shape
		uncertain_pred = uncertain_prediction_BEV_interp(boxBEV[i,:], W_BEV, Vs_BEV[i])
		centers, covs = uncertain_pred.calc_uncertainty_corners()
		assert np.all(covs-Vs_BEV[i][[1,0,5,2], :, :] == 0)

	print('No bug in recovering the uncertainty of corners.')

if __name__ == '__main__':
	main()
