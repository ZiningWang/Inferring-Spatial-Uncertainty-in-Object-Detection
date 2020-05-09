import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../'))
import utils.metric_utils#probability_utils
from utils.probability_utils import Hujie_uncertainty_reader as unc_reader
from utils.probability_utils import cov_interp_3D

imageSet_dir = '/mnt/d/Berkeley/Kitti_data/object/kitti/ImageSets/'
list_dir = imageSet_dir+'{}.txt'.format('val_Hujie')

#BEV corner uncertainty
unc_dir = '/mnt/d/Berkeley/Kitti_data/predictions/Hujie_unc_full_val50/unc_data/'

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
	unc_data = unc_reader(unc_dir,file_lists)
	#test one frame
	W = unc_data['homogeneous_w'][test_frame]
	Vs_diag = unc_data['points_unc'][test_frame][0]
	Vs = np.array([np.diag(Vs_diag[i]) for i in range(Vs_diag.shape[0])])
	print(W[0], type(W[0].size), Vs.shape, unc_data['corners'][test_frame].shape)
	cov_interpolation = cov_interp_3D(W,Vs)

if __name__ == '__main__':
	main()
