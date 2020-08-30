import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../'))
from utils.probability_utils import Hujie_uncertainty_reader as unc_reader
from utils.kitti_utils import save_kitti_format
from utils.calibration import Calibration
from utils.simple_stats import get_dirs

data_dir = '/mnt/d/Berkeley/Kitti_data/object' 
actual_test = False
pred_3d_dir = ('/mnt/d/Berkeley/Kitti_data/predictions/Hujie_unc_full_val50/')

def main():
	list_dir, img_dir, lidar_dir, calib_dir, label_3d_dir = get_dirs(data_dir, actual_test, val_set='Hujie')
	with open(list_dir) as fi:
		file_lists = fi.read().splitlines() 
		file_lists.sort()
	unc_data = unc_reader(pred_3d_dir, file_lists)
	for filename in file_lists:
		frame = int(filename.split('.')[0])
		bbox3d = np.concatenate([unc_data['box3D'][frame], unc_data['ry'][frame].reshape((-1,1))], axis=1)
		calib = Calibration(os.path.join(calib_dir, filename + '.txt'))
		img_shape = [1240, 375]
		save_kitti_format(frame, calib, bbox3d, pred_3d_dir+'data/', np.ones(bbox3d.shape[0]), img_shape)

if __name__ == '__main__':
	main()