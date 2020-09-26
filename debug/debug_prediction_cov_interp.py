import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../'))
from utils.simple_stats import get_dirs, read_points_cam, label_reader
from utils.probability_utils import Hujie_uncertainty_reader as unc_reader
from utils.probability_utils import cov_interp_3D, W_Vs_3D_to_BEV_full
from utils.metric_utils import uncertain_prediction_BEV_interp, label_uncertainty_IoU
from utils.kitti_utils import box3d_to_boxBEV, box3DtoObj
from utils.metric_plots import draw_spatial_uncertainty_contour
from utils.paul_geometry import clip_by_BEV_box

# directories
pred_dir = '/mnt/d/Berkeley/Kitti_data/predictions/Hujie_unc_full_val50/'
data_dir = '/mnt/d/Berkeley/Kitti_data/object'
list_dir, img_dir, lidar_dir, calib_dir, label_3d_dir = get_dirs(data_dir, False, val_set='Hujie')
output_dir = './results/Hujie_unc_full_val50/prediction_spatial_uncertainty/'
os.makedirs(output_dir, exist_ok=True)

test_frame = 4214 

def test_interp_recovery(boxBEVs, W_BEV, Vs_BEV):
	# test the interpolated cov are the same at sampled points
	uncertain_preds = []
	for i in range(boxBEVs.shape[0]):
		assert np.all(np.array(Vs_BEV[i].shape == np.array([6, 2, 2]))), Vs_BEV.shape
		uncertain_pred = uncertain_prediction_BEV_interp(boxBEVs[i,:], W_BEV, Vs_BEV[i])
		centers, covs = uncertain_pred.calc_uncertainty_corners()
		if i == 7:
			ws = np.array([[0, 0, 1]])
			print(uncertain_pred.cov_interpolation.Vinterp)
			ww_flatten = uncertain_pred.cov_interpolation.extract_upper_triangle_ws(ws)
			print(ww_flatten)
			t_covs = uncertain_pred.cov_interpolation.interps(ws)
			print(t_covs)
			w, v = np.linalg.eig(t_covs)
			print(np.expand_dims(w, axis=-2).shape, v.shape)
			if np.sum(w < 0) > 0:
				print('Warning: Negative variance detected at a sample')
				w[w < 0.02**2] = 0.02**2
				t_covs = np.matmul(np.transpose(v, axes=(0, 2, 1)), np.expand_dims(w, axis=-2) * v)
				print(t_covs.shape)
		assert np.all(covs-Vs_BEV[i][[1,0,5,2], :, :] == 0)
		uncertain_preds.append(uncertain_pred)
	print('No bug in recovering the uncertainty of corners.')
	return uncertain_preds

def test_spatial_distribution(output_file, box3Ds, rys, uncertain_preds, points_cam, grid_size=0.1, sample_grid=0.02):
	points_clip_BEVs = []
	label_boxBEVs = []
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[10,7])
	axes.scatter(points_cam[:, 0], points_cam[:, 2], c='y', marker='x', s=0.1)#
	IoUcalculator = label_uncertainty_IoU(grid_size=grid_size, range=2)
	for i, _ in enumerate(uncertain_preds):
		points_clip_BEVs.append(clip_by_BEV_box(points_cam, box3Ds[i, 0:3], box3Ds[i, 3:6], rys[i], buffer_size = 0.2)[:,[0,2]])
		cntr = draw_spatial_uncertainty_contour(box3DtoObj(box3Ds[i], rys[i]), points_clip_BEVs[i], uncertain_preds[i], IoUcalculator, axes, sample_grid, grid_size)
	fig.colorbar(cntr, ax=axes)
	for i in range(len(points_clip_BEVs)):
		axes.scatter(points_clip_BEVs[i][:, 0], points_clip_BEVs[i][:, 1], c='r', marker='x', s=3)

	axes.set_xlim(-15.0,5.0) #axes.set_xlim(0.0, 60.0)  #
	axes.set_ylim(10.0,37.0) #axes.set_ylim(-20.0, 20.0)  #
	plt.savefig(output_file)
	plt.close()


def main():
	#with open(list_dir) as fi:
	#	file_lists = fi.read().splitlines() 
	file_lists = ["00{:04d}".format(test_frame)]
	file_lists.sort()
	unc_data = unc_reader(pred_dir,file_lists)
	label_data = label_reader(label_3d_dir,file_lists,calib_dir)
	#test one frame
	boxBEV = box3d_to_boxBEV(unc_data['box3D'][test_frame], ry=unc_data['ry'][test_frame])
	W_BEV, Vs_BEV = W_Vs_3D_to_BEV_full(unc_data['homogeneous_w'][test_frame], unc_data['points_unc_sigma'][test_frame]**2)

	uncertain_preds = test_interp_recovery(boxBEV, W_BEV, Vs_BEV)
	#print(unc_data['points_unc_sigma'][test_frame])

	points_cam = read_points_cam(lidar_dir, test_frame, label_data)
	test_spatial_distribution(os.path.join(output_dir, "{:06d}".format(test_frame)+'.png'), unc_data['box3D'][test_frame], unc_data['ry'][test_frame], uncertain_preds, points_cam)



if __name__ == '__main__':
	main()
