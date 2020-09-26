import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../'))
from utils.paul_geometry import lidar_to_camera, clip_by_BEV_box, draw_birdeye
from utils.metric_utils import label_inference_BEV, label_uncertainty_IoU, uncertain_prediction_BEVdelta
from utils.simple_stats import label_reader
from utils.kitti_utils import box3DtoObj

KEYS = ['X','Y','Z','intensity'] #intensity is semantic label
SIM_DATA_DIR = "/mnt/d/Berkeley/Kitti_data/sim_data/"
FILENAME = "2020-04-23-11-51-35_Velodyne-HDL-32-Data.csv"
FILELIST = ['004214.txt']


def plot_frame_with_uncertainty(output_dir, label_data, frame, gt_idxs, points_cam, grid_size=0.1, sample_grid=0.02):
	out_file_name = "{:06d}".format(frame)
	points_clip_BEVs = []
	label_boxBEVs = []
	objs = []
	for i, gt_idx in enumerate(gt_idxs):
		box3D = label_data['box3D'][frame][gt_idx]
		ry = label_data['ry'][frame][gt_idx]
		label_boxBEV = [box3D[0],box3D[2],box3D[5],box3D[4],ry]
		points_clip = clip_by_BEV_box(points_cam, box3D[0:3], box3D[3:6], ry, buffer_size = 0.2)
		points_clip_BEV = points_clip[:,[0,2]]
		obj = box3DtoObj(label_data['box3D'][frame][gt_idx], label_data['ry'][frame][gt_idx])
		points_clip_BEVs.append(points_clip_BEV)
		label_boxBEVs.append(label_boxBEV)
		objs.append(obj)
	draw_introduction_with_uncertainty(output_dir, out_file_name, points_cam, points_clip_BEVs, label_boxBEVs, objs, grid_size=grid_size, sample_grid=sample_grid)

def draw_introduction_with_uncertainty(output_dir, out_file_name, points_cam, points_clip_BEVs, label_boxBEVs, objs, grid_size=0.1, sample_grid=0.02):
	dr = 1
	gs = 0.2
	pscale = 0
	inference = label_inference_BEV(gen_std=gs, degree_register=dr, prob_outlier=0.03)
	IoUcalculator = label_uncertainty_IoU(grid_size=grid_size, range=2)
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[8,14])
	for i in range(len(points_clip_BEVs)):
		certain_label = uncertain_prediction_BEVdelta(label_boxBEVs[i])
		uncertain_label = inference.infer(points_clip_BEVs[i], label_boxBEVs[i], prior_scaler=pscale)
		#draw spatial distribution
		cntr = draw_subplot(objs[i],points_clip_BEVs[i], certain_label, uncertain_label, IoUcalculator, axes, fig, sample_grid, grid_size, [gs], [dr])
	#fig.colorbar(cntr, ax=axes)
	axes.scatter(points_cam[:, 0], points_cam[:, 2], c='y', marker='x', s=0.1)#
	for i in range(len(points_clip_BEVs)):
		points_clip_BEV = points_clip_BEVs[i]
		axes.scatter(points_clip_BEV[:, 0], points_clip_BEV[:, 1], c='r', marker='x', s=3)
	axes.set_xlim(-20.0,20.0) #axes.set_xlim(0.0, 60.0)  #
	axes.set_ylim(-10.0,60.0) #axes.set_ylim(-20.0, 20.0)  #
	out_file_path = os.path.join(output_dir, out_file_name+'.png')
	plt.savefig(out_file_path)
	plt.close()

def draw_subplot(obj,points_clip_BEV, certain_label, uncertain_label, IoUcalculator, axes, fig, sample_grid, grid_size, gs_set, col_set):
	levels = np.array(range(256))/255.0
	#draw spatial distribution
	sample_points, pxShape = IoUcalculator.get_sample_points(uncertain_label)
	px = uncertain_label.sample_prob(sample_points, sample_grid=sample_grid)/grid_size**2
	innerContour, outerContour = uncertain_label.calc_uncertainty_box(std=2)
	innerContour = np.vstack((innerContour, innerContour[0,:]))
	outerContour = np.vstack((outerContour, outerContour[0,:]))
	#axes.plot(outerContour[:,0], outerContour[:,1],'y--', linewidth=1)
	#axes.plot(innerContour[:, 0], innerContour[:, 1], 'y--', linewidth=1)
	px[px>1] = 1.0
	#px[px<0.04]=-0.1
	jet_map = plt.cm.jet
	#cmap.set_under('white',0.1)
	cntr = axes.contourf(sample_points[:,0].reshape(pxShape), sample_points[:,1].reshape(pxShape), px.reshape(pxShape), levels, cmap=jet_map, vmin=0, vmax=1)
	draw_birdeye(obj, axes, fill=False, color='r', linewidth=1)
	#axes.scatter(points_clip_BEV[:, 0], points_clip_BEV[:, 1], c='r', marker='x', s=5)
	axes.set(xticks=[],yticks=[])
	axes.set_facecolor((0,0,0.53))
	return cntr

def read_sim_data(filedir):
	data = {key:[] for key in KEYS}
	file = open(filedir)
	reader = csv.DictReader(file, delimiter=",")
	for row in reader:
		for key in KEYS:
			data[key].append(row[key])
	np_data = {}
	np_data['points'] = np.array([data['X'], data['Y'], data['Z']], dtype=np.float).transpose()
	np_data['labels'] = np.array(data['intensity'], dtype=np.float).astype(np.int64)
	return np_data

def get_point_cam(points):
	# for simulation data, to camera frame
	points_cam = points[:,[0,2,1]]
	points_cam[:,1] = points_cam[:,1]*-1
	return points_cam

def main():
	data_dir = '/mnt/d/Berkeley/Kitti_data/object'
	folder_name = 'training'
	img_dir = os.path.join(data_dir, folder_name, 'image_2')
	label_3d_dir = os.path.join(data_dir, folder_name, 'label_2')
	calib_dir = os.path.join(data_dir, folder_name, 'calib')
	difficulty = 'HARD'
	view = 'G'
	sim_file_dir = os.path.join(SIM_DATA_DIR, FILENAME)

	output_dir = 'results/uncertainty_analysis/introduction_plot_sim/'
	os.makedirs(output_dir, exist_ok=True)

	label_data = label_reader(label_3d_dir,FILELIST,calib_dir)
	for id_file ,file in enumerate(FILELIST):
		frame = int(file.split('.')[0])
		np_data = read_sim_data(sim_file_dir)
		points_cam = get_point_cam(np_data['points'])
		all_gts =  np.logical_and(np.array(label_data['difficulty'][frame])>=-1,np.array(label_data['class_id'][frame])==1)
		active_gt_idxs = np.argwhere(all_gts)
		gt_idxs = [active_gt_idxs.item(i) for i in range(active_gt_idxs.size)]
		plot_frame_with_uncertainty(output_dir, label_data, frame, gt_idxs, points_cam, grid_size=0.1, sample_grid=0.02)

if __name__ == '__main__':
	main()