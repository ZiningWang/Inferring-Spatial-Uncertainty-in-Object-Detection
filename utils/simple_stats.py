import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Arrow
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import scipy.misc
import logging
import coloredlogs
from PIL import Image, ImageDraw, ImageFont
from easydict import EasyDict as edict
from tqdm import tqdm
import glob
import cv2
import math

sys.path.append('../')
from utils.paul_geometry import draw_3dbox_in_2d, read_calib_mat, camera_to_lidar, draw_birdeye, cal_box3d_iou, lidar_to_camera, clip_by_BEV_box, get_cov_ellipse, draw_birdeye_ellipse
from utils.metric_plots import *
from utils.kitti_utils import box3d_to_boxBEV
from utils.probability_utils import W_Vs_3D_to_BEV_full
from utils.metric_utils import label_inference_BEV, label_uncertainty_IoU, uncertain_prediction_BEVdelta, uncertain_prediction_BEV, uncertain_prediction_BEV_interp

#modify this to view different networks
networks = ['PointRCNN']
#['STD']#['AVOD']#['SECOND']#['PIXORbad']#['PIXORnew']#['Voxel']#['PointRCNN']#['ProbPIXOR']#['ProbCalibPIXOR']#['PIXORwaymo']#['ProbPIXORwaymo']#['ProbCalibPIXORwaymo]#['ProbPIXORwaymo_boxonly']
pred_3d_dirs = []
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/std_val/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/avod_kitti_val50/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/second_kitti_val50/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/pixor_bad_kitti_val50/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/pixor_new_kitti_val50/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/voxelnet_kitti_val50/')
pred_3d_dirs.append('/data/RPN/coop_DiFeng/PointRCNN_val50/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/probablistic_pixor/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/probablistic_calibrated_pixor/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/waymo_detection_2d/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/waymo_detection_2d_probablistic/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/waymo_detection_2d_probablistic_calibrated/')
#pred_3d_dirs.append('/data/RPN/coop_DiFeng/waymo_detection_2d_probablistic_boxonly/')

actual_test = False
data_dir = '/home/msc/KITTI/object' #'/data/RPN/coop_DiFeng/WaymoData' #

def get_dirs(data_dir_in, actual_test_in, val_set=None):
	if actual_test_in == False:
		folder_name = 'training'
		if val_set == 'Hujie':
			print('Using validataion set of Hujie Pan.')
			list_name = 'val_Hujie'
		else:
			list_name = 'val'#'val_uncertainty_debug'#
		label_3d_dir = os.path.join(data_dir_in, folder_name, 'label_2')
	else:
		folder_name = 'testing'
		list_name = 'test'
	if data_dir_in.find('Waymo')!=-1 or data_dir_in.find('waymo') != -1:
		list_dir = '/data/RPN/coop_DiFeng/waymo_detection_2d/imageset/val.txt'#
	else:
		list_dir = os.path.join(data_dir_in, 'kitti/ImageSets/{}.txt'.format(list_name))#

	img_dir = os.path.join(data_dir_in, folder_name, 'image_2')
	lidar_dir = os.path.join(data_dir_in, folder_name, 'velodyne')
	calib_dir = os.path.join(data_dir_in, folder_name, 'calib')
	return list_dir, img_dir, lidar_dir, calib_dir, label_3d_dir


def label_reader(label_3d_dir,file_lists,calib_dir,max_num=7518):
	#specify the range of files or names of files
	start_file_num = -1
	end_file_num = 1000000
	test_name = None#['007065']
	using_waymo = False
	if label_3d_dir.find('Waymo')!=-1 or label_3d_dir.find('waymo') != -1:
		#require a special converter for waymo dataset
		using_waymo = True

	attrs = ['class','class_id','box2D','box3D','ry','calib','occlusion','truncation','difficulty','alpha']
	label_data = {attr:[[] for name in range(max_num)] for attr in attrs}
	print('reading label_data from: {}'.format(label_3d_dir))
	for label_file in tqdm(file_lists):
		file_num = label_file.split('.')[0] 
		name = file_num
		frame = int(name)
		if test_name is not None:
			if name not in test_name:
				continue
		if int(float(name)) <= start_file_num:
			continue
		if int(float(name)) > end_file_num:
			break
		calib_mat = read_calib_mat(calib_dir, frame)
		label_data['calib'][frame] = calib_mat
		label_3d_file_dir = os.path.join(label_3d_dir, "{}.txt".format(file_num))
		#print("label_3d_file_dir: {}".format(label_3d_file_dir))
		with open(label_3d_file_dir,'r') as fi:
			line = fi.readline()
			while len(line)>4:
				label_infos = line.split()	
				# print("label_infos: {}".format(label_infos))	
				label_data['class'][frame].append(label_infos[0])
				if (label_infos[0]=='Car') or (label_infos[0]=='Van') or (label_infos[0]=='Truck'):
					#vehicles
					label_data['class_id'][frame].append(1)
				else:
					label_data['class_id'][frame].append(-1)
				trun = float(label_infos[1])
				occ  = int(label_infos[2])
				label_data['alpha'][frame] = float(label_infos[3])
				label_data['truncation'][frame].append(trun)
				label_data['occlusion'][frame].append(occ)
				if trun<=0.15 and occ<=0:
					label_data['difficulty'][frame].append(0)
				elif trun<=0.3 and occ<=1:
					label_data['difficulty'][frame].append(1)
				elif trun<=0.5 and occ<=2:
					label_data['difficulty'][frame].append(2)
				else:
					label_data['difficulty'][frame].append(-1)
				x1 = int(float(label_infos[4]))
				y1 = int(float(label_infos[5]))
				x2 = int(float(label_infos[6]))
				y2 = int(float(label_infos[7]))
				label_data['box2D'][frame].append([x1,y1,x2,y2])
				obj_h = float(label_infos[8])
				obj_w = float(label_infos[9])
				obj_l = float(label_infos[10])
				x_cam = float(label_infos[11])
				if not using_waymo:
					y_cam = float(label_infos[12])
					z_cam = float(label_infos[13])
					label_data['ry'][frame].append(float(label_infos[14]))
				else:
					#data converter for waymo, since it uses x,y,z center in LIDAR frame
					y_cam = float(label_infos[13])+obj_h/2
					z_cam = float(label_infos[12])
					label_data['ry'][frame].append(-float(label_infos[14]))
				label_data['box3D'][frame].append([x_cam,y_cam,z_cam,obj_h,obj_w,obj_l])
				
				line = fi.readline()
	return label_data

def detect_reader(pred_3d_dir,file_lists,waymo_in_label_dir,max_num=7518):
	attrs = ['class','box2D','box3D','ry','score','uncertainty','file_exists','alpha']
	pred_data = {attr:[[] for name in range(max_num)] for attr in attrs}
	print('reading prediction_data from: {}'.format(pred_3d_dir))
	count = 0
	using_waymo = False
	if waymo_in_label_dir or pred_3d_dir.find('Waymo') != -1:
		# require a special converter for waymo dataset
		using_waymo = True
	if os.path.isfile(pred_3d_dir+'uncertainty_calibration/scale.txt'):
		print('using calibration result for prediction uncertainty')
		with open(pred_3d_dir+'uncertainty_calibration/scale.txt','r') as fi:
			line = fi.readline()
			uncertainty_calibs = line.split()
			cosry_s = np.sqrt(float(uncertainty_calibs[0]))
			sinry_s = np.sqrt(float(uncertainty_calibs[1]))
			xc1_s = np.sqrt(float(uncertainty_calibs[2]))
			xc2_s = np.sqrt(float(uncertainty_calibs[3]))
			logl_s = np.sqrt(float(uncertainty_calibs[4]))
			logw_s = np.sqrt(float(uncertainty_calibs[5]))
	else:
		xc1_s = 1; xc2_s = 1; logl_s = 1; logw_s = 1; cosry_s = 1; sinry_s = 1
	for pred_file in tqdm(file_lists):
		file_num = pred_file.split('.')[0] 
		name = file_num
		frame = int(name)
		pred_file_dir = os.path.join(pred_3d_dir,'data/', "{}.txt".format(file_num))
		pred_uncertainty_dir = os.path.join(pred_3d_dir,'uncertainty/', "{}.txt".format(file_num))
		#print("pred_file_dir: {}".format(pred_file_dir))
		if not os.path.isfile(pred_file_dir):
			continue
		count += 1
		pred_data['file_exists'][frame] = True
		with open(pred_file_dir,'r') as fi:
			line = fi.readline()
			while len(line)>4:
				pred_infos = line.split()
				pred_data['class'][frame].append(pred_infos[0])
				pred_data['alpha'][frame].append(float(pred_infos[3]))
				alpha = float(pred_infos[3])
				x1 = float(pred_infos[4])
				y1 = float(pred_infos[5])
				x2 = float(pred_infos[6])
				y2 = float(pred_infos[7])
				pred_data['box2D'][frame].append([x1,y1,x2,y2])
				# Some predictions have negative lengths...
				obj_h = abs(float(pred_infos[8]))
				obj_w = abs(float(pred_infos[9]))
				obj_l = abs(float(pred_infos[10]))
				x_cam = float(pred_infos[11])
				if not using_waymo:
					y_cam = float(pred_infos[12])
					z_cam = float(pred_infos[13])
					pred_data['ry'][frame].append(float(pred_infos[14]))
				else:
					# data converter for waymo, since it uses x,y,z center in LIDAR frame
					y_cam = float(pred_infos[13]) + obj_h / 2
					z_cam = float(pred_infos[12])
					pred_data['ry'][frame].append(-float(pred_infos[14]))
				pred_data['box3D'][frame].append([x_cam,y_cam,z_cam,obj_h,obj_w,obj_l])
				pred_data['score'][frame].append(float(pred_infos[-1]))
				line = fi.readline()
		if not os.path.isfile(pred_uncertainty_dir):
			continue
		with open(pred_uncertainty_dir,'r') as fi:
			line = fi.readline()
			while len(line)>4:
				uncertainty_infos = line.split()
				cosry = np.sqrt(float(uncertainty_infos[0]))*cosry_s
				sinry = np.sqrt(float(uncertainty_infos[1]))*sinry_s
				#angle uncertainty read but should not be used
				xc1 = np.sqrt(float(uncertainty_infos[2]))*xc1_s
				xc2 = np.sqrt(float(uncertainty_infos[3]))*xc2_s
				logl = np.sqrt(float(uncertainty_infos[4]))*logl_s
				logw = np.sqrt(float(uncertainty_infos[5]))*logw_s
				pred_data['uncertainty'][frame].append([xc1,xc2,logl,logw,cosry,sinry])
				line = fi.readline()
		assert(len(pred_data['uncertainty'][frame])==len(pred_data['score'][frame]))

	print('detection files read: ',count)
	return pred_data


def hacker_reader(folder,file_lists,max_num=7518):
	difficulties = ['EASY','MODERATE','HARD']
	views = ['3D','G','2D']
	det_attrs = ['fp','gt','iou']
	gt_attrs = ['tp','fn','det']

	hack_data = {diff:{'det':{view:{attr:[[] for name in range(max_num)] for attr in det_attrs} for view in views}, 
						'gt':{view:{attr:[[] for name in range(max_num)] for attr in gt_attrs} for view in views}} for diff in difficulties}
	print('reading hack_data from: {}'.format(folder))
	for difficulty in tqdm(difficulties):
		hack_det_file = 'hack_det_{}_evaluation.txt'.format(difficulty)
		hack_gt_file = 'hack_gt_{}_evaluation.txt'.format(difficulty)
		with open(os.path.join(folder,hack_det_file),'r') as fi:
			line = fi.readline()
			line = fi.readline()
			while len(line) > 4:
				hack_info = line.split(',')
				frame = int(float(hack_info[0]))
				idx = int(float(hack_info[1]))
				assert (idx == len(hack_data[difficulty]['det'][views[0]][det_attrs[0]][frame])), 'unsorted or missing detection index'
				for q,attr in enumerate(det_attrs[0:2]):
					for p,view in enumerate(views):
						hack_data[difficulty]['det'][view][attr][frame].append(int(float(hack_info[2+q*3+p])))
				#for iou, save float
				q = len(det_attrs)-1
				attr = det_attrs[q]
				for p,view in enumerate(views):
						hack_data[difficulty]['det'][view][attr][frame].append(float(hack_info[2+q*3+p]))
				line = fi.readline()

		with open(os.path.join(folder,hack_gt_file),'r') as fi:
			line = fi.readline()
			line = fi.readline()
			while len(line) > 0:
				hack_info = line.split(',')
				frame = int(float(hack_info[0]))
				idx = int(float(hack_info[1]))
				assert (idx == len(hack_data[difficulty]['gt'][views[0]][gt_attrs[0]][frame])), 'unsorted or missing gt index'
				for q,attr in enumerate(gt_attrs):
					for p,view in enumerate(views):
						hack_data[difficulty]['gt'][view][attr][frame].append(int(float(hack_info[2+q*3+p])))
				line = fi.readline()
		#assign associations to gts labeled as fns
		for view in views:
			for frame in range(len(hack_data[difficulty]['gt'][view]['fn'])):
				max_iou = [0 for gt_idx in hack_data[difficulty]['gt'][view]['fn'][frame]]
				for j, gt_idx in enumerate(hack_data[difficulty]['det'][view]['gt'][frame]):
					iou = hack_data[difficulty]['det'][view]['iou'][frame][j]
					if gt_idx>=0 and hack_data[difficulty]['gt'][view]['det'][frame][gt_idx]<0 and iou>=max_iou[gt_idx]:
						hack_data[difficulty]['gt'][view]['det'][frame][gt_idx] = j
	return hack_data

def read_points_cam(lidar_dir, frame, label_data):
	if lidar_dir.find('Waymo')!=-1:
		name = "{:08d}".format(frame)
	else:
		name = "{:06d}".format(frame)
	lidar_path = os.path.join(lidar_dir,name+'.bin')
	points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,4)
	points = points[:,0:3]
	if lidar_dir.find('Waymo')!=-1:
		points_cam = np.array((points[:,0],points[:,2],points[:,1])).transpose()
	else:
		calib_mat = label_data['calib'][frame]
		xs, ys, zs = lidar_to_camera(points[:, 0], points[:, 1], points[:, 2], calib_mat)
		points_cam = np.array((xs,ys,zs)).transpose()
	return points_cam

def evaluate_JIoU(label_data, pred_data, frame, gt_idxs, pd_idxs, points_cam, grid_size=0.1, sample_grid=0.1, unc_data=None):
	assert(len(gt_idxs)==len(pd_idxs))
	inference = label_inference_BEV(degree_register=1,gen_std=0.25, prob_outlier=0.8,boundary_sample_interval=0.05)
	IoUcalculator = label_uncertainty_IoU(grid_size=grid_size, range=3)
	uncertain_labels = []
	JIoUs = []
	pred_uncertainty_ind = True if pred_data['uncertainty'][frame] or unc_data else False
	uncertainty_format = unc_data['uncertainty_format'] if unc_data else None
	label_boxBEVs = box3d_to_boxBEV(np.array(label_data['box3D'][frame]), np.array(label_data['ry'][frame]))
	pred_boxBEVs = box3d_to_boxBEV(np.array(pred_data['box3D'][frame]), np.array(pred_data['ry'][frame]))
	if uncertainty_format == 'full':
		W_BEV, Vs_BEV = W_Vs_3D_to_BEV_full(unc_data['homogeneous_w'][frame], unc_data['points_unc_sigma'][frame]**2)
		assert (Vs_BEV.shape[0]==pred_boxBEVs.shape[0]), 'inconsistent prediction and uncertainty data at frame %d, %d vs %d.' % (frame, Vs_BEV.shape[0], pred_boxBEVs.shape[0])
	for i, gt_idx in enumerate(gt_idxs):
		points_clip_BEV = clip_by_BEV_box(points_cam, label_data['box3D'][frame][gt_idx][0:3], label_data['box3D'][frame][gt_idx][3:6],
										  label_data['ry'][frame][gt_idx], buffer_size = 0.1)[:,[0,2]]
		uncertain_labels.append(inference.infer(points_clip_BEV, label_boxBEVs[gt_idx]))
		certain_label = uncertain_prediction_BEVdelta(label_boxBEVs[gt_idx])
		pd_idx = pd_idxs[i]
		if pd_idx >= 0:
			certain_pred = uncertain_prediction_BEVdelta(pred_boxBEVs[pd_idx])
			if pred_uncertainty_ind:
				if uncertainty_format == 'full':
					if frame < 10:
						print('using interpolated corner uncertainty as prediction uncertainty')
					uncertain_pred = uncertain_prediction_BEV_interp(pred_boxBEVs[pd_idx], W_BEV, Vs_BEV[pd_idx])
				elif uncertainty_format == 'corner':
					assert False, 'Unimplemented prediction uncertainty model.'
				else:
					uncertain_pred = uncertain_prediction_BEV(pred_boxBEVs[pd_idx], pred_data['uncertainty'][frame][pd_idx])
			else:
				uncertain_pred = uncertain_prediction_BEVdelta(pred_boxBEVs[pd_idx])
			JIoU = IoUcalculator.calc_IoU(uncertain_labels[i], [uncertain_pred,certain_label,certain_pred], sample_grid=sample_grid)
		else:
			JIoU = [0] + IoUcalculator.calc_IoU(uncertain_labels[i], [certain_label], sample_grid=sample_grid) + [0]
		JIoUs.append(JIoU)
	return JIoUs

def main():
	# (1) get file list, label and images
	list_dir, img_dir, lidar_dir, calib_dir = get_dirs(data_dir, actual_test)

	num_net = len(networks)
	pred_datas = [[] for net in networks]
	hack_datas = [[] for net in networks]

	output_root = './results/'
	output_dir = output_root + 'results_visualizer_new'
	for net in networks:
		output_dir += '_'+net
	output_dir += '/'
	os.makedirs(output_dir, exist_ok=True)


	with open(list_dir) as fi:
		file_lists = fi.read().splitlines() 
	file_lists.sort()

	label_data = label_reader(label_3d_dir,file_lists,calib_dir)
	for inet in range(len(networks)):
		hack_datas[inet] = hacker_reader(pred_3d_dirs[inet], file_lists)
		pred_datas[inet] = detect_reader(pred_3d_dirs[inet], file_lists, label_3d_dir.find('waymo') != -1)

	#all labels, JIoUs vs IoU
	difficulty = 'HARD'
	view = 'G'
	frames = [int(file.split('.')[0]) for file in file_lists]
	all_gt_idxs = [{net:[] for net in networks} for frame in frames]
	all_pd_idxs = [{net:[] for net in networks} for frame in frames]
	all_IoU_dic = [{net:[] for net in networks} for frame in frames]
	corner_totalVariances = []
	for id_file ,file in enumerate(tqdm(file_lists)):
		frame = int(file.split('.')[0])
		points_cam = read_points_cam(lidar_dir, frame, label_data)
		num_active_gts = 0
		IoU_dic = {}
		all_gts = np.zeros(len(hack_datas[0][difficulty]['gt'][view]['tp'][frame]))
		for inet, net in enumerate(networks):
			gts = np.logical_or(np.array(hack_datas[inet][difficulty]['gt'][view]['tp'][frame])==1,np.array(hack_datas[inet][difficulty]['gt'][view]['fn'][frame])==1)
			all_gts = np.logical_or(all_gts, gts)
		active_gt_idxs = np.argwhere(all_gts)
		num_active_gts += active_gt_idxs.size
		for inet, net in enumerate(networks):
			output_JIoU_dir = output_dir + net + '_JIoUs/'
			os.makedirs(output_JIoU_dir, exist_ok=True)
			if active_gt_idxs.size>0:
				ass_det_idxs = np.array(hack_datas[inet][difficulty]['gt'][view]['det'][frame])[active_gt_idxs]
				gt_idxs = [active_gt_idxs.item(i) for i in range(active_gt_idxs.size)]
				pd_idxs =  [ass_det_idxs.item(i) for i in range(ass_det_idxs.size)]
				IoU = []
				for pd_idx in pd_idxs:
					if pd_idx>=0:
						IoU.append(hack_datas[inet][difficulty]['det'][view]['iou'][frame][pd_idx])
					else:
						IoU.append(0)
				all_gt_idxs[id_file][net] = gt_idxs
				all_pd_idxs[id_file][net] = pd_idxs

				JIoUs = evaluate_JIoU(label_data, pred_datas[inet], frame, gt_idxs, pd_idxs, points_cam, grid_size=0.1, sample_grid=0.02)
				for iI in range(len(IoU)):
					tmp = [IoU[iI]] + JIoUs[iI]
					all_IoU_dic[id_file][net].append(tmp)

		if active_gt_idxs.size>0:
			pass
			#corner_totalVariances.append(get_corner_variances(label_data, frame, gt_idxs, points_cam))
			#plot_multiview_label(label_data, img_dir, points_cam, frame, gt_idxs, output_JIoU_dir, IoU_dic=all_IoU_dic[id_file], pd_idxs=pd_idxs, pred_data=pred_datas[len(pred_datas)-1])
	output_JIoU_dir = output_dir + 'summary/uncertaintyV3_for_labelwellness_0.25_0.8_deg1/'
	write_JIoU(output_JIoU_dir, networks, frames, all_gt_idxs, all_pd_idxs, all_IoU_dic)
	#write_corner_variances(output_root, corner_totalVariances)

	'''
	#count common fn's and plot
	difficulty = 'HARD'
	view = 'G'
	num_fns = np.zeros(num_net)
	num_common_fns = 0
	regression_error_idxs = []
	output_common_fn_dir = output_dir + 'common_fns/'
	os.makedirs(output_common_fn_dir, exist_ok=True)
	frame_idxs_pairs = []
	iou_hists = {inet: [] for inet in range(len(networks))}
	print('find common false negatives from {} and view {}'.format(difficulty,view))
	for file in tqdm(file_lists):
		frame = int(file.split('.')[0])
		points_cam = read_points_cam(lidar_dir, frame, label_data)
		fns = []
		for inet, net in enumerate(networks):
			fns.append(hack_datas[inet][difficulty]['gt'][view]['fn'][frame])
		fns = np.array(fns)==1
		common_fns = fns[0,:]
		for inet in range(1,num_net):
			common_fns = np.logical_and(common_fns,fns[inet,:])
		common_fns_idx = np.argwhere(common_fns)
		num_common_fns += common_fns_idx.size
		if common_fns_idx.size > 0:
			idxs = [common_fns_idx.item(icommon) for icommon in range(common_fns_idx.size)]
			#plot_multiview_label(label_data, img_dir, points_cam, frame, idxs, output_common_fn_dir)

			#find overlaps and plot historgram
			for inet in range(len(networks)):
				ass_gts = hack_datas[inet][difficulty]['det'][view]['gt'][frame]
				for idx in idxs:
					max_iou = 0.0
					for j,ass_gt in enumerate(ass_gts):
						if idx == ass_gt:
							max_iou = max(max_iou, hack_datas[inet][difficulty]['det'][view]['iou'][frame][j])
					if max_iou>0.1 and max_iou<0.70001: #DEBUG
						regression_error_idxs.append([frame,idx, max_iou])
						print('DEBUG: #regression error: ',frame,idx, max_iou)
					iou_hists[inet].append([frame,idx,max_iou])
			#count the number of points inside the gt box
			frame_idxs_pairs.append((frame,idxs))
	output_common_fn_stats_dir = output_common_fn_dir+'stats/'
	plot_histogram_network_wise(output_common_fn_stats_dir, 'iou_histogram.png', networks, iou_hists, 2, x_label='IoU', y_label='number of false negatives',n_bins=50)
	plot_npoint_histogram_label(label_data, points_cam, frame_idxs_pairs, output_common_fn_stats_dir)
	write_stats_label(label_data, frame_idxs_pairs, output_common_fn_stats_dir, description="false positive stats: ", reg_err_idxs=regression_error_idxs)
	'''




if __name__ == '__main__':
	main()
