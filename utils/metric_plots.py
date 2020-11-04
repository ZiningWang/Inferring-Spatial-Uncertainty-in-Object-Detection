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
from utils.paul_geometry import draw_3dbox_in_2d, read_calib_mat, camera_to_lidar, draw_birdeye, cal_box3d_iou, lidar_to_camera, clip_by_BEV_box, get_cov_ellipse, draw_birdeye_ellipse, center_to_corner_BEV, align_BEV_to_corner


def draw_spatial_uncertainty_contour(obj, points_clip_BEV, uncertain_class, IoUcalculator, axes, sample_grid, grid_size):
	levels = np.array(range(256))/255.0
	#draw spatial distribution
	sample_points, pxShape = IoUcalculator.get_sample_points(uncertain_class)
	px = uncertain_class.sample_prob(sample_points, sample_grid=sample_grid)/grid_size**2
	innerContour, outerContour = uncertain_class.calc_uncertainty_box(std=2)
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

def plot_multiview_label(label_data, img_dir, points_cam, frame, idxs, output_dir, IoU_dic=None, pd_idxs=None, pred_data=None, draw_uncertainty=True):
	name = "{:06d}".format(frame)
	out_file_name = name
	img_path = os.path.join(img_dir,name+'.png')
	plt.figure(figsize=(10,10))
	ax = plt.gca()
	gs = gridspec.GridSpec(3,3)
	if os.path.exists(img_path):
		img = scipy.misc.imread(img_path)
		ax = plt.subplot(gs[0,0:3])
		ax.imshow(img)
		for idx in idxs:
			box2D = label_data['box2D'][frame][idx]
			ax.add_patch(patches.Rectangle((box2D[0],box2D[1]), box2D[2]-box2D[0], box2D[3]-box2D[1], edgecolor='red', fill=False))
	for idx in idxs:
		out_file_name += '_{}'.format(idx)
	#DEBUG inference
	ax = plt.subplot(gs[1:3,0:3])
	ax.scatter(points_cam[:,0],points_cam[:,2],c='b',marker='o',s=0.1)
	ax.set_xlim(-20.0,20.0)#ax.set_xlim(0.0,60.0)#
	ax.set_ylim(0.0,60.0)#ax.set_ylim(-20.0,20.0)#
	if draw_uncertainty:
		infer_gt_uncertainty(label_data, points_cam, frame, idxs, ax)
	centers = []
	for idx in idxs:
		box3D = label_data['box3D'][frame][idx]
		ry = label_data['ry'][frame][idx]
		#x_lidar, y_lidar, z_lidar = camera_to_lidar(box3D[0], box3D[1], box3D[2], calib_mat)
		obj = edict({})
		obj.R = ry
		obj.width = box3D[4]
		obj.length = box3D[5]
		obj.x = box3D[0]#-y_lidar
		obj.y = box3D[2]#x_lidar
		centers.append([obj.x, obj.y])
		draw_birdeye(obj, ax, fill=False, color='g', linewidth=1.0)
		points_clip = clip_by_BEV_box(points_cam, box3D[0:3], box3D[3:6], ry, buffer_size = 0.1)
		#x_lidar, y_lidar, z_lidar = camera_to_lidar(points_clip[:,0], points_clip[:,1], points_clip[:,2], calib_mat)
		#points_lidar_clip = np.array([x_lidar, y_lidar, z_lidar]).transpose()
		ax.scatter(points_clip[:,0],points_clip[:,2],c='r',marker='o',s=0.2)

	if IoU_dic:
		annotate_IoUs(ax, IoU_dic, centers)
	if pd_idxs and pred_data:
		for idx in pd_idxs:
			if idx<0:
				continue
			box3D = pred_data['box3D'][frame][idx]
			ry = pred_data['ry'][frame][idx]
			# x_lidar, y_lidar, z_lidar = camera_to_lidar(box3D[0], box3D[1], box3D[2], calib_mat)
			obj = edict({})
			obj.R = ry
			obj.width = box3D[4]
			obj.length = box3D[5]
			obj.x = box3D[0]  # -y_lidar
			obj.y = box3D[2]  # x_lidar
			centers.append([obj.x, obj.y])
			draw_birdeye(obj, ax, fill=False, color='b', linewidth=1.0)

	out_file_name += '.png'
	out_file_path = os.path.join(output_dir,out_file_name)
	plt.savefig(out_file_path)
	plt.close()

def write_stats_label(label_data, frame_idxs_pairs, output_dir,description='Unknown: ',reg_err_idxs=None):
	from time import gmtime, strftime
	time_str = strftime("%a, %d %b %Y %H:%M:%S", gmtime())
	difficulty_counts = [0,0,0,0]
	frame_counts = len(frame_idxs_pairs)
	idx_counts = 0
	for frame, idxs in frame_idxs_pairs:
		name = "{:06d}".format(frame)
		for idx in idxs:
			occ  = label_data['occlusion'][frame][idx]
			trun = label_data['truncation'][frame][idx]
			idx_counts+=1
			if trun<=0.15 and occ<=0:
				difficulty_counts[0]+=1
			elif trun<=0.3 and occ<=1:
				difficulty_counts[1]+=1
			elif trun<=0.5 and occ<=2:
				difficulty_counts[2]+=1
			else:
				difficulty_counts[3]+=1
	with open(os.path.join(output_dir,'statistics.txt'),'a') as fo:
		fo.write('\n')
		fo.write(description+time_str+'\n')
		fo.write('# frames: {}\n'.format(frame_counts))
		fo.write('# object: {}\n'.format(idx_counts))
		fo.write('# difficulties (E,M,H,Unknown): {}, {}, {}, {}\n'.format(difficulty_counts[0],difficulty_counts[1],difficulty_counts[2],difficulty_counts[3]))
		if reg_err_idxs:
			fo.write('# regression errors: {}\n'.format(len(reg_err_idxs)))
			for reg_idx in reg_err_idxs:
				fo.write('   regression errors(frame, idx, iou): {}, {}, {}\n'.format(reg_idx[0],reg_idx[1],reg_idx[2]))


def plot_npoint_histogram_label(label_data, points_cam, frame_idxs_pairs, output_dir):
	key_names = ['ground_truth']
	npoints_hists = {ikey:[] for ikey in range(len(key_names))}
	for frame, idxs in tqdm(frame_idxs_pairs):
		for idx in idxs:
			box3D = label_data['box3D'][frame][idx]
			ry = label_data['ry'][frame][idx]
			points_clip = clip_by_BEV_box(points_cam, box3D[0:3], box3D[3:6], ry, buffer_size = 0.25)
			npoints_hists[0].append([frame,idx,min(100,points_clip.shape[0])])
	plot_histogram_network_wise(output_dir, 'num_points_histogram.png', key_names, npoints_hists, 2, x_label='number of points', y_label='number of false negatives', n_bins=50)


def plot_histogram_network_wise(output_dir, file_name, networks, plot_data, icol, x_label='None', y_label='number', n_bins=10):
	os.makedirs(output_dir, exist_ok=True)
	plt.figure(figsize=(10,10))
	gs = gridspec.GridSpec(len(networks),len(networks))
	ax = plt.gca()
	for inet in range(len(networks)):
		iou_hist = np.array(plot_data[inet])
		ax = plt.subplot(gs[inet,:])
		ax.hist(iou_hist[:,icol], bins=n_bins)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
	plt.savefig(os.path.join(output_dir,file_name))


def infer_gt_uncertainty(label_data, points_cam, frame, idxs, ax=None):
	from metric_utils import label_inference_BEV, label_uncertainty_IoU
	inference = label_inference_BEV(degree_register=1)
	for idx in idxs:
		box3D = label_data['box3D'][frame][idx]
		ry = label_data['ry'][frame][idx]
		points_clip = clip_by_BEV_box(points_cam, box3D[0:3], box3D[3:6], ry, buffer_size = 0.1)
		points_clip_BEV = points_clip[:,[0,2]]
		uncertain_label = inference.infer(points_clip_BEV, [box3D[0],box3D[2],box3D[5],box3D[4],ry])
		centers_plot, covs_plot = uncertain_label.calc_uncertainty_contour()
		#DEBUG_eigvals = np.zeros((covs_plot.shape[0],2))
		for i in range(covs_plot.shape[0]):
			draw_birdeye_ellipse(ax, covs_plot[i,:,:].reshape((2,2)), centers_plot[i,:].reshape(2), nstd=1, alpha=0.1)#1.0/covs_plot.shape[0])
	return

def annotate_IoUs(ax, IoU_dic, centers, offset = [0,2.0], give_name=True):
	c = 0
	for key in IoU_dic:
		c += 1
		for i in range(len(centers)):
			x = centers[i][0] + c * offset[0]
			y = centers[i][1] + c * offset[1]
			if give_name:
				IoU_text = 'net%d: '% (c)
			else:
				IoU_text = ''
			for IoU in IoU_dic[key][i]:
				IoU_text += '%1.2f, ' % (IoU)
			ax.text(x,y,IoU_text, size=11)

def write_JIoU(output_dir, networks, frames, all_gt_idxs, all_pd_idxs, all_IoU_dic):
	os.makedirs(output_dir, exist_ok=True)
	for net in networks:
		with open(os.path.join(output_dir, net+'_IoU_summary.txt'), 'w') as fo1:
			with open(os.path.join(output_dir, net+'_interest.txt'), 'w') as fo2:
				fo1.write('frame, gt_idx, det_idx,  IoU, JIoU(gt_unc&det_unc), JIoU(gt_unc&gt), JIoU(gt_unc&det), JIoU ratio\n')
				fo2.write('frame, gt_idx, det_idx,  IoU, JIoU(gt_unc&det_unc), JIoU(gt_unc&gt), JIoU(gt_unc&det), JIoU ratio\n')
				for idfile, frame in enumerate(frames):
					gt_idxs = all_gt_idxs[idfile][net]
					pd_idxs = all_pd_idxs[idfile][net]
					IoU_dics = all_IoU_dic[idfile][net]
					for i, gt_idx in enumerate(gt_idxs):
						pd_idx = pd_idxs[i]
						IoU_dic = IoU_dics[i]
						fo1.write('{:06d}, {:5d}, {:6d}, {:5.2f}, {:6.2f}, {:8.2f}, {:8.2f}, {:8.2f}\n'.format(frame, gt_idx, pd_idx, IoU_dic[0], IoU_dic[1], IoU_dic[2], IoU_dic[3], IoU_dic[1]/IoU_dic[2]))
						if IoU_dic[2]<0.7:
							fo2.write('{:06d}, {:5d}, {:6d}, {:5.2f}, {:6.2f}, {:8.2f}, {:8.2f}, {:8.2f}\n'.format(frame, gt_idx, pd_idx, IoU_dic[0], IoU_dic[1], IoU_dic[2], IoU_dic[3], IoU_dic[1]/IoU_dic[2]))


def rewrite_detection_aligned(output_dir, pred_data, label_data, hack_data):
	#rewrite the prediction to align it with its associated label (the gt object with highest IoU)
	#bounding box is aligned to the closest corner
	views = ['2D', 'G', '3D']
	difficulty = 'HARD'
	output_corner_dir = os.path.join(output_dir, 'corner_aligned', 'data')
	output_center_dir = os.path.join(output_dir, 'center_aligned', 'data')
	os.makedirs(output_corner_dir, exist_ok=True)
	os.makedirs(output_center_dir, exist_ok=True)
	if output_dir.find('waymo') != -1 or output_dir.find('Waymo') != -1:
		using_waymo = True
	else:
		using_waymo = False
	for frame in tqdm(range(len(pred_data['file_exists']))):
		if pred_data['file_exists'][frame]:
			if using_waymo:
				file1name = os.path.join(output_corner_dir, '{:08d}.txt'.format(frame))
				file2name = os.path.join(output_center_dir, '{:08d}.txt'.format(frame))
			else:
				file1name = os.path.join(output_corner_dir, '{:06d}.txt'.format(frame))
				file2name = os.path.join(output_center_dir, '{:06d}.txt'.format(frame))
			with open(file1name, 'w') as fo1, open(file2name, 'w') as fo2:
			#output a prediction file
				for det_idx in range(len(pred_data['box3D'][frame])):
					det_ry = pred_data['ry'][frame][det_idx]
					det_box3D = pred_data['box3D'][frame][det_idx]
					det_boxBEV = [det_box3D[0], det_box3D[2], det_box3D[5], det_box3D[4], det_ry]
					det_cornersBEV = center_to_corner_BEV(det_boxBEV)
					det_cornersBEV_dist = np.linalg.norm(det_cornersBEV, axis=1)
					det_box3D_corner_aligned = det_box3D.copy()
					det_box3D_center_aligned = det_box3D.copy()
					associated_gt = -1
					for view in views:
						if hack_data[difficulty]['det'][view]['gt'][frame][det_idx]>0:
							associated_gt = hack_data[difficulty]['det'][view]['gt'][frame][det_idx]
					if associated_gt > -1: #is associated with a ground truth
						#align the bounding box
						gt_idx = associated_gt
						gt_ry = label_data['ry'][frame][gt_idx]
						gt_box3D = label_data['box3D'][frame][gt_idx]
						gt_boxBEV = [gt_box3D[0], gt_box3D[2], gt_box3D[5], gt_box3D[4], gt_ry]
						gt_cornersBEV = center_to_corner_BEV(gt_boxBEV)
						#get the nearest corner
						det_corner_idx = np.argmin(det_cornersBEV_dist)
						#create corner aligned and center aligned boxes
						det_boxBEV_corner_aligned = align_BEV_to_corner(det_boxBEV, det_corner_idx, gt_boxBEV[2:4])
						det_box3D_corner_aligned[0] = det_boxBEV_corner_aligned[0]
						det_box3D_corner_aligned[2] = det_boxBEV_corner_aligned[1]
						det_box3D_corner_aligned[5] = det_boxBEV_corner_aligned[2]
						det_box3D_corner_aligned[4] = det_boxBEV_corner_aligned[3]
						det_box3D_center_aligned[5] = gt_box3D[5]
						det_box3D_center_aligned[4] = gt_box3D[4]
					#write the new prediction file
					det_box2D = pred_data['box2D'][frame][det_idx]
					det_score = pred_data['score'][frame][det_idx]
					if not using_waymo:
						y_center_aligned = det_box3D_center_aligned[1]
						z_center_aligned = det_box3D_center_aligned[2]
						y_corner_aligned = det_box3D_corner_aligned[1]
						z_corner_aligned = det_box3D_corner_aligned[2]
						ry = pred_data['ry'][frame][det_idx]
					else:
						# data converter for waymo, since it uses x,y,z center in LIDAR frame
						y_center_aligned = det_box3D_center_aligned[2]
						z_center_aligned = det_box3D_center_aligned[1]-det_box3D_center_aligned[3]/2
						y_corner_aligned = det_box3D_corner_aligned[2]
						z_corner_aligned = det_box3D_corner_aligned[1]-det_box3D_corner_aligned[3] / 2
						ry = -pred_data['ry'][frame][det_idx]
					fo1.write('{:s} {:d} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:.2f} {:.3f} {:.2f} {:.4f}\n'.format(pred_data['class'][frame][det_idx],
													-1, -1, pred_data['alpha'][frame][det_idx], det_box2D[0], det_box2D[1], det_box2D[2], det_box2D[3], 
													det_box3D_corner_aligned[3], det_box3D_corner_aligned[4], det_box3D_corner_aligned[5], 
													det_box3D_corner_aligned[0], y_corner_aligned, z_corner_aligned, ry, det_score
													))
					fo2.write('{:s} {:d} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:.2f} {:.3f} {:.2f} {:.4f}\n'.format(pred_data['class'][frame][det_idx],
													-1, -1, pred_data['alpha'][frame][det_idx], det_box2D[0], det_box2D[1], det_box2D[2], det_box2D[3], 
													det_box3D_center_aligned[3], det_box3D_center_aligned[4], det_box3D_center_aligned[5], 
													det_box3D_center_aligned[0], y_center_aligned, z_center_aligned, ry, det_score
													))



def get_corner_variances(label_data, frame, gt_idxs, points_cam):
	from metric_utils import label_inference_BEV, label_uncertainty_IoU, uncertain_prediction_BEVdelta, uncertain_prediction_BEV
	inference = label_inference_BEV(degree_register=1,gen_std=0.3, prob_outlier=0.03)
	uncertain_labels = []
	corner_totalVariances = []
	center_totalVariances = []
	LIDAR_variances = []
	for i, gt_idx in enumerate(gt_idxs):
		box3D = label_data['box3D'][frame][gt_idx]
		ry = label_data['ry'][frame][gt_idx]
		gt_boxBEV = [box3D[0],box3D[2],box3D[5],box3D[4],ry]
		points_clip = clip_by_BEV_box(points_cam, box3D[0:3], box3D[3:6], ry, buffer_size = 0.1)
		points_clip_BEV = points_clip[:,[0,2]]
		uncertain_labels.append(inference.infer(points_clip_BEV, [box3D[0],box3D[2],box3D[5],box3D[4],ry]))
		means, covs = uncertain_labels[i].calc_uncertainty_corners()
		#cov is 4x2x2, totalVariance is 4x1
		totalVariance = np.trace(covs,axis1=1,axis2=2)
		gt_cornersBEV = center_to_corner_BEV(gt_boxBEV)
		gt_cornersBEV_dist = np.linalg.norm(gt_cornersBEV, axis=1)
		sort_idx = np.argsort(gt_cornersBEV_dist)
		mean_center,cov_center = uncertain_labels[i].calc_uncertainty_points(np.array([[0,0]]))
		totalVariance_center = np.trace(cov_center,axis1=1,axis2=2)
		corner_totalVariances.append(totalVariance[sort_idx])
		center_totalVariances.append(totalVariance_center)
		if uncertain_labels[i].mean_LIDAR_variance:
			LIDAR_variances.append([uncertain_labels[i].mean_LIDAR_variance])
		else:
			LIDAR_variances.append([-1])
	#return a N*5 totalVariance matrix of each corner + a center, it is sorted according to the distance to origin (ascending)
	return np.concatenate((np.stack(corner_totalVariances,axis=0), np.stack(center_totalVariances,axis=0), np.stack(LIDAR_variances,axis=0)), axis=1)

def write_corner_variances(output_root, corner_totalVariances):
	#corner_totalVariances is N*6, which is [corner1, corner2, corner3, corner4, center, LIDAR_measurement_noise_variance(LIDAR_variance)]
	output_file = output_root+'Lshapes/uncertaintyV2/Waymo_0.3_0.03_corner_totalVariances_summary.txt'
	output_file2 = output_root + 'Lshapes/uncertaintyV2/Waymo_0.3_0.03_corner_totalVariances.txt'
	cTVs = np.vstack(corner_totalVariances)
	mean_cTV = np.mean(cTVs[:,0:5], axis=0)
	std_cTV = np.std(cTVs[:,0:5], axis=0)
	with open(output_file, 'w') as fo:
		fo.write('mean {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(mean_cTV[0], mean_cTV[1], mean_cTV[2], mean_cTV[3], mean_cTV[4]))
		fo.write('std  {:.2f} {:.2f} {:.2f} {:.2f}'.format(std_cTV[0], std_cTV[1], std_cTV[2], std_cTV[3], std_cTV[4]))
	with open(output_file2, 'w') as fo:
		fo.write('corner1 corner2 corner3 corner4 center LIDAR\n')
		for i in range(cTVs.shape[0]):
			fo.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.5f}\n'.format(cTVs[i,0],cTVs[i,1],cTVs[i,2],cTVs[i,3],cTVs[i,4],cTVs[i,5]))


def calc_mAPs(pred_data, label_data, metric, metric_thres=np.arange(0,1.01,0.01)):
	mAPs = np.zeros_like(metric_thres)
	