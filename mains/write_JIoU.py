# This is a simplified version from simple_stats aimed for only generating JIoUs, but with the support for other prediction uncertainty form
import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../'))
import utils.metric_utils#probability_utils
from utils.probability_utils import Hujie_uncertainty_reader as unc_reader
from utils.probability_utils import cov_interp_3D 
from utils.simple_stats import label_reader, detect_reader, hacker_reader, evaluate_new_IoU, get_dirs

# Labels and data
actual_test = False
data_dir = '/mnt/d/Berkeley/Kitti_data/object' #'/home/msc/KITTI/object' #'/data/RPN/coop_DiFeng/WaymoData' #
label_uncertainty_subfolder =  'summary/uncertaintyV3_for_labelwellness_0.25_0.8_deg1/'

# Detection results
networks = ['PointRCNN_unc_full']
pred_3d_dirs = {}
pred_3d_dirs['PointRCNN_unc_full'] = ('/mnt/d/Berkeley/Kitti_data/predictions/Hujie_unc_full_val50/')

def main():
	# (1) get file list, label and images
	list_dir, img_dir, lidar_dir, calib_dir, label_3d_dir = get_dirs(data_dir, actual_test)

	num_net = len(networks)
	pred_datas = [[] for net in networks]
	hack_datas = [[] for net in networks]
	unc_datas = [[] for net in networks]

	output_root = './results/'
	output_dir = output_root + 'results_visualizer_new'
	for net in networks:
		output_dir += '_'+net
	output_dir += '/'
	os.makedirs(output_dir, exist_ok=True)
	# Read data from filelist.
	with open(list_dir) as fi:
		file_lists = fi.read().splitlines() 
	file_lists.sort()
	file_lists = [file_lists[0]]

	label_data = label_reader(label_3d_dir,file_lists,calib_dir)
	for inet, net in enumerate(networks):
		hack_datas[inet] = hacker_reader(pred_3d_dirs[net], file_lists)
		pred_datas[inet] = detect_reader(pred_3d_dirs[net], file_lists)
		if 'unc' in pred_3d_dirs[net]:
			unc_datas[inet] = unc_reader(pred_3d_dirs[net], file_lists)
		else:
			unc_datas[inet] = None

	#all labels, newIoUs vs IoU
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
			output_newIoU_dir = output_dir + net + '_newIoUs/'
			os.makedirs(output_newIoU_dir, exist_ok=True)
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

				NewIoUs = evaluate_new_IoU(label_data, pred_datas[inet], frame, gt_idxs, pd_idxs, points_cam, grid_size=0.1, sample_grid=0.02, unc_data=unc_datas[inet])
				for iI in range(len(IoU)):
					tmp = [IoU[iI]] + NewIoUs[iI]
					all_IoU_dic[id_file][net].append(tmp)

		if active_gt_idxs.size>0:
			pass
			#corner_totalVariances.append(get_corner_variances(label_data, frame, gt_idxs, points_cam))
			#plot_multiview_label(label_data, img_dir, points_cam, frame, gt_idxs, output_newIoU_dir, IoU_dic=all_IoU_dic[id_file], pd_idxs=pd_idxs, pred_data=pred_datas[len(pred_datas)-1])
	output_newIoU_dir = output_dir + label_uncertainty_subfolder
	write_NewIoU(output_newIoU_dir, networks, frames, all_gt_idxs, all_pd_idxs, all_IoU_dic)
	#write_corner_variances(output_root, corner_totalVariances)


if __name__ == '__main__':
	main()