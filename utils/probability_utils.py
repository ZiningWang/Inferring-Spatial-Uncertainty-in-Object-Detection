import numpy as np
import os
import sys
from tqdm import tqdm
from utils.kitti_utils import boxes3d_to_corners3d


class cov_interp:
	#interpterlate covariance matrix of Var[X(w)] = cov[\Phi(Y)ww^T\Phi(Y)^T] where w is n*1 and Y is m*n (Var[X] is m*m)
	#Y is a fixed random variable matrix, w is the coordinate (input argument)
	def __init__(self, W, Vs):
		#W is tuple of k input w's
		#Vs are the k corresponding Var[X(w)]'s
		#the solved preprocessed matrix is Vinterp so that new Var[X(w)] = flatten(uppertriangle(ww^T))^T * Vinterp
		k = len(W)
		Vstack = np.zeros([k,Vs[0].size])
		self.m = Vs[0].shape[0]
		self.n = W[0].size
		wwstack = np.zeros([self.ww_flatten_size(),k])
		for i in range(k):
			w = W[i]
			ww = np.matmul(w.reshape([self.n,1]),w.reshape([1,self.n]))
			wwstack[:,i] = self.extract_upper_triangle_ww(ww)
			Vstack[i,:] = Vs[i].reshape([1,-1])
		#print(w, ww, self.extract_upper_triangle_ww(ww))
		#print(np.transpose(wwstack).shape, Vstack.shape)
		#print(np.linalg.eig(wwstack))
		#print(wwstack)
		self.Vinterp = np.linalg.solve(np.transpose(wwstack),Vstack)

	def interp(self,w):
		#a single point w: n*1
		ww = np.matmul(w.reshape([self.n,1]),w.reshape([1,self.n]))
		ww_flatten = self.extract_upper_triangle_ww(ww)
		Vout = ww_flatten*self.Vinterp
		return Vout.reshape([self.m,self.m])

	def interps(self,ws):
		#multiple points ws: k*n
		wws_flatten = self.extract_upper_triangle_ws(ws)
		Vouts = np.matmul(wws_flatten, self.Vinterp)
		return Vouts.reshape([ws.shape[0], self.m, self.m])

	def ww_flatten_size(self):
		return int(self.n*(self.n+1)/2)

	def extract_upper_triangle_ww(self,ww):
		return ww[np.triu_indices(self.n)]

	def extract_upper_triangle_ws(self,ws):
		k = ws.shape[0]
		wws_flatten = np.zeros([k,self.ww_flatten_size()])
		j0 = 0
		for i in range(self.n):
			wws_flatten[:,j0:j0+self.n-i] = ws[:,i:i+1] * ws[:,i:]
			j0 += self.n-i
		return wws_flatten


class cov_interp_BEV_full(cov_interp):
	#all the same, BEV requires k = 6 points (eg. 4 corners + 2 points on the side)
	def __init__(self, W, Vs):
		assert len(W) == 6
		cov_interp.__init__(self, W, Vs) 


class cov_interp_3D(cov_interp):
	#3D is a little bit different, because pitch and roll angle are not active
	#if take the full triangular matrix, would require 10 points
	#if ignore the pitch and roll, would require 8 points, eg. 6(BEV, say on the top) points + 2 bottom corner points
	def __init__(self, W, Vs):
		#for 3D, only take these cols points
		self.flatten_idx = [0,2,3,4,6,7,8,9]
		cov_interp.__init__(self, W, Vs)

	def ww_flatten_size(self):
		return len(self.flatten_idx)

	def extract_upper_triangle_ww(self,ww):
		ww_flatten = ww[np.triu_indices(self.n)]
		return ww_flatten[self.flatten_idx]

	def extract_upper_triangle_ws(self,ws):
		wws_flatten = cov_interp.extract_upper_triangle_ws(self,ws)
		return wws_flatten[:,self.flatten_idx]


#def bbox_uncertainty_to_spatial_uncertainty_BEV(boxBEV, std):
	#assume inputs are joint Gaussian, use importance sampling with student distribution to calculate variances of corners

def W_Vs_3D_to_BEV_full(W_3D, Vs_diag_3D):
	assert len(Vs_diag_3D.shape) == 3, 'The Vs shape should be n*8*3.'
	W_BEV = W_3D[[0,1,2,4,5,7], :][:, [0,2,3]]
	Vs_BEV = np.zeros([Vs_diag_3D.shape[0], 6, 2, 2])
	for i in range(Vs_diag_3D.shape[0]):
		Vs_diag = Vs_diag_3D[i]
		Vs = np.array([np.diag(Vs_diag[i]) for i in range(Vs_diag.shape[0])])
		Vs_BEV[i] = Vs[[0,1,2,4,5,7], :, :][:, [0,2], :][:, :, [0,2]]
	return W_BEV, Vs_BEV

def Hujie_uncertainty_reader(pred_dir, file_lists, max_num=7518):
	import pickle
	pickle_dir = pred_dir + 'unc_data/'
	attrs = ['box3D','corners','points_unc','ry','homogeneous_w','uncertainty_format']
	unc_data = {attr:[[] for name in range(max_num)] for attr in attrs}
	print('reading data from: {}'.format(pickle_dir))
	for file in tqdm(file_lists):
		file_num = file.split('.')[0]
		name = file_num
		frame = int(name)
		if int(float(name)) >= max_num:
			break
		corner_unc_file_dir = os.path.join(pickle_dir,"{}_UNC.pickle".format(file_num))
		unc_info = pickle.load(open(corner_unc_file_dir,"rb"))
		# box3D is [x, y_top, z, h, w, l]
		unc_data['box3D'][frame] = unc_info['bbox3d'][:,0:6]
		unc_data['ry'][frame] = unc_info['bbox3d'][:,6]
		unc_data['corners'][frame] = boxes3d_to_corners3d(unc_info['bbox3d'])
		if 'full' in pickle_dir:
			unc_data['homogeneous_w'][frame] = np.array([[0.5,-1,0.5,1],[0.5,-1,-0.5,1],[-0.5,-1,-0.5,1],[0.5,0,-0.5,1],[0.5,-1,0,1],[0,-1,-0.5,1],[0.5,-0.5,-0.5,1],[-0.5,-1,0.5,1]])  #8*3
			unc_data['uncertainty_format'] = 'full'
		else:
			print('The uncertainty configuration cannot infer full covariance distribution inside the bounding box!')
			assert False, 'The warning is treated as error for now.'
			unc_data['homogeneous_w'][frame] = np.array([[0.5,-1,0.5,1],[0.5,-1,-0.5,1],[-0.5,-1,-0.5,1],[-0.5,-1,0.5,1],[0.5,0,0.5,1],[0.5,0,-0.5,1],[-0.5,0,-0.5,1],[-0.5,-0.5,0.5,1]])  #8*3
			unc_data['uncertainty_format'] = 'corner'
		unc_data['points_unc'][frame] = unc_info['bbox3d_sigma'].reshape([-1,8,3])
	return unc_data

def Hujie_unctainty_to_prediction():
	# This is to transform the uncertainty data to Kitti prediction format.