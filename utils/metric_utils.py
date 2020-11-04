import numpy as np
import math
from scipy.stats import multivariate_normal
from utils.paul_geometry import z_normed_corners_default, center_to_corner_BEV
from utils.probability_utils import cov_interp_BEV_full

##############################################################
### Utilities
##############################################################
def create_BEV_box_sample_grid(sample_grid):
	x = np.arange(-0.5 + sample_grid / 2, 0.5, sample_grid)
	y = x
	zx, zy = np.meshgrid(x, y)
	z_normed = np.concatenate((zx.reshape((-1, 1)), zy.reshape((-1, 1))), axis=1)
	return z_normed
##############################################################
### Inference of Label Uncertainty class ###
##############################################################

class label_inference:
	def __int__(self):
		# x is label, y is observation
		self.description = 'a template of label inference'

	def generator(self, x, y):
		# return p(y|x)
		prob_y_x = 0
		return prob_y_x

	def register(self, y, z):
		# return a matrix of association between y and z, or p(z|y)
		prob_z_y = 0
		return prob_z_y

	def infer(self, prob_y_x, prob_z_y):
		# return the reference
		prob_x_y = 0
		return prob_x_y


class label_inference_BEV(label_inference):
	def __init__(self, degree_register=0, gen_std=0.2, prob_outlier=0.1, eps_prior_scaler=0.04, boundary_sample_interval=0.1):
		#eps_prior is to prevent deprecated posterior variance. It the posterior covariance is not full rank, add some epsilon of prior
		self.description = 'use Gaussian model to get posterior covariance of labels'
		# l0=3.89+-0.44, w0=1.63+-0.10, h0=1.52+-0.13
		# self.dim_mean0 = [3.89,1.63]
		self.generator_std = gen_std  # GMM std, in meter
		self.boundary_sample_interval = boundary_sample_interval
		# the prob_outlier should be normalized by the number of registration
		self.prob_outlier = 2*np.pi*self.generator_std**2*prob_outlier/(1-prob_outlier)*(2*degree_register+1)#prob_outlier * (2*degree_register+1) / 3
		self.eps_prior_scaler = eps_prior_scaler
		# use Variantional Bayes to allow analytical result with multiple register
		self.degree_register = degree_register
		self.z_coords = [z_coord * self.boundary_sample_interval for z_coord in
						 range(-self.degree_register, self.degree_register + 1)]

		self.num_z_y = 1 + 2 * degree_register  # maximum number of registered z given y
		self.feature_dim = 6
		self.space_dim = 2 
		self.surface_dim = self.space_dim-1

	def generator(self, z_surface, y):
		# modeled as Gaussian, only store the first and second order term (discard constant term)
		ny = y.shape[0]
		K = z_surface.shape[2]
		prob_y_x_quad = np.zeros((ny, K, self.feature_dim, self.feature_dim))
		prob_y_x_lin = np.zeros((ny, K, 1, self.feature_dim))
		var_generator = self.generator_std ** 2
		for ik in range(K):
			Jacobians_point = self.label.Jacobian_surface(z_surface[:, :, ik])
			# norm(y-Jacobian*x_feature, 2)/self.generator_std^2
			for iy in range(ny):
				Jacobian_point = np.squeeze(Jacobians_point[iy, :, :])
				prob_y_x_quad[iy, ik, :, :] = np.matmul(Jacobian_point.transpose(), Jacobian_point) / var_generator
				prob_y_x_lin[iy, ik, :, :] = -2 * np.matmul(y[iy, :], Jacobian_point) / var_generator

		return (prob_y_x_quad, prob_y_x_lin)

	def register(self, y, z=None):
		# y: ny*2
		ny = y.shape[0]
		nz = self.num_z_y
		prob_z_y = np.zeros((ny, nz))
		yz_distances = np.zeros((ny,nz))
		z_surface_out = np.zeros((ny,self.surface_dim, nz))

		yc = np.matmul(y - self.label.x0[0:self.space_dim], self.label.rotmat)
		zc = yc / self.label.x0[self.space_dim:2*self.space_dim]  # transposed twice
		zc_norm = np.abs(zc)
		z_embed = zc / zc_norm.max(1, keepdims=True) * self.label.x0[self.space_dim:2*self.space_dim] / 2  # z in 3D space, map to surface
		z_surface = self.label.project(self.label.x0, z_embed)  # z in surface_manifold

		var_generator = self.generator_std ** 2
		for iz, z_coord in enumerate(self.z_coords):
			z_surface_out[:, :, iz] = self.label.add_surface_coord(z_surface, z_coord).reshape((ny,self.surface_dim))
			z_embeded = self.label.embedding(self.label.x0, z_surface_out[:,:,iz])
			yz_distances[:, iz] = np.linalg.norm(yc - z_embed, 2, axis=1)
			prob_z_y[:, iz] = np.exp(-yz_distances[:, iz]**2 / 2 / var_generator)

		prob_inlier = np.sum(prob_z_y, axis=1)
		for iz in range(len(self.z_coords)):
			prob_z_y[:, iz] = prob_z_y[:, iz] / (prob_inlier + self.prob_outlier)
		return prob_z_y, z_surface_out, yz_distances

	def infer(self, y, boxBEV, prior_scaler=0):
		# only one z possible z for each y, so prob_z_y no need
		self.label = self.construct_uncertain_label(boxBEV) 
		if y.shape[0] < self.space_dim+1:
			#too few observations, directly use prior
			self.label.set_posterior(self.label.x0, self.return_prior_precision_matrix())
			return self.label
		else:
			prob_z_y, z_surface, yz_distances = self.register(y)
			prob_y_x_quad, prob_y_x_lin = self.generator(z_surface, y)
			ny = prob_y_x_quad.shape[0]
			# get Gaussian of feature vector
			Q = np.squeeze(np.tensordot(prob_z_y, prob_y_x_quad, axes=((0, 1), (0, 1))))  # self.feature_dim*self.feature_dim
			P = np.tensordot(prob_z_y, prob_y_x_lin, axes=((0, 1), (0, 1))).reshape((self.feature_dim, 1))
			Q += prior_scaler*self.label.Q0_feature #scale the prior of variance, default is 0, which is no prior
			if np.linalg.matrix_rank(Q,tol=2.5e-3) < Q.shape[0]:
				Q += self.eps_prior_scaler * self.label.Q0_feature
			self.label.set_posterior(np.linalg.solve(-2 * Q, P), np.linalg.inv(Q))
			#estimate the LIDAR measurement noise
			mean_variance = np.sum(np.sum(prob_z_y*yz_distances**2, axis=1), axis=0)/np.sum(np.sum(prob_z_y, axis=1),axis=0)/2
			self.label.mean_LIDAR_variance = mean_variance
			return self.label

	def return_prior_precision_matrix(self):
		return np.linalg.inv(self.eps_prior_scaler*self.label.Q0_X)
	def construct_uncertain_label(self, boxBEV):
		return uncertain_label_BEV(boxBEV)

class label_inference_3D(label_inference_BEV):
	def __init__(self, degree_register=0, gen_std=0.2, prob_outlier=0.1, eps_prior_scaler=0.04, boundary_sample_interval=0.1):
		label_inference_BEV.__init__(self, degree_register, gen_std, prob_outlier, eps_prior_scaler, boundary_sample_interval)
		self.z_coords = []
		for zBEV_coord in range(-self.degree_register, self.degree_register + 1):
			self.z_coords.append(np.array([zBEV_coord*self.boundary_sample_interval, 0]))
		for zH_coord in range(-self.degree_register, 0):
			self.z_coords.append(np.array([0, zH_coord*self.boundary_sample_interval]))
		for zH_coord in range(1, self.degree_register + 1):
			self.z_coords.append(np.array([0, zH_coord*self.boundary_sample_interval]))


		self.num_z_y = 1 + 4 * degree_register  # neighborhood points in 2D surface (axis aligned)
		self.feature_dim = 8 #BEV + y center + height
		self.space_dim = 3
		self.surface_dim = self.space_dim-1
		
	def return_prior_precision_matrix(self):
		return np.linalg.inv(self.eps_prior_scaler*self.label.Q0_feature)
	def construct_uncertain_label(self, box3D):
		return uncertain_label_3D(box3D)



##############################################################
### JIoU Calculation class ###
##############################################################

class label_uncertainty_IoU:
	def __init__(self, grid_size=0.1, range=3.0):
		self.grid_size = grid_size
		self.range = range  # sample upto how many times of the size of the car

	def calc_IoU(self, uncertain_label, pred_boxBEVs, sample_grid=0.1):
		sample_points, _ = self.get_sample_points(uncertain_label)
		px = uncertain_label.sample_prob(sample_points, sample_grid=sample_grid)
		JIoUs = []
		for i in range(len(pred_boxBEVs)):
			py = pred_boxBEVs[i].sample_prob(sample_points, sample_grid=sample_grid)
			if np.sum(py) > 0 and np.sum(px) > 0:
				JIoUs.append(self.Jaccard_discrete(px, py).item())
			else:
				JIoUs.append(0)
		return JIoUs

	def get_sample_points(self, uncertain_label):
		x = np.arange(-self.range / 2 + self.grid_size / 2, self.range / 2, self.grid_size)
		y = x
		zx, zy = np.meshgrid(x, y)
		z_normed = np.concatenate((zx.reshape((-1, 1)), zy.reshape((-1, 1))), axis=1)
		# samples are aligned with label bounding box
		sample_points = np.matmul(z_normed * uncertain_label.x0[2:4],
								  uncertain_label.rotmat.transpose()) + uncertain_label.x0[0:2]
		return sample_points, np.array((x.shape[0], y.shape[0]))
	
	def Jaccard_discrete(self, px, py):
		# Yiyang's Implementation
		similarity = np.array(0)
		sort_index = np.argsort(px / (py+np.finfo(np.float64).eps))
		px = px[sort_index]
		py = py[sort_index]

		px_sorted_sum = np.zeros(px.size)
		py_sorted_sum = np.zeros(py.size)

		py_sorted_sum[0] = py[0]

		for i in range(1, px.size):
			px_sorted_sum[px.size - i - 1] = px_sorted_sum[px.size - i] + px[px.size - i]
			py_sorted_sum[i] = py_sorted_sum[i - 1] + py[i]

		idx = np.argwhere((px > 0) & (py > 0))
		for i in idx:
			x_y_i = px[i] / py[i]
			temp = px[i] / (px_sorted_sum[i] + x_y_i * py_sorted_sum[i])
			similarity = similarity + temp

		return similarity


##############################################################
### Uncertain Label class ###
##############################################################

class uncertain_label:
	def __init__(self, box, prior_std):
		self.description = "a template of uncertain label"
		self.posterior = None

	def embedding(self, z_surface):
		# embed surface points from surface coord to Euclidean coord
		z_embed = []
		return z_embed

	def project(self, z_embed):
		# project surface points from Euclidean coord to surface coord
		z_surface = []
		return z_surface

	def sample_prob(self, points):
		# sample the spatial distribution at points
		prob = []
		return prob

class uncertain_label_BEV(uncertain_label):
	def __init__(self, boxBEV, x_std=[0.44, 0.10, 0.25, 0.25, 0.1745]):
		self.x0 = np.array(boxBEV).reshape(5)
		# reverse ry so that the rotation is x->z instead of z->x normally
		self.x0[4] *= -1
		ry = self.x0[4]
		self.rotmat = np.array([[math.cos(ry), -math.sin(ry)], [math.sin(ry), math.cos(ry)]])
		self.feature0 = np.array(
			[self.x0[0], self.x0[1], self.x0[2] * math.cos(ry), self.x0[2] * math.sin(ry), self.x0[3] * math.cos(ry),
			 self.x0[3] * math.sin(ry)])
		# initialize prior uncertainty
		self.dim_std = [x_std[0], x_std[1]]  # l,w
		self.pos_std = [x_std[2], x_std[3]]  # assume to be the same as dimension?
		self.ry_std = [x_std[4]]  # rotation, about 10deg.
		self.Q0_feature = np.diag(1 / np.array([self.pos_std[0], self.pos_std[1],
												self.dim_std[0] * abs(math.cos(ry)) + self.x0[2] * self.ry_std[0] * abs(math.sin(ry)),
												self.dim_std[0] * abs(math.sin(ry)) + self.x0[2] * self.ry_std[0] * abs(math.cos(ry)),
												self.dim_std[1] * abs(math.cos(ry)) + self.x0[3] * self.ry_std[0] * abs(math.sin(ry)),
												self.dim_std[1] * abs(math.sin(ry)) + self.x0[3] * self.ry_std[0] * abs(math.cos(ry))]) ** 2)
		self.Q0_X = np.diag(
			1 / np.array([self.pos_std[0], self.pos_std[1], self.dim_std[0], self.dim_std[1], self.ry_std[0]]) ** 2)
		self.posterior = None
		self.mean_LIDAR_variance = None

	def set_posterior(self, mean, cov):
		# assert (cov.shape[0]==6),  "must set posterior of feature vector!"
		self.posterior = (mean, cov)

	def get_corner_norms(self):
		corner_norms = z_normed_corners_default
		return corner_norms

	def embedding(self, x0, z_surface):
		#x0 = [x,z,l,w,ry]
		# surface is 1D in BEV, from 0 to 4 ([-l,-w]/2->[l,-w]/2->[l,w]/2->[-l,w]/2), l along 1st axis
		assert(x0.shape[0]==5)
		z_surface[z_surface > 4] -= 4
		z_surface[z_surface < 0] += 4
		z_surface.reshape(-1)
		z_embed = np.zeros((z_surface.shape[0], 2))
		temp_idx = np.squeeze(z_surface >= 3)
		z_embed[temp_idx, 1] = ((-(z_surface[temp_idx] - 3) + 0.5) * x0[3]).reshape(-1)
		z_embed[temp_idx, 0] = -x0[2] / 2
		temp_idx = np.squeeze(np.logical_and(z_surface >= 2, z_surface < 3))
		z_embed[temp_idx, 1] = x0[3] / 2
		z_embed[temp_idx, 0] = ((-(z_surface[temp_idx] - 2) + 0.5) * x0[2]).reshape(-1)
		temp_idx = np.squeeze(np.logical_and(z_surface >= 1, z_surface < 2))
		z_embed[temp_idx, 1] = (((z_surface[temp_idx] - 1) - 0.5) * x0[3]).reshape(-1)
		z_embed[temp_idx, 0] = x0[2] / 2
		temp_idx = np.squeeze(np.logical_and(z_surface >= 0, z_surface < 1))
		z_embed[temp_idx, 1] = -x0[3] / 2
		z_embed[temp_idx, 0] = ((z_surface[temp_idx] - 0.5) * x0[2]).reshape(-1)
		return z_embed

	def project(self, x0, z_embed):
		#x0 = [x,z,l,w,ry]
		assert(x0.shape[0]==5)
		z_normed = z_embed / x0[2:4]
		amax = np.argmax(np.abs(z_normed), axis=1)
		z_surface = np.zeros((z_normed.shape[0]))

		temp_idx = np.squeeze(np.logical_and(amax == 1, z_normed[:, 1] < 0))
		z_surface[temp_idx] = z_normed[temp_idx, 0] + 0.5
		temp_idx = np.squeeze(np.logical_and(amax == 0, z_normed[:, 0] >= 0))
		z_surface[temp_idx] = z_normed[temp_idx, 1] + 0.5 + 1
		temp_idx = np.squeeze(np.logical_and(amax == 1, z_normed[:, 1] >= 0))
		z_surface[temp_idx] = -z_normed[temp_idx, 0] + 0.5 + 2
		temp_idx = np.squeeze(np.logical_and(amax == 0, z_normed[:, 0] < 0))
		z_surface[temp_idx] = -z_normed[temp_idx, 1] + 0.5 + 3
		return z_surface

	def add_surface_coord(self, z_surface, z_coord):
		return z_surface+z_coord


	def Jacobian_surface(self, z_surface):
		# parameterize surface point w.r.t. BEV box feature vector
		# the feature vector is (xc1, xc2, cos(ry)*l, sin(ry)*l, cos(ry)*w, sin(ry)*w), 1*6
		z_normed = self.embedding(self.x0, z_surface) / self.x0[2:4]
		Jacobian = self.Jacobian_z_normed(z_normed)
		return Jacobian

	def Jacobian_z_normed(self, z_normed):
		# parameterize surface point w.r.t. BEV box feature vector
		# the feature vector is (xc1, xc2, cos(ry)*l, sin(ry)*l, cos(ry)*w, sin(ry)*w), 1*6
		nz = z_normed.shape[0]
		Jacobian = np.zeros((nz, 2, 6))
		Jacobian[:, 0, 0] = 1.0
		Jacobian[:, 1, 1] = 1.0
		Jacobian[:, 0, 2] = z_normed[:, 0]
		Jacobian[:, 1, 3] = z_normed[:, 0]
		Jacobian[:, 1, 4] = z_normed[:, 1]
		Jacobian[:, 0, 5] = -z_normed[:, 1]
		return Jacobian

	def sample_boundary_points(self):
		z_surface = np.arange(0, 4, 0.05)
		z_embed = self.embedding(self.x0, z_surface)
		z_normed = z_embed / self.x0[2:4]
		return z_surface, z_embed, z_normed

	def calc_uncertainty_box(self, std=1):
		#by default, calculate the contour of 1 standard deviation by variance along the boundary of bounding box
		z_normed = np.array([[0.5,0],[0,0.5],[-0.5,0],[0,-0.5]])
		centers, covs = self.calc_uncertainty_points(z_normed)
		#get variance along the direction to center
		direction = centers - self.x0[0:2]
		lengths = np.linalg.norm(direction, axis=1, keepdims=True)
		direction = direction / lengths
		#get a n*1 distance array
		dists2 = np.sum(np.sum(covs*np.expand_dims(direction,axis=2),axis=1)*direction, axis=1).reshape([-1,1])
		dists = np.sqrt(dists2)
		ratio = 1+dists*std/lengths
		outer_corners = self.x0[0:2] + direction[[0,0,2,2],:]*lengths[[0,0,2,2]]*ratio[[0,0,2,2]] + direction[[3,1,1,3],:]*lengths[[3,1,1,3]]*ratio[[3,1,1,3]]
		inner_corners = self.x0[0:2] + direction[[0,0,2,2],:]*lengths[[0,0,2,2]]/ratio[[0,0,2,2]] + direction[[3,1,1,3],:]*lengths[[3,1,1,3]]/ratio[[3,1,1,3]]
		return inner_corners, outer_corners

	def calc_uncertainty_contour_boundary(self, std=1):
		#by default, calculate the contour of 1 standard deviation by variance along the boundary of bounding box
		_, _, z_normed = self.sample_boundary_points()
		centers, covs = self.calc_uncertainty_points(z_normed)
		#get variance along the direction to center
		direction = centers - self.x0[0:2]
		lengths = np.linalg.norm(direction, axis=1, keepdims=True)
		direction = direction / lengths
		#get a n*1 distance array
		dists2 = np.sum(np.sum(covs*np.expand_dims(direction,axis=2),axis=1)*direction, axis=1).reshape([-1,1])
		dists = np.sqrt(dists2)
		ratio = 1+dists*std/lengths
		outer_contour_boundary = self.x0[0:2] + direction*ratio*lengths
		inner_contour_boundary = self.x0[0:2] - direction/ratio*lengths
		return inner_contour_boundary, outer_contour_boundary 

	def calc_uncertainty_contour(self):
		# cov is for self.label.x0
		_, _, z_normed = self.sample_boundary_points()
		# cov at the surface point, than return the covariance of these points

		return self.calc_uncertainty_points(z_normed)

	def calc_uncertainty_corners(self):
		return self.calc_uncertainty_points(self.get_corner_norms())

	def calc_uncertainty_points(self, z_normed):
		nz = z_normed.shape[0]
		cov = self.posterior[1]
		Jacobians_point = self.Jacobian_z_normed(z_normed)
		covs_out = np.zeros((z_normed.shape[0], 2, 2))
		centers_out = np.matmul(z_normed * self.x0[2:4], self.rotmat.transpose()) + self.x0[0:2]
		if cov.shape[0] == 6:
			for i in range(nz):
				Jacobian_point = np.squeeze(Jacobians_point[i, :, :])
				covs_out[i, :, :] = (Jacobian_point @ cov) @ Jacobian_point.transpose()
		elif cov.shape[0] == 5:
			lin_trans = np.zeros((2, 5))
			lin_trans[0, 0] = 1
			lin_trans[1, 1] = 1
			for i in range(nz):
				lin_trans[0, 2] = z_normed[i, 0]
				lin_trans[1, 3] = z_normed[i, 1]
				covs_out[i, :, :] = (lin_trans @ cov) @ lin_trans.transpose()
		return centers_out, covs_out

	def sample_prob(self, points, sample_grid=0.1):
		# sample probablity given points
		nk = points.shape[0]
		cov = self.posterior[1]
		probs = np.zeros(nk)
		if np.max(np.sqrt(np.linalg.eig(cov)[0]))<sample_grid*5:
			#empirical coeffienct, if cov too low, then sample_grid is not thick enough. Just assume the distribution is uniform (delta uncertainty)
			l2 = self.x0[2] / 2
			w2 = self.x0[3] / 2
			points_aligned_to_pred = np.matmul(points - self.x0[0:2], self.rotmat)
			clip_idx = points_aligned_to_pred[:, 0] >= -l2
			clip_idx = np.logical_and(clip_idx, points_aligned_to_pred[:, 0] < l2)
			clip_idx = np.logical_and(clip_idx, points_aligned_to_pred[:, 1] >= -w2)
			clip_idx = np.logical_and(clip_idx, points_aligned_to_pred[:, 1] < w2)
			# calculate JIoU with discrete sample
			n_in = np.sum(clip_idx)
			if n_in > 0:
				probs[clip_idx] = 1 / n_in
		else:
			#use the definition we proposed for spatial distribution
			z_normed = create_BEV_box_sample_grid(sample_grid)
			centers_out, covs_out = self.calc_uncertainty_points(z_normed)
			for i in range(covs_out.shape[0]):
				tmp_probs = multivariate_normal.pdf(points, mean=centers_out[i, :], cov=covs_out[i, :, :])
				# exponents = np.sum((points-centers_out[i,:])*((points-centers_out[i,:])@ np.linalg.inv(covs_out[i,:,:])),axis=1)/-2
				# tmp_probs = np.exp(exponents)
				probs += tmp_probs
			probs /= np.sum(probs)
		return probs

class uncertain_label_3D(uncertain_label):
	# A minimum unccertain_label_3D. TODO: implement sample_prob member
	def __init__(self, box3D, x_std=[0.44, 0.10, 0.10, 0.25, 0.25, 0.25, 0.1745]):
		# input box3D is xc,y_top,zc,h,w,l,ry in KITTI format
		# x0 is xc,yc,zc,l,h,w 
		self.x0 = np.array(box3D).reshape(7)
		self.x0[3:6] = self.x0[[5,3,4]]
		self.x0[1] = self.x0[1]-self.x0[4]/2
		ry = self.x0[6]
		self.rotmat = np.array([[math.cos(ry), 0, -math.sin(ry)], [0,1,0], [math.sin(ry), 0, math.cos(ry)]])
		self.feature0 = np.array(
			[self.x0[0], self.x0[1], self.x0[2], self.x0[3] * math.cos(ry), self.x0[3] * math.sin(ry), self.x0[5] * math.cos(ry),
			 self.x0[5] * math.sin(ry), self.x0[4]])
		# initialize prior uncertainty
		self.dim_std = [x_std[0], x_std[1], x_std[2]]  # l,h,w
		self.pos_std = [x_std[3], x_std[4], x_std[5]]  # assume to be the same as dimension?
		self.ry_std = [x_std[4]]  # rotation, about 10deg.
		self.Q0_feature = np.diag(1 / np.array([self.pos_std[0], self.pos_std[1], self.pos_std[2],
												self.dim_std[0] * abs(math.cos(ry)) + self.x0[3] * self.ry_std[0] * abs(math.sin(ry)),
												self.dim_std[0] * abs(math.sin(ry)) + self.x0[3] * self.ry_std[0] * abs(math.cos(ry)),
												self.dim_std[2] * abs(math.cos(ry)) + self.x0[4] * self.ry_std[0] * abs(math.sin(ry)),
												self.dim_std[2] * abs(math.sin(ry)) + self.x0[4] * self.ry_std[0] * abs(math.cos(ry)),
												self.dim_std[1]]) ** 2)
		self.posterior = None
		self.mean_LIDAR_variance = None

	def set_posterior(self, mean, cov):
		# assert (cov.shape[0]==6),  "must set posterior of feature vector!"
		self.posterior = (mean, cov)

	def embedding(self, x0, z_surface):
		#x0=[x,y,z,l,h,w,ry]
		# surface is 1D in BEV, from 0 to 4 ([-l,-w]/2->[l,-w]/2->[l,w]/2->[-l,w]/2), l along 1st axis
		# height is [-h, 0]
		z_embed = np.zeros((z_surface.shape[0], 3))
		x0_BEV = x0[[0,2,3,5,6]]
		z_embed[:,[0,2]] = uncertain_label_BEV.embedding(self, x0_BEV, z_surface[:,0])
		z_surface[z_surface[:,1]<-0.5,1] += 1
		z_surface[z_surface[:,1]> 0.5,1] -= 1 
		z_embed[:,1] = z_surface[:,1]*x0[4]
		return z_embed

	def project(self, x0, z_embed):
		z_surface = np.zeros((z_embed.shape[0],2))
		x0_BEV = x0[[0,2,3,5,6]]
		z_surface[:,0] = uncertain_label_BEV.project(self, x0_BEV, z_embed[:,[0,2]])
		z_surface[:,1] = z_embed[:,1]/x0[4]

		return z_surface

	def add_surface_coord(self, z_surface, z_coord):
		z_surface_added = z_surface+z_coord
		z_surface_added[:,1] = np.minimum(-0.5,np.maximum(z_surface_added[:,1],0.5))
		return z_surface_added

	def Jacobian_z_normed(self, z_normed):
		# parameterize surface point w.r.t. BEV box feature vector
		# the feature vector is (x, y, z, cos(ry)*l, sin(ry)*l, cos(ry)*w, sin(ry)*w, h), 1*8
		nz = z_normed.shape[0]
		Jacobian = np.zeros((nz, 3, 8))
		Jacobian[:, 0, 0] = 1.0
		Jacobian[:, 1, 1] = 1.0
		Jacobian[:, 2, 2] = 1.0
		Jacobian[:, 0, 3] = z_normed[:, 0]
		Jacobian[:, 1, 4] = z_normed[:, 0]
		Jacobian[:, 1, 5] = z_normed[:, 2]
		Jacobian[:, 0, 6] = -z_normed[:, 2]
		Jacobian[:, 2, 7] = z_normed[:, 1]
		return Jacobian

	def Jacobian_surface(self, z_surface):
		# parameterize surface point w.r.t. BEV box feature vector
		z_normed = self.embedding(self.x0, z_surface) / self.x0[3:6]
		Jacobian = self.Jacobian_z_normed(z_normed)
		return Jacobian

	def calc_uncertainty_corners(self):
		z_normed = np.array([[0.5,0.5,0.5,1],[0.5,0.5,-0.5,1],[-0.5,0.5,-0.5,1],[-0.5,0.5,0.5,1],[0.5,-0.5,0.5,1],[0.5,-0.5,-0.5,1],[-0.5,-0.5,-0.5,1],[-0.5,-0.5,0.5,1]])
		nz = z_normed.shape[0]
		cov = self.posterior[1]
		Jacobians_point = self.Jacobian_z_normed(z_normed)
		covs_out = np.zeros((z_normed.shape[0], 3, 3))
		assert(cov.shape[0] == 8)
		for i in range(nz):
			Jacobian_point = np.squeeze(Jacobians_point[i, :, :])
			covs_out[i, :, :] = (Jacobian_point @ cov) @ Jacobian_point.transpose()
		return covs_out

##############################################################
### Uncertain Prediction class ###
##############################################################

class uncertain_prediction():
	def __init__(self, box):
		self.description = "a template of uncertain prediction"
		self.posterior = None

	def sample_prob(self, points):
		prob = []
		return prob

class uncertain_prediction_BEVdelta(uncertain_prediction):
# actually a deterministic prediction, just to unify the API
	def __init__(self, boxBEV):
		self.boxBEV = boxBEV
		self.x0 = np.array(boxBEV).reshape(5)
		# reverse ry so that the rotation is x->z instead of z->x normally
		self.x0[4] *= -1
		self.ry = self.x0[4]
		self.cry = math.cos(self.ry)
		self.sry = math.sin(self.ry)
		self.rotmat = np.array([[self.cry, -self.sry], [self.sry, self.cry]])

	def get_corner_norms(self):
		corner_norms = z_normed_corners_default
		return corner_norms

	def calc_uncertainty_corners(self, std=1):
		return self.calc_uncertainty_points(self.get_corner_norms())

	def calc_uncertainty_box(self, std=1):
		#by default, calculate the contour of 1 standard deviation by variance along the boundary of bounding box
		corners = center_to_corner_BEV(self.boxBEV)
		return corners, corners

	def calc_uncertainty_points(self, std=1):
		assert False, 'This must be overloaded by the child class.'
		return None

	def sample_prob(self, points, sample_grid=0.1):

		return self.sample_prob_delta(points)

	def sample_prob_delta(self, points, sample_grid=0.1):
		# sample probablity given points
		nk = points.shape[0]
		probs = np.zeros(nk)
		l2 = self.x0[2] / 2
		w2 = self.x0[3] / 2
		points_aligned_to_pred = np.matmul(points - self.x0[0:2], self.rotmat)
		clip_idx = points_aligned_to_pred[:, 0] >= -l2
		clip_idx = np.logical_and(clip_idx, points_aligned_to_pred[:, 0] < l2)
		clip_idx = np.logical_and(clip_idx, points_aligned_to_pred[:, 1] >= -w2)
		clip_idx = np.logical_and(clip_idx, points_aligned_to_pred[:, 1] < w2)
		# calculate JIoU with discrete sample
		n_in = np.sum(clip_idx)
		if n_in > 0:
			probs[clip_idx] = 1/n_in
		return probs

class uncertain_prediction_BEV(uncertain_prediction_BEVdelta):
# The uncertain prediction bounding box from Di Feng's model, not the same as WZN's label uncertainty model
# std = [xc1, xc2, log(l), log(w), 0, 0]
# approximate std(l) = exp(std(log(l)))-1
	def __init__(self, boxBEV, std):
		uncertain_prediction_BEVdelta.__init__(self,boxBEV)
		self.std0 = std
		self.feature0 = np.array([self.x0[0], self.x0[1], self.x0[2], self.x0[3]]) # dim 4 #self.x0[2]*np.exp(0.5*std[2]**2), self.x0[3]*np.exp(0.5*std[2]**2)]
		#print(std[2],np.sqrt(np.exp(std[2]**2)*(np.exp(std[2]**2)-1)),std[3],np.sqrt(np.exp(std[3]**2)*(np.exp(std[3]**2)-1)))
		self.cov0_feature = np.diag(np.array([std[0]**2, std[1]**2, (self.x0[2]*(np.exp(std[2])-1))**2, (self.x0[3]*(np.exp(std[3])-1))**2]))#WZN: the wrong one until uncertaintyV3 is np.exp(std[3])-1] # np.exp(std[2]**2)*(np.exp(std[2]**2)-1)*self.x0[2]**2, np.exp(std[3]**2)*(np.exp(std[3]**2)-1)*self.x0[3]**2
		self.max_sample_grid = max(0.01,min(std[0],std[1])/2)
		#print("This is an uncertainty prediction with max_sample_grid={}".format(self.max_sample_grid))

	def Jacobian_z_normed(self, z_normed):
		# parameterize surface point w.r.t. BEV box feature vector
		# the output vector is (xc1+cos(ry)*l-sin(ry)*w, xc2+sin(ry)*l+cos(ry)*w), 1*2
		nz = z_normed.shape[0]
		Jacobian = np.zeros((nz, 2, 4))
		Jacobian[:, 0, 0] = 1.0
		Jacobian[:, 1, 1] = 1.0
		Jacobian[:, 0, 2] = z_normed[:, 0]*self.cry
		Jacobian[:, 0, 3] = -z_normed[:, 1]*self.sry
		Jacobian[:, 1, 2] = z_normed[:, 0]*self.sry
		Jacobian[:, 1, 3] = z_normed[:, 1]*self.cry
		return Jacobian

	def calc_uncertainty_box(self, std=1):
		#by default, calculate the contour of 1 standard deviation by variance along the boundary of bounding box
		z_normed = np.array([[0.5,0],[0,0.5],[-0.5,0],[0,-0.5]])
		centers, covs = self.calc_uncertainty_points(z_normed)
		#get variance along the direction to center
		direction = centers - self.x0[0:2]
		lengths = np.linalg.norm(direction, axis=1, keepdims=True)
		direction = direction / lengths
		#get a n*1 distance array
		dists2 = np.sum(np.sum(covs*np.expand_dims(direction,axis=2),axis=1)*direction, axis=1).reshape([-1,1])
		dists = np.sqrt(dists2)
		ratio = 1+dists*std/lengths
		outer_corners = self.x0[0:2] + direction[[0,0,2,2],:]*lengths[[0,0,2,2]]*ratio[[0,0,2,2]] + direction[[3,1,1,3],:]*lengths[[3,1,1,3]]*ratio[[3,1,1,3]]
		inner_corners = self.x0[0:2] + direction[[0,0,2,2],:]*lengths[[0,0,2,2]]/ratio[[0,0,2,2]] + direction[[3,1,1,3],:]*lengths[[3,1,1,3]]/ratio[[3,1,1,3]]
		return inner_corners, outer_corners

	def calc_uncertainty_points(self, z_normed):
		nz = z_normed.shape[0]
		cov = self.cov0_feature
		Jacobians_point = self.Jacobian_z_normed(z_normed)
		covs_out = np.zeros((z_normed.shape[0], 2, 2))
		centers_out = np.matmul(z_normed * self.x0[2:4], self.rotmat.transpose()) + self.x0[0:2]
		for i in range(nz):
			Jacobian_point = np.squeeze(Jacobians_point[i, :, :])
			covs_out[i, :, :] = (Jacobian_point @ cov) @ Jacobian_point.transpose()
		return centers_out, covs_out

	def sample_prob(self, points, sample_grid=0.1):
		#WZN: I think for this model we can have explicit theoretical solution
		# sample probablity given points
		sample_grid = min(sample_grid, self.max_sample_grid)
		nk = points.shape[0]
		cov = self.cov0_feature
		if np.max(np.sqrt(np.linalg.eig(cov)[0]))<sample_grid*5:
			#empirical coeffienct, if cov too low, then sample_grid is not thick enough. Just assume the distribution is uniform (delta uncertainty)
			probs = self.sample_prob_delta(points)
		else:
			probs = np.zeros(nk)
			z_normed = create_BEV_box_sample_grid(sample_grid)
			centers_out, covs_out = self.calc_uncertainty_points(z_normed)
			for i in range(covs_out.shape[0]):
				tmp_probs = multivariate_normal.pdf(points, mean=centers_out[i, :], cov=covs_out[i, :, :])
				# exponents = np.sum((points-centers_out[i,:])*((points-centers_out[i,:])@ np.linalg.inv(covs_out[i,:,:])),axis=1)/-2
				# tmp_probs = np.exp(exponents)
				probs += tmp_probs
			probs /= np.sum(probs)
		return probs

class uncertain_prediction_BEV_interp(uncertain_prediction_BEVdelta):
	# The uncertain prediction bounding box with sampled corners (from Hujie's paper) as input.
	def __init__(self, boxBEV, Ws, Vs):
		'''
		Construct the box with corner uncertainty inputs.

		Args:
			Ws: The homogeneous coordinates of sampled points. (4 or 6 points)
			Vs: The covariance matrix of sampled points.
		'''
		uncertain_prediction_BEVdelta.__init__(self, boxBEV)
		self.num_points = len(Ws)
		assert Vs.shape[1] == 2, 'Varaicne matrix error.'
		if self.num_points == 6:
			self.cov_interpolation = cov_interp_BEV_full(Ws, Vs)
		elif self.num_points == 4: 
			self.cov_interpolation = cov_interp_BEV_corner(Ws, Vs)
		else: 
			raise ValueError('the number of sampled points can only be 4 or 6 for BEV while %d is given.' % self.num_points)

	def calc_uncertainty_box(self, std=1):
		return uncertain_prediction_BEV.calc_uncertainty_box(self, std=std)

	def calc_uncertainty_points(self, z_normed):
		ws = np.concatenate((z_normed, np.ones([z_normed.shape[0], 1])), axis=1)
		covs_out = self.cov_interpolation.interps(ws)
		centers_out = np.matmul(z_normed * self.x0[2:4], self.rotmat.transpose()) + self.x0[0:2]
		return centers_out, covs_out

	def sample_prob(self, points, sample_grid=0.1):
		nk = points.shape[0]
		probs = np.zeros(nk)
		z_normed = create_BEV_box_sample_grid(sample_grid)
		centers_out, covs_out = self.calc_uncertainty_points(z_normed)	

		# This is to avoid negative variance due to interpolation.
		w, v = np.linalg.eig(covs_out)
		if np.sum(w < 0) > 0:
			print('Warning: Negative variance detected at a sample')
			w[w < 0.02**2] = 0.02**2
			covs_out = np.matmul(np.transpose(v, axes=(0, 2, 1)), np.expand_dims(w, axis=-2) * v)

		for i in range(covs_out.shape[0]):
			tmp_probs = multivariate_normal.pdf(points, mean=centers_out[i, :], cov=covs_out[i, :, :])
			probs += tmp_probs
		probs /= np.sum(probs)
		return probs

