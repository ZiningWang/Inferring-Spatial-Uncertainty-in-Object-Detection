import os
import sys
import numpy as np
import math
import shapely.geometry
import shapely.affinity

from shapely.geometry import Point as sgPoint
from shapely.geometry.polygon import Polygon as sgPolygon
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Arrow, Ellipse
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, norm
from scipy.spatial import ConvexHull
from tqdm import tqdm

#import pycuda.driver as drv
#import pycuda.tools
#import pycuda.autoinit
#from pycuda.compiler import SourceModule

sys.path.append('../')
#WZN: important, the default sequence of corner points
z_normed_corners_default = np.array([[1,-1],[1,1],[-1,1],[-1,-1]])/2


def cpu_voxel(points, 
              q_lvl, 
              size):
    points_copy = np.copy(points)
    points_copy = np.floor(points_copy/q_lvl)
    points_copy = points_copy.astype(int)

    voxel = np.zeros((size[0],size[1],size[2]), dtype=np.float32)
    for i in points_copy:
        voxel[i[0], i[1], i[2]] = 1.0 

    return voxel

def cpu_voxel_WZN(points, 
              q_lvl, 
              size):
    points_copy = np.copy(points)
    points_copy = np.floor(points_copy/q_lvl)
    points_copy = points_copy.astype(int)

    voxel = np.zeros((size[0],size[1],size[2]), dtype=np.float32)
    voxel[points_copy[:,0],points_copy[:,1],points_copy[:,2]] = 1.0   

    return voxel
    

def match_gt_label(bbox, gt_bboxes):
    # print("shape: {} {}".format(np.shape([bbox]), np.shape(gt_bboxes)))
    iou = bbox2d_iou([bbox], gt_bboxes)
    # print("iou: {}".format(iou))
    max_iou = np.argmax(iou)
    iou = np.asarray(iou)
    iou = iou[0, max_iou]
    # print("max_iou: {} {:.2f}".format(max_iou, iou))

    return max_iou, iou

def get_closest_locs(loc_a, loc_b):
    if len(loc_a) == 0:
        return []

    dists = []
    for loc in loc_a:
        x1 = loc[0]
        x2 = loc_b[0]
        y1 = loc[1]
        y2 = loc_b[1]
        z1 = loc[2]
        z2 = loc_b[2]

        # dist_xy = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        # dist_z = 
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        dists.append(dist)
    # print("dists: {}".format(dists))
    min_dist_idx = np.argmin(dists)

    x1 = loc_a[min_dist_idx][0]
    x2 = loc_b[0]

    y1 = loc_a[min_dist_idx][1]
    y2 = loc_b[1]

    z1 = loc_a[min_dist_idx][2]
    z2 = loc_b[2]

    min_dist_xy = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    min_dist_z = np.abs(z1-z2)
    # print("min idx: {} {:.2f}".format(min_dist_idx, min_dist))

    return loc_a[min_dist_idx], min_dist_xy, min_dist_z




def readCalibration(calib_dir, img_idx, cam):
    # load 3x4 projection matrix
    cam = 2
    with open(os.path.join(calib_dir, '%06d.txt' % img_idx)) as f:
        P = f.readlines()
    P = P[cam].strip().split()
    P = [float(value) for value in P[1:]]
    P = np.reshape(np.array(P), (3, 4))
    return P


def read_objects(label_path, only_car=True):
    objects = []
    with open(label_path, 'r') as fp:
        for line in fp:
            read = line.split()
            if len(read) > 2:
                obj = edict({})
                obj.lbl_txt = read[0]
                obj.truncated = float(read[1])
                obj.bbox = np.array([float(v) for v in read[4:8]])
                obj.height = float(read[8])
                obj.width = float(read[9])
                obj.length = float(read[10])
                obj.x = float(read[11])
                obj.y = float(read[12])
                obj.z = float(read[13])
                obj.R = float(read[14])
                obj.score = float(read[15]) if len(read) > 15 else 1.0
                objects.append(obj)
    if only_car:
        objects = [obj for obj in objects if obj.lbl_txt in ['Car', 'Van', 'car', 'van']]

    return objects

def draw_birdeye_arrow(arrow, ax, **kwargs):
    x = arrow.x
    y = arrow.y
    dx = arrow.dx
    dy = arrow.dy
    width = arrow.width


    o = Arrow(x, y, dx, dy, width, **kwargs)
    ax.add_patch(o)

    return o

def draw_cylinder(cyl, ax, **kwargs):
    x = cyl.x
    y = cyl.y
    z = cyl.z
    radii = cyl.radii
    height = cyl.height

    X = np.linspace(x-radii, x+radii, 100)
    Z = np.linspace(z, z+height, 2)
    Xc, Zc = np.meshgrid(X, Z)
    Yc1 = y + np.sqrt(radii**2 - (Xc-x)**2)
    Yc2 = y - np.sqrt(radii**2 - (Xc-x)**2)

    ax.plot_surface(Xc, Yc1, Zc, **kwargs)
    ax.plot_surface(Xc, Yc2, Zc, **kwargs)
    return

def draw_birdeye_circle(loc, ax, **kwargs):
    xy = (loc.x, loc.y)
    radius = loc.radii

    o = Circle(xy, radius, **kwargs)
    ax.add_patch(o)
    # 
    # ax.plot(corners[[2, 3], 0], corners[[2, 3], 1], color='C1', **kwargs_)
    return o

def draw_birdeye(obj, ax, **kwargs):

    globalR = obj.R
    globalR = -globalR # for camera coordinate
    globalR = np.array([[math.cos(globalR), -math.sin(globalR)],
                        [math.sin(globalR), math.cos(globalR)]])
    corners = np.array([[-obj.length, -obj.width],
                        [-obj.length, obj.width],
                        [obj.length, obj.width],
                        [obj.length, -obj.width]]) / 2.0
    corners = np.transpose(corners)
    corners = np.dot(globalR, corners)
    corners = np.transpose(corners) + np.array([obj.x, obj.y])
    # corners = np.array([obj.x, obj.y]) + corners
    o = Polygon(corners, True, **kwargs)
    ax.add_patch(o)
    # kwargs_ = {'lw': 3, 'alpha': 0.5}
    # ax.plot(corners[[2, 3], 0], corners[[2, 3], 1], color='C1', **kwargs_)
    return o


def draw_birdeye_ellipse(ax, cov, centre, nstd=1, **kwargs):
    alpha = 0.2
    if 'alpha' in kwargs:
        alpha = min(kwargs['alpha'],alpha)
    e = get_cov_ellipse(cov,centre, nstd, fc='red', alpha = alpha)
    ax.add_patch(e)



def get_cov_ellipse(cov, centre, nstd, **kwargs):
    #WZN
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)


def projectToImage(pts_3D, P):
    """
    projects 3D points in the World coordinate system on to 2D image plane
    using the given projection matrix P.
    """
    pts_3D = np.reshape(pts_3D, (-1, 3))
    pts_3D = np.transpose(pts_3D)
    pts_3D = np.vstack([pts_3D, 1])
    pts_2D = np.matmul(P, pts_3D)
    pts_2D = pts_2D[:2]/pts_2D[-1]
    pts_2D = np.transpose(pts_2D)
    return pts_2D

def projectToImage_kitti(pts_3D, P):
    """
    PROJECTTOIMAGE projects 3D points in given coordinate system in the image
    plane using the given projection matrix P.

    Usage: pts_2D = projectToImage(pts_3D, P)
    input: pts_3D: 3xn matrix
          P:      3x4 projection matrix
    output: pts_2D: 2xn matrix

    last edited on: 2012-02-27
    Philip Lenz - lenz@kit.edu
    """
    # project in image
    mat = np.vstack((pts_3D, np.ones((pts_3D.shape[1]))))

    pts_2D = np.dot(P, mat)

    # scale projected points
    pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
    pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]
    pts_2D = np.delete(pts_2D, 2, 0)

    return pts_2D


def project_pts3_to_image(pts3, P):
    """
    projects 3D points in the World coordinate system on to 2D image plane
    using the given projection matrix P.
    """
    assert (len(pts3.shape) == 2)
    assert (pts3.shape[1] == 3)
    assert (P.shape == (3, 4))
    pts3 = np.hstack([pts3, np.ones((len(pts3), 1))])  # homogeneous
    pts2 = P.dot(pts3.T).T
    pts2 = pts2[:, :2] / (pts2[:, [-1]] + 1e-8)
    return pts2


def computeOrientation3D(object, P):
    """
    takes an OBJECT and a projection matrix P and projects the 3D
    object orientation vector into the image plane.
    """

    # compute rotational matrix around yaw axis
    R = [[np.cos(object.ry),  0, np.sin(object.ry)],
         [0,               1,              0],
         [-np.sin(object.ry), 0, np.cos(object.ry)]]

    # orientation in object coordinate system
    orientation_3D = [[0.0, object.l],
                      [0.0, 0.0],
                      [0.0, 0.0]]

    # rotate and translate in camera coordinate system, project in image
    orientation_3D      = R * orientation_3D
    orientation_3D[0, :] += object.t[0]
    orientation_3D[1, :] += object.t[1]
    orientation_3D[2, :] += object.t[2]

    # vector behind image plane?
    if any(orientation_3D[2, :] < 0.1):
        orientation_2D = []
    else:
        # project orientation into the image plane
        orientation_2D = projectToImage(orientation_3D, P)
    return orientation_2D

def rad2deg(rad):
    return rad * 180. / np.pi

def deg2rad(deg):
    return deg / 180. * np.pi

# ------ traffic light detection

import scipy.ndimage.morphology

# ------------- L4 CAMERA ASSUMPTION
img_width = 1280
img_height = 720
f = 0.9464038706620487
K = np.array([[f * img_width, 0., img_width / 2, 0],
             [0., f * img_width, img_height / 2+30, 0],
             [0., 0., 1., 0]])

def get_interest_map(far):
    """
    Based on traffic light distance (Fin meters),
    this function returns probability map of possible locations.
    """

    # --- horizontal locations on 5 meter high in world coordinate
    height = -3.5
    x = np.arange(-4, 12, 1)
    x = x.reshape((-1, 1))
    high_horizon = np.concatenate([x, np.ones_like(x) * height, np.ones_like(x) * far], 1)

    # --- {3, 7, 11} meters right and 2.5 meter high in world coordinate
    height = -1.
    x = np.arange(3, 12, 4)
    x = x.reshape((-1, 1))
    right_candidate = np.concatenate([x, np.ones_like(x) * height, np.ones_like(x) * far], 1)

    p_world = np.concatenate([high_horizon, right_candidate], 0)
    p_img = project_pts3_to_image(p_world, K)

    # --- if close, search for top region in image coordinate
    if far < 8:
        x = np.arange(600, 1280, 50)
        x = x.reshape((-1, 1))
        y = 5
        close = np.concatenate([x, np.ones_like(x) * y], 1)
        p_img = np.concatenate([p_img, close], 0)

    # --- consider only locations in image
    ll = np.array([0, 0])           # lower-left
    ur = np.array([img_width, img_height])      # upper-right
    inidx = np.all(np.logical_and(ll <= p_img, p_img <= ur), axis=1)
    inbox = p_img[inidx]
    inbox = inbox.astype(np.int)

    interest = np.zeros((img_height, img_width))
    interest[inbox[:, 1], inbox[:, 0]] = 1
    interest = scipy.ndimage.morphology.distance_transform_edt(interest-1)
    interest = np.exp(-interest / 30**2)
    interest = (interest - np.min(interest)) / (np.max(interest) - np.min(interest))
    return interest

def rotation2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, -s], [s, c]])
    return R

def bbox_iou(box1, box2, x1y1x2y2=True):
    # x1y1x2y2: True for when box values are given as min_x, min_y, max_x, max_y
    # False for when box values are given as center_x, center_y, width, height
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def bbox2d_iou(boxes2d_a, boxes2d_b, x1y1x2y2=True):
    N1 = len(boxes2d_a)
    N2 = len(boxes2d_b)
    if N2 == 0:
        output = np.zeros((N1, 1), dtype=np.float32)
    else:
        output = np.zeros((N1, N2), dtype=np.float32)
        for idx in range(N1):
            for idy in range(N2):
                output[idx, idy] = bbox_iou(boxes2d_a[idx], boxes2d_b[idy], x1y1x2y2)

    return output


def cal_iou2d(box1, box2):
    # Input:
    #   box1/2: x, y, w, l, r
    # Output:
    #   iou
    x1, y1, w1, l1, r1 = box1
    x2, y2, w2, l2, r2 = box2
    c1 = shapely.geometry.box(-l1 / 2.0, -w1 / 2.0, l1 / 2.0, w1 / 2.0)
    c2 = shapely.geometry.box(-l2 / 2.0, -w2 / 2.0, l2 / 2.0, w2 / 2.0)

    c1 = shapely.affinity.rotate(c1, -r1, use_radians=True)
    c2 = shapely.affinity.rotate(c2, -r2, use_radians=True)

    c1 = shapely.affinity.translate(c1, -y1, x1)
    c2 = shapely.affinity.translate(c2, -y2, x2)

    intersect = c1.intersection(c2)

    return intersect.area / (c1.area + c2.area - intersect.area)

def cal_iou_bev(box1, box2):
    # box coordinate Lidar
    # cython coordinate camera
    # Input:
    #   box1/2: x, y, w, l, r
    # Output:
    #   iou

    # essentially same as cal_iou2d but use cython

    x1, y1, w1, l1, r1 = box1
    x2, y2, w2, l2, r2 = box2

    obj_box1 = edict({})
    obj_box2 = edict({})

    obj_box1.x = -y1
    obj_box1.y = 0.0
    obj_box1.z = x1
    obj_box1.height = 0.0
    obj_box1.width = w1
    obj_box1.length = l1
    obj_box1.R = r1 

    obj_box2.x = -y2
    obj_box2.y = 0.0
    obj_box2.z = x2
    obj_box2.height = 0.0 
    obj_box2.width = w2
    obj_box2.length = l2
    obj_box2.R = r2

    iou_bev = cy_2d_iou.cy_2d_iou(obj_box1, obj_box2, -1)

    return iou_bev

def cal_box2d_iou(boxes2d_a, boxes2d_b):
    # Inputs:
    #   boxes2d_a: (N1, 5) x,y,w,l,r
    #   boxes2d_b: (N2, 5) x,y,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    # print("N1, N2: {} {}".format(np.shape(boxes2d_a), np.shape(boxes2d_b)))
    if len(np.shape(boxes2d_b)) == 3:
        boxes2d_b = np.squeeze(boxes2d_b, 1)
    N1 = len(boxes2d_a)
    N2 = len(boxes2d_b)
    output = np.zeros((N1, N2), dtype=np.float32)
    for idx in range(N1):
        for idy in range(N2):
            output[idx, idy] = cal_iou2d(boxes2d_a[idx], boxes2d_b[idy])

    return output

def cal_z_intersect(cz1, h1, cz2, h2):
    b1z1, b1z2 = cz1, cz1 + h1
    b2z1, b2z2 = cz2, cz2 + h2
    if b1z1 > b2z2 or b2z1 > b1z2:
        return 0
    elif b2z1 <= b1z1 <= b2z2:
        if b1z2 <= b2z2:
            return h1
        else:
            return (b2z2 - b1z1)
    elif b1z1 < b2z1 < b1z2:
        if b2z2 <= b1z2:
            return h2
        else:
            return (b1z2 - b2z1)

def cal_iou3d(box1, box2):
    # Input:
    #   box1/2: x, y, z, h, w, l, r
    # Output:
    #   iou

    # Assume box1 and box2 are in Lidar coordinate.
    # Cython function was implemented in Camera coordinate.
    # Lidar x,y,z => Camera z,

    x1, y1, z1, h1, w1, l1, r1 = box1
    x2, y2, z2, h2, w2, l2, r2 = box2

    obj_box1 = edict({})
    obj_box2 = edict({})

    obj_box1.x = -y1
    obj_box1.y = -z1
    obj_box1.z = x1
    obj_box1.height = h1
    obj_box1.width = w1
    obj_box1.length = l1
    obj_box1.R = r1 

    obj_box2.x = -y2
    obj_box2.y = -z2
    obj_box2.z = x2
    obj_box2.height = h2 
    obj_box2.width = w2
    obj_box2.length = l2
    obj_box2.R = r2

    iou_3d = cy_3d_iou.cy_3d_iou(obj_box1, obj_box2, -1)

    return iou_3d

def cal_box2d_iou_match(boxes2d_a, boxes2d_b):
    N1 = len(boxes2d_a)
    output = np.zeros((N1,1), dtype=np.float32)
    for idx in range(N1):
        output[idx] = float(bbox_iou(boxes2d_a[idx], boxes2d_b[idx], True))
    return output

def cal_box_bev_iou(boxes_bev_a, boxes_bev_b):
    # x, y, w, l ,r
    N1 = len(boxes_bev_a)
    output = np.zeros((N1,1), dtype=np.float32)
    for idx in range(N1):
        output[idx] = float(cal_iou_bev(boxes_bev_a[idx], boxes_bev_b[idx]))
    return output

def cal_box3d_iou_match(boxes3d_a, boxes3d_b):
    N1 = len(boxes3d_a)
    output = np.zeros((N1,1), dtype=np.float32)
    for idx in range(N1):
        output[idx] = float(cal_iou3d(boxes3d_a[idx], boxes3d_b[idx]))
    return output


def cal_box3d_iou(boxes3d_a, boxes3d_b, cal_3d=0):
    # Inputs:
    #   boxes3d: (N1, 7) x,y,z,h,w,l,r
    #   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    N1 = len(boxes3d_a)
    N2 = len(boxes3d_b)
    output = np.zeros((N1, N2), dtype=np.float32)

    # print("shape: {} {}".format(np.shape(boxes3d_a), np.shape(boxes3d_b)))

    for idx in range(N1):
        for idy in range(N2):
            if cal_3d:
                output[idx, idy] = float(
                    cal_iou3d(boxes3d_a[idx], boxes3d_b[idy]))
            else:
                output[idx, idy] = float(
                    cal_iou2d(boxes3d_a[idx, [0, 1, 4, 5, 6]], boxes3d_b[idy, [0, 1, 4, 5, 6]]))

    return output

def rotation_nms_3d(boxes_center, bbox_score, thresh = 0.75):
    # for additional filtering
    # input:

    # 3d box: x y z h w l r
    # 2d box: h l y x r
    #   bbox2d center coordiante: (N, 5) x, y, w, l, r
    #   bbox_score: (N,1) 0~1 float according to boxes above

    # print("boxes_center: {}".format(boxes_center))
    # print("score: {}".format(bbox_score))
    bbox3d = boxes_center.copy()
    # bbox2d = bbox3d[:, [0, 1, 4, 5, 6]]
    # bbox2d[:, 4] = -bbox2d[:, 4] # rotation upside-down

    bbox_score = np.squeeze(bbox_score)

    order = bbox_score.argsort()[::-1]

    # print("order: {}".format(order))
    keep_idx = []

    while order.size > 0:
        i = order[0]
        keep_idx.append(i) # keep most confident bbox regardless of any other bbox
        iou = cal_box3d_iou(bbox3d[[i]], bbox3d[order[1:]])[0]
        inds = np.where(iou <= thresh)[0]
        order = order[inds+1]
    return keep_idx


def rotation_nms(boxes_center, bbox_score, thresh = 0.75, bev_box = False):
    # for additional filtering
    # input:

    # 3d box: x y z h w l r
    # 2d box: h l y x r
    #   bbox2d center coordiante: (N, 5) x, y, w, l, r
    #   bbox_score: (N,1) 0~1 float according to boxes above

    # if bev_box == False: 3d box as input
    # bev_box true: then bev box as input: x, y, w, l, r

    if bev_box is False:
        bbox3d = boxes_center.copy()
        bbox2d = bbox3d[:, [0, 1, 4, 5, 6]]
    else:
        bbox2d = boxes_center
    # bbox2d[:, 4] = -bbox2d[:, 4] # rotation upside-down

    bbox_score = np.squeeze(bbox_score)

    order = bbox_score.argsort()[::-1]

    # print("order: {}".format(order))
    keep_idx = []

    while order.size > 0:
        i = order[0]
        keep_idx.append(i) # keep most confident bbox regardless of any other bbox
        iou = cal_box2d_iou(bbox2d[[i]], bbox2d[order[1:]])[0]
        inds = np.where(iou <= thresh)[0]
        order = order[inds+1]
    return keep_idx

def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle

def camera_to_lidar(x, y, z, calib_mat):
    R0 = calib_mat['R0']
    Tr_velo_to_cam = calib_mat['Tr_velo2cam']

    p = np.array([x, y, z, np.ones_like(x)])
    p = np.matmul(np.linalg.inv(np.array(R0)), p)
    p = np.matmul(np.linalg.inv(np.array(Tr_velo_to_cam)), p)
    p = p[0:3]
    return tuple(p)

def camera_to_lidar_box(boxes, calib_mat=None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z, calib_mat), h, w, l, -ry - np.pi / 2
        rz = angle_in_limit(rz)
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 7)

def lidar_to_camera(x, y, z, calib_mat=None):
    if calib_mat is not None:
        R0 = calib_mat['R0']
        Tr_velo_to_cam = calib_mat['Tr_velo2cam']
    else:
        R0 = cfg.MATRIX_R_RECT_0
        Tr_velo_to_cam = cfg.MATRIX_T_VELO_2_CAM

    p = np.array([x, y, z, np.ones_like(x)])
    p = np.matmul(np.array(Tr_velo_to_cam), p)
    p = np.matmul(np.array(R0), p)
    p = p[0:3]
    
    return tuple(p)

def lidar_to_camera_box(boxes, calib_mat=None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, ry = lidar_to_camera(
            x, y, z, calib_mat=calib_mat), h, w, l, ry
        # ry = angle_in_limit(ry)
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret).reshape(-1, 7)

# this just for visulize and testing
def lidar_box3d_to_camera_box(boxes3d, calib_mat=None, cal_projection=False):
    # (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = len(boxes3d)
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)
    lidar_boxes3d_corner = center_to_corner_box3d(boxes3d, coordinate='lidar', calib_mat=calib_mat)
    if calib_mat is not None:
        P2 = calib_mat['P2']
    else:
        P2 = np.array(cfg.MATRIX_P2)

    for n in range(num):
        box3d = lidar_boxes3d_corner[n]
        box3d = lidar_to_camera_point(box3d, calib_mat=calib_mat)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        points = np.matmul(P2, points).T
        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]

        minx = int(np.clip(np.min(points[:, 0]), 0, None))
        maxx = int(np.clip(np.max(points[:, 0]), None, cfg.IMAGE_WIDTH))
        miny = int(np.clip(np.min(points[:, 1]), 0, None))
        maxy = int(np.clip(np.max(points[:, 1]), None, cfg.IMAGE_HEIGHT))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d

def box3d_to_label(batch_box3d, batch_cls, batch_bbox_2d, batch_score=[], include_score = False, coordinate='camera', batch_trunc=None, batch_diff=None, batch_alpha=None):
    # Input:
    #   (N, N', 7) x y z h w l r
    #   (N, N')
    #   cls: (N, N') 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate(input): 'camera' or 'lidar'
    # Output:
    #   label: (N, N') N batches and N lines
    # warn("to label")
    batch_label = []
    if include_score:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(15)]) + '\n'
        # warn("len calib {}".format(len(batch_calib)))
        for boxes2d, boxes3d, scores, clses in zip(batch_bbox_2d, batch_box3d, batch_score, batch_cls):
            label = []
            # warn("calib: {}".format(calib_mat))
            for box2d, box3d, score, cls in zip(boxes2d, boxes3d, scores, clses):
                # box3d = box
                # box2d = bbox_2d
                x, y, z, h, w, l, r = box3d
                x_min, y_min, x_max, y_max = box2d
                box3d = [h, w, l, x, y, z, r]
                box2d = [x_min,y_min,x_max,y_max]

                label.append(template.format(
                    cls, 0, 0, -10, *box2d, *box3d, float(score)))
            batch_label.append(label)
    else:
        template = '{} ' +  '{:.2f} ' + '{:d} '  + ' '.join(['{:.2f}' for i in range(12)]) + '\n'
        # warn("len calib {}".format(len(batch_calib)))
        for boxes2d, boxes3d, clses, truncs, diffs, alphas in zip(batch_bbox_2d, batch_box3d, batch_cls, batch_trunc, batch_diff, batch_alpha):
            label = []
            # warn("calib: {}".format(calib_mat))
            for box2d, box3d, cls, trunc, diff, alpha in zip(boxes2d, boxes3d, clses, truncs, diffs, alphas):
                # box3d = box
                # box2d = bbox_2d
                x, y, z, h, w, l, r = box3d
                x_min, y_min, x_max, y_max = box2d
                box3d = [h, w, l, x, y, z, r]
                box2d = [x_min,y_min,x_max,y_max]

                label.append(template.format(
                    cls, trunc, diff, alpha, *box2d, *box3d))
            batch_label.append(label)        

    return batch_label
def center_to_corner_box3d(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2], \
            [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        yaw = -yaw # => chagne to camera coordinate
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    return ret

def center_to_corner_box2d(boxes_center):
    # (N, 5) -> (N, 4, 2)
    # x, y, w, l, r
    # Attention:
    # coordinate between box2d and box3d function are different.
    # This works in original cartesian cooridnate.
    # For 3d, it works in lidar coordinate.

    # N = boxes_center.shape[0]
    ret = np.zeros((4, 2), dtype=np.float32)

    # for i in range(N):
    box = boxes_center
    x, y, w, l, r = box
    # print("box: {}".format(box))
    translation = np.array([[-y, x]])
    # size = box[2:4]
    rotation = r

    trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
        [-l / 2, -l / 2, l / 2, l / 2], \
        [w / 2, -w / 2, w / 2, -w / 2]])

    # re-create 3D bounding box in velodyne coordinate system
    yaw = rotation
    yaw = -yaw # => chagne to camera coordinate
    rotMat = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]])

    cornerPosInVelo = np.dot(rotMat, trackletBox) + \
        np.tile(translation, (4, 1)).T
    box3d = cornerPosInVelo.transpose()
    ret = box3d

    return ret

def center_to_corner_BEV(boxBEV):
    #WZN boxBEV to corners in BEV
    #boxBEV is [xcam, zcam, l, w, ry]
    x0 = np.array(boxBEV).reshape(5)
    x0[4] *= -1
    ry = x0[4]
    rotmat = np.array([[math.cos(ry), -math.sin(ry)], [math.sin(ry), math.cos(ry)]])
    corners = np.matmul(z_normed_corners_default * x0[2:4], rotmat.transpose()) + x0[0:2]
    return corners

def align_BEV_to_corner(boxBEV, corner_idx, new_size):
    #WZN boxBEV to new size, but keep one corner unchanged
    new_lw = np.array(new_size).reshape(2)
    x0 = np.array(boxBEV).reshape(5)
    x0[4] *= -1
    ry = x0[4]
    rotmat = np.array([[math.cos(ry), -math.sin(ry)], [math.sin(ry), math.cos(ry)]])
    new_center = x0[0:2] + np.matmul(z_normed_corners_default[corner_idx,:] * (x0[2:4]-new_lw), rotmat.transpose())
    new_boxBEV = boxBEV.copy()
    new_boxBEV[0] = new_center[0]
    new_boxBEV[1] = new_center[1]
    new_boxBEV[2] = new_lw[0]
    new_boxBEV[3] = new_lw[1]
    new_boxBEV[4] = boxBEV[4]
    return new_boxBEV


def lidar_to_bird_view(x, y, x_range, y_range, factor=1):
    x_min, x_max, x_size = x_range
    y_min, y_max, y_size = y_range
    a = (x - x_min) / x_size * factor
    b = (y - y_min) / y_size * factor
    a = np.clip(a, a_max=(x_max - x_min) / x_size * factor, a_min=0)
    b = np.clip(b, a_max=(y_max - y_min) / y_size * factor, a_min=0)
    return a, b

# def readCalibration(calib_dir, img_idx, cam):
#     # load 3x4 projection matrix
#     cam = 2
#     with open(os.path.join(calib_dir, '%06d.txt' % img_idx)) as f:
#         P = f.readlines()
#     P = P[cam].strip().split()
#     P = [float(value) for value in P[1:]]
#     P = np.reshape(np.array(P), (3, 4))
#     return P

def clip_by_boundary(points, x_min = 0.0, x_max = 70.4, y_min = -40.0, y_max = 40.0, z_min = -3.5, z_max = 3.5):
    bound_x = np.logical_and(points[:, 0] >= x_min, points[:, 0] <= x_max)
    bound_y = np.logical_and(points[:, 1] >= y_min, points[:, 1] <= y_max)
    bound_z = np.logical_and(points[:, 2] >= z_min, points[:, 2] <= z_max)

    bound = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    return points[bound]

def clip_by_projection(points, P, y2, x2, y1=0, x1=0):
    pts_2D = projectToImage_kitti(points[:,0:3].transpose(), P)
    pts_2D = pts_2D.transpose()
    clipped_idx = (pts_2D[:, 0] <= x2+500) & (pts_2D[:, 0] >= x1-500) & (pts_2D[:, 1] <= y2+150) & (pts_2D[:, 1] >= y1-150)
    return points[clipped_idx]

def load_P(calib_dir, img_idx):
    calib_file      = "{}/{:06d}.txt".format(calib_dir, img_idx)
    raw_calib       = load_kitti_calib(calib_file)
    calib_matrix    = calib_gathered(raw_calib)
    P               = calib_to_P(calib_matrix)
    return P

def load_kitti_calib(velo_calib_path):
    with open(velo_calib_path) as fi:
        lines = fi.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2' : P2.reshape(3,4),
            'P3' : P3.reshape(3,4),
            'R0' : R0.reshape(3,3),
            'Tr_velo2cam' : Tr_velo_to_cam.reshape(3, 4)}

def calib_gathered(raw_calib):
    calib = np.zeros((4, 12))
    calib[0, :] = raw_calib['P2'].reshape(12)
    calib[1, :] = raw_calib['P3'].reshape(12)
    calib[2, :9] = raw_calib['R0'].reshape(9)
    calib[3, :] = raw_calib['Tr_velo2cam'].reshape(12)

    return calib



def calib_to_P(calib):
    #WZN: get the actual overall calibration matrix from Lidar coord to image
    #calib is 4*12 read by imdb
    #P is 3*4 s.t. uvw=P*XYZ1
    C2V = np.vstack((np.reshape(calib[3,:],(3,4)),np.array([0,0,0,1])))
    R0 = np.hstack((np.reshape(calib[2,:],(4,3)),np.array([[0],[0],[0],[1]])))
    P2 = np.reshape(calib[0,:],(3,4))
    P = np.matmul(np.matmul(P2,R0),C2V)
    return P

def random_dropout_points(points, keep_ratio = 0.8, min_points = 20):
    num_points = len(points)
    if len(points) < min_points:
        return points

    np.random.shuffle(points)
    num_keep_points = max(min_points, int(keep_ratio*num_points))
    return points[0:num_keep_points]

def random_perturb_points(points, max_pert=0.1):
    N = len(points)
    random_pert = np.random.uniform(-max_pert, max_pert, (N, 3))
    points[:,0:3] = points[:,0:3] + random_pert
    return points

'''
            def random_sample_locs_origin(pert_x, pert_y, pert_z, N, x=0.0,y=0.0,z=0.0):

                Randomly sample location around the origin
                Since ground truth points are located at the origin.

                locs = []
                for _ in range(N):
                    loc_x = x + np.random.uniform(-pert_x, pert_x)
                    loc_y = y + np.random.uniform(-pert_y, pert_y)
                    loc_z = z + np.random.uniform(-pert_z, pert_z)
                    loc = [loc_x, loc_y, loc_z]
                    locs.append(loc)
                return locs
'''

def loc_inverse_projection(h, w, l, R, bbox, P, calib_mat):
    obj = edict({})
    obj.height = h + np.random.uniform(-max_pert_size, max_pert_size)
    obj.width = w + np.random.uniform(-max_pert_size, max_pert_size)
    obj.length = l + np.random.uniform(-max_pert_size, max_pert_size)
    obj.R = R# + np.random.uniform(-max_pert_r, max_pert_r)
    obj.Rl = 0.0
    obj.bbox = bbox.tolist()

    P = np.asarray(calib_mat['P2'][0:3,]).astype(np.float32)

    obj = pyxyz.figure_out_xyz_cy(obj, P)
    # print("obj: {}".format(obj))
    x,y,z = obj.x, obj.y, obj.z

    x_lidar, y_lidar, z_lidar = camera_to_lidar(x, y, z, calib_mat)
    loc_lidar = [x_lidar, y_lidar, z_lidar]

    return loc_lidar

def random_sample_obj_inverse_projection(h, w, l, R, bbox, P, calib_mat, N = 5):
    objs = []
    ratio_min = 0.8
    ratio_max = 1.2    
    obj = edict({})
    ratios = np.linspace(ratio_min, ratio_max, N)
    for ratio in ratios:
        obj.height = h*ratio
        obj.width = w*ratio
        obj.length = l*ratio
        obj.R = normalize_angle(R)

        obj.Rl = 0.0
        obj.bbox = bbox.tolist()

        P = np.asarray(calib_mat['P2'][0:3,]).astype(np.float32)

        obj = pyxyz.figure_out_xyz_cy(obj, P)
        # print("obj: {}".format(obj))
        x,y,z = obj.x, obj.y, obj.z

        x_lidar, y_lidar, z_lidar = camera_to_lidar(x, y, z, calib_mat)
        if x_lidar >= 70.4 or x_lidar <=0 or y_lidar <= -40.0 or y_lidar >= 40.0 or z_lidar >= 3.5 or z_lidar <= -3.5:
            continue
        obj_lidar = [obj, x_lidar, y_lidar, z_lidar]
        objs.append(obj_lidar)
    return objs


def random_sample_inverse_projection(h, w, l, R, bbox, P, calib_mat, N, equal_distance = None):


    obj = edict({})

    locs = []

    s = 0.6

    ratio_min = 1.0-s
    ratio_max = 1.0+s

    if equal_distance is not None:
        ratios = [ratio_min, ratio_max]
        x_lidars = []
        y_lidars = []
        z_lidars = []
        for idx, ratio in enumerate(ratios):            
            obj.height = h*ratio
            obj.width = w*ratio
            obj.length = l*ratio
            obj.R = normalize_angle(R)
            obj.Rl = 0.0
            obj.bbox = bbox.tolist()
            P = np.asarray(calib_mat['P2'][0:3,]).astype(np.float32)
            obj = pyxyz.figure_out_xyz_cy(obj, P)
            x,y,z = obj.x, obj.y, obj.z
            x_lidar, y_lidar, z_lidar = camera_to_lidar(x, y, z, calib_mat)

            x_lidars.append(x_lidar)
            y_lidars.append(y_lidar)
            z_lidars.append(z_lidar)

        dist = np.sqrt((x_lidars[1] - x_lidars[0])**2 + (y_lidars[1] - y_lidars[0])**2 + (z_lidars[1] - z_lidars[0])**2)
        theta = np.arctan((y_lidars[1] - y_lidars[0])/(x_lidars[1]-x_lidars[0]))
        x_unit = (x_lidars[1] - x_lidars[0]) / dist
        y_unit = (y_lidars[1] - y_lidars[0]) / dist
        z_unit = (z_lidars[1] - z_lidars[0]) / dist

        N = int(np.ceil(dist / equal_distance))
        for n_idx in range(N):
            x_lidar = x_lidars[0] + n_idx * x_unit * equal_distance
            y_lidar = y_lidars[0] + n_idx * y_unit * equal_distance
            z_lidar = z_lidars[0] + n_idx * z_unit * equal_distance

            if x_lidar >= 70.4 or x_lidar <=0 or y_lidar <= -40.0 or y_lidar >= 40.0 or z_lidar >= 3.5 or z_lidar <= -3.5:
                continue
            loc_lidar = [x_lidar, y_lidar, z_lidar]
            locs.append(loc_lidar)

    else:
        ratios = np.linspace(ratio_min, ratio_max, N)
        for ratio in ratios:
            obj.height = h*ratio
            obj.width = w*ratio
            obj.length = l*ratio
            obj.R = normalize_angle(R)

            obj.Rl = 0.0
            obj.bbox = bbox.tolist()

            P = np.asarray(calib_mat['P2'][0:3,]).astype(np.float32)

            obj = pyxyz.figure_out_xyz_cy(obj, P)
            # print("obj: {}".format(obj))
            x,y,z = obj.x, obj.y, obj.z

            x_lidar, y_lidar, z_lidar = camera_to_lidar(x, y, z, calib_mat)
            if x_lidar >= 70.4 or x_lidar <=0 or y_lidar <= -40.0 or y_lidar >= 40.0 or z_lidar >= 3.5 or z_lidar <= -3.5:
                continue
            loc_lidar = [x_lidar, y_lidar, z_lidar]
            locs.append(loc_lidar)
    return locs



def rotation_limit_suppression(pred_rot_angles, rot_angle, diff_limit = np.pi/3):
    diff_rot_angles = np.abs(pred_rot_angles - rot_angle)
    pred_rot_angles_flipped = pred_rot_angles + np.pi
    diff_rot_angles_flipped_1 = np.abs(pred_rot_angles_flipped - rot_angle)

    rot_angle_flipped = rot_angle + np.pi
    diff_rot_angles_flipped_2 = np.abs(pred_rot_angles - rot_angle_flipped)

    diff_rot_angles_min = np.minimum(np.minimum(diff_rot_angles, diff_rot_angles_flipped_1), diff_rot_angles_flipped_2)

    diff_rot_angles_idx = diff_rot_angles_min <= diff_limit
    return diff_rot_angles_idx

def rotate_points(points, rz):

    # rz is counter-clockwise direction
    # we need transpose on matrix since we transposed points

    N = points.shape[0]
    points_rotate = np.hstack([points[:, 0:3], np.ones((N, 1))])
    mat = np.zeros((4, 4))
    mat[2, 2] = 1
    mat[3, 3] = 1
    mat[0, 0] = np.cos(rz)
    mat[0, 1] = -np.sin(rz)
    mat[1, 0] = np.sin(rz)
    mat[1, 1] = np.cos(rz)
    mat = np.transpose(mat)
    points[:, 0:3] = np.matmul(points_rotate, mat)[:, 0:3]

    return points

def normalize_angle(rots):
    r_int = rots // np.pi
    rots_ = rots - r_int * np.pi
    return rots_

def translate_points(points, x_delta, y_delta, z_delta, copy=False):
    if copy == True:
        points_trans = np.copy(points)
    else:
        points_trans = points
    points_trans[:, 0] = points_trans[:, 0] + x_delta
    points_trans[:, 1] = points_trans[:, 1] + y_delta
    points_trans[:, 2] = points_trans[:, 2] + z_delta
    return points_trans

def move_points(points, x_center, y_center, z_center, x_delta, y_delta, z_delta, r_delta):
    # 1) translate whole points, make center of object locates in the origin
    # 2) rotation by r_delta for whole points
    # 3) translate whole points, by x_delta, y_delta, and z_delta

    # 1) translation to the origin
    points = translate_points(points, -x_center, -y_center, 0.0) # we don't translate along z direction

    # 2) rotation along z axis
    # It's very important to know that in a Kitti label, rotation is plus in clockwise direction since it actually rotate along ry axis of camera coordinate, not rz axis of lidar coordinate which are opposite direction.
    points = rotate_points(points, -r_delta)

    # 3) translate by delta <- estimation error we will give around ground truth location
    points = translate_points(points, x_delta, y_delta, z_delta) # we don't translate along z direction

    return points

def get_sample_radii(x, y, w, l, r, max_radii=4.0, min_buffer=0.1, max_buffer=0.4):
    corner_box2d = center_to_corner_box2d([x,y,w,l,r])
    dist_corner = np.sqrt(corner_box2d[:,0]**2 + corner_box2d[:,1]**2)

    obj_center_dist = np.sqrt(x**2+y**2)

    far_corner_idx = np.argmax(dist_corner)

    far_corner_dist = max(dist_corner)
    min_radii_1 = far_corner_dist + np.random.uniform(min_buffer, max_buffer)
    # print("far_corner_dist: {:.2f}".format(far_corner_dist))
    sample_radii = min(min_radii_1, max_radii)

    return sample_radii


def center_box_to_bbox(boxes_center, img_w, img_h, P):
    boxes_corner = center_to_corner_box3d(boxes_center)
    boxes_corner = np.reshape(boxes_corner, (-1, 3))
    points_2d = projectToImage_kitti(boxes_corner[:, 0:3].transpose(), P)
    points_2d = points_2d.transpose()
    points_2d = np.reshape(points_2d, (-1, 8, 2))

    boxes2d_x1 = np.clip(np.amin(points_2d[:, :, 0], axis=1), 0, img_w)
    boxes2d_y1 = np.clip(np.amin(points_2d[:, :, 1], axis=1), 0, img_h)
    boxes2d_x2 = np.clip(np.amax(points_2d[:, :, 0], axis=1), 0, img_w)
    boxes2d_y2 = np.clip(np.amax(points_2d[:, :, 1], axis=1), 0, img_h)    

    boxes2d_x1 = np.expand_dims(boxes2d_x1, -1)
    boxes2d_y1 = np.expand_dims(boxes2d_y1, -1)
    boxes2d_x2 = np.expand_dims(boxes2d_x2, -1)
    boxes2d_y2 = np.expand_dims(boxes2d_y2, -1)

    boxes2d = np.hstack([boxes2d_x1, boxes2d_y1, boxes2d_x2, boxes2d_y2])
    return boxes2d    

def filter_by_iou(boxes_center, img_w, img_h, gt_bbox_2d, P, iou_threshold):
    boxes2d = center_box_to_bbox(boxes_center, img_w, img_h, P)
    iou = cal_box2d_iou_match(boxes2d, gt_bbox_2d)
    # print("iou: {}".format(iou))
    iou_idx = iou>iou_threshold
    iou_idx = np.squeeze(iou_idx, 1)

    return iou_idx

def filter_by_objectness(p_obj, obj_threshold):
    return np.squeeze(p_obj >= obj_threshold)

def get_center_angles(rots):
    center_angles = np.ones(len(rots), float) * np.pi
    center_angles[np.abs(rots - np.pi/2.0) <= np.pi/4] = np.pi/2.0
    center_angles[np.abs(rots) < np.pi/4] = 0.0    
    return center_angles

def clip_by_height(points, z_max, z_min):
    clip_idx = np.logical_and(points[:,2] <= z_max, points[:,2] >= z_min)
    return points[clip_idx]


def clip_by_radii(points, r):
    dist = np.sqrt(points[:,0]**2 + points[:,1]**2)
    clip_idx = dist<r
    points_clip = points[clip_idx]
    return points_clip

def check_num_points(points, radii, min_num_points = 5):
    points = clip_by_radii(points, radii)
    if len(points) < min_num_points:
        return False
    else:
        return True

def get_cluster_cls(cluster_anchors, h, w, l):
    n_cluster = len(cluster_anchors)
    if n_cluster == 1:
        return 0
        
    cluster_overlap = np.zeros(n_cluster, dtype=np.float32)

    pred_box = np.array([0.0, 0.0, 0.0, h, w, l, 0.0])
    
    for cluster_idx, cluster_size in enumerate(cluster_anchors):
        cluster_h = cluster_size[0]
        cluster_w = cluster_size[1]
        cluster_l = cluster_size[2]
        cluster_box = np.array([0.0, 0.0, 0.0, cluster_h, cluster_w, cluster_l, 0.0])
        cluster_overlap[cluster_idx] = cal_iou3d(cluster_box, pred_box)

    return np.argmax(cluster_overlap)

def get_cluster_cls_2d(cluster_anchors, w, l):
    n_cluster = len(cluster_anchors)
    if n_cluster == 1:
        return 0
        
    cluster_overlap = np.zeros(n_cluster, dtype=np.float32)

    pred_box = np.array([0.0, 0.0, w, l, 0.0])
    
    for cluster_idx, cluster_size in enumerate(cluster_anchors):
        cluster_w = cluster_size[0]
        cluster_l = cluster_size[1]
        cluster_box = np.array([0.0, 0.0, cluster_w, cluster_l, 0.0])
        cluster_overlap[cluster_idx] = cal_iou_bev(cluster_box, pred_box)

    return np.argmax(cluster_overlap)

def get_cluster_cls_BEV(cluster_anchors, w, l):
    n_cluster = len(cluster_anchors)
    if n_cluster == 1:
        return 0
        
    cluster_overlap = np.zeros(n_cluster, dtype=np.float32)

    pred_box = np.array([0.0, 0.0, 0.0, 0.0, w, l, 0.0])
    
    for cluster_idx, cluster_size in enumerate(cluster_anchors):
        # cluster_h = cluster_size[0]
        cluster_w = cluster_size[0]
        cluster_l = cluster_size[1]
        cluster_box = np.array([0.0, 0.0, 0.0, 0.0, cluster_w, cluster_l, 0.0])
        cluster_overlap[cluster_idx] = cal_iou3d(cluster_box, pred_box)

    return np.argmax(cluster_overlap)


def clip_by_obj_boundary(points, obj_size, obj_dir, buffer_size = 0.8):

    # direction: horizontal = zero, vertical = one
    # sizes: h, w, l
    # For horizontal, clip : x direction [-w/2, w/2] y direction [-l/2, l/2]
    # For vertical, clip : x direction [-l/2, l/2], y direction [-w/2, w/2]
    obj_dir = 'Vertical' if obj_dir == np.pi/2.0 else 0.0

    obj_h, obj_w, obj_l = obj_size[0], obj_size[1], obj_size[2]
    if obj_dir == 'Vertical': # vertical
        bound_x = np.logical_and(points[:, 0] >= -obj_l/2-buffer_size, points[:, 0] <= obj_l/2+buffer_size)
        bound_y = np.logical_and(points[:, 1] >= -obj_w/2-buffer_size, points[:, 1] <= obj_w/2+buffer_size)
    else: # horizontal
        bound_x = np.logical_and(points[:, 0] >= -obj_w/2-buffer_size, points[:, 0] <= obj_w/2+buffer_size)
        bound_y = np.logical_and(points[:, 1] >= -obj_l/2-buffer_size, points[:, 1] <= obj_l/2+buffer_size)

    bound = np.logical_and(bound_x, bound_y)
    return points[bound]

def clip_by_BEV_box(points, obj_location, obj_size, ry, buffer_size = 0.1, exclude_ground=0.2):
    #WZN: bird's eye view bounding box, in camera frame
    #points is Nx3
    # exclude ground is to threshold distance to subtract points near the ground so as to get rid of them
    Q = np.array([[math.cos(ry),0,-math.sin(ry)],[0,1,0],[math.sin(ry),0,math.cos(ry)]])
    x, y, z = obj_location[0], obj_location[1], obj_location[2]
    h2, w2, l2 = obj_size[0]/2.0, obj_size[1]/2.0, obj_size[2]/2.0
    max_wl = max(w2,l2)+buffer_size
    #initial filt
    points_clip = points[points[:,0]<x+max_wl,:]
    points_clip = points_clip[points_clip[:,0]>x-max_wl,:]
    points_clip = points_clip[points_clip[:,2]<z+max_wl,:]
    points_clip = points_clip[points_clip[:,2]>z-max_wl,:]
    points_clip = points_clip[points_clip[:,1]>y-2*h2-buffer_size,:]
    points_clip = points_clip[points_clip[:,1]<y+buffer_size-exclude_ground,:]
    #then rotate and filter by box
    points_clip_rotate = np.matmul(Q,(points_clip-np.array([[x,y,z]])).transpose()).transpose()
    clip_idx = points_clip_rotate[:,0]>-l2-buffer_size
    clip_idx = np.logical_and(clip_idx,points_clip_rotate[:,0]<l2+buffer_size)
    clip_idx = np.logical_and(clip_idx,points_clip_rotate[:,2]>-w2-buffer_size)
    clip_idx = np.logical_and(clip_idx,points_clip_rotate[:,2]<w2+buffer_size)
    return points_clip[clip_idx,:]


def obj_match_idx_dist(x, y, xs, ys):
    x_diff = xs - x
    y_diff = ys - y
    dist = np.sqrt(x_diff**2 + y_diff**2)
    min_dist = min(dist)
    idx = np.argmin(dist)
    return idx, min_dist

def obj_match_idx(x, y, xs, ys):
    x_diff = xs - x
    y_diff = ys - y
    dist = np.sqrt(x_diff**2 + y_diff**2)
    idx = np.argmin(dist)
    return idx

def cal_distance(obj, approx_locs):
    N = len(approx_locs)
    output = np.zeros(N, np.float32)

    for i in range(N):
        dist = np.sqrt((obj.x - approx_locs[i][0])**2 + (obj.y - approx_locs[i][1])**2)
        output[i] = dist

    return output

def points_quantization(points, quants_level):
    points = np.floor(points/quants_level) * quants_level
    # print("point shape: {}".format(np.shape(points)))
    points = np.unique(points, axis=0)
    return points

def sample_points_rectangle(points, size, center_angle, num_points, min_required_points=5, buffer_size=1.0):
    points = clip_by_obj_boundary(points, size, center_angle, buffer_size)
    N_points, dims = np.shape(points) # number of points inside the region
    n_sampled = 0
    points_sampled = np.zeros((num_points, dims), np.float32)

    mean_x = 0.0
    mean_y = 0.0
    mean_z = 0.0
    # print("N_points: {}".format(N_points))

    if N_points < min_required_points:
        return points_sampled, False

    n_points_set = num_points // N_points
    if n_points_set >= 1:
        points_n = np.tile(points, n_points_set)
        points_sampled[0:N_points*n_points_set, :] = np.tile(points, (n_points_set,1))
        n_sampled = N_points*n_points_set

    if n_sampled<num_points:
        np.random.shuffle(points)
        points_sampled[n_sampled:num_points, :] = points[0:num_points-n_sampled, :]
        n_sampled = num_points

    return points_sampled, True

def sample_points(points, cyl_radii, cyl_height, num_points, z_min=-0.5, z_max=2.5, min_required_points=5):
    # Function: sample_points
    # : sample points inside specified radius and height. Currently not use height information
    # input
    # points: list of points, (N, 3)
    # num_points: number of points to be sampled.
    # In case of number of points inside the region(: N_points) is less than num_points, then we resample it.
    # z_min = -0.5
    # z_max = 2.5

    points = clip_by_radii(points, cyl_radii)
    points = clip_by_height(points, z_max, z_min)
    N_points, dims = np.shape(points) # number of points inside the region
    n_sampled = 0
    points_sampled = np.zeros((num_points, dims), np.float32)

    mean_x = 0.0
    mean_y = 0.0
    mean_z = 0.0
    # print("N_points: {}".format(N_points))



    if N_points < min_required_points:
        return points_sampled, False

    n_points_set = num_points // N_points
    if n_points_set >= 1:
        points_n = np.tile(points, n_points_set)
        points_sampled[0:N_points*n_points_set, :] = np.tile(points, (n_points_set,1))
        n_sampled = N_points*n_points_set

    if n_sampled<num_points:
        np.random.shuffle(points)
        points_sampled[n_sampled:num_points, :] = points[0:num_points-n_sampled, :]
        n_sampled = num_points

    return points_sampled, True

def cal_iou(obj, sample):
    gt_obj = shapely.geometry.box(-obj.l / 2.0, -obj.w / 2.0, obj.l / 2.0, obj.w / 2.0)
    gt_obj = shapely.affinity.rotate(gt_obj, -obj.r, use_radians=True)
    gt_obj = shapely.affinity.translate(gt_obj, -obj.y, obj.x)

    sample_area = shapely.geometry.Point(-sample.y, sample.x).buffer(sample.r)

    intersect = sample_area.intersection(gt_obj)
    gt_obj_area = obj.l * obj.w

    ratio = intersect.area / gt_obj_area

    return ratio


def cal_sample_iou(gt_objs, sample_locs, sample_radiis):
    N1 = len(gt_objs)
    N2 = len(sample_locs)

    output = np.zeros((N1, N2), dtype=np.float32)

    obj = edict({})
    sample = edict({})

    for idx in range(N1):
        for idy in range(N2):
            obj.x = gt_objs[idx, 0]
            obj.y = gt_objs[idx, 1]
            obj.h = gt_objs[idx, 3]
            obj.w = gt_objs[idx, 4]
            obj.l = gt_objs[idx, 5]
            obj.r = gt_objs[idx, 6]

            sample.x = sample_locs[idy, 0]
            sample.y = sample_locs[idy, 1]
            sample.r = sample_radiis[idy]

            output[idx, idy] = float(cal_iou(obj, sample))
    return output

def cal_box_points_iouBEV(boxBEV, points):
    #WZN: use convex hull to fit the geometry of points
    #points must be nx2
    gt_obj = shapely.geometry.box(-boxBEV[2] / 2.0, -boxBEV[3] / 2.0, boxBEV[2] / 2.0, boxBEV[3] / 2.0)
    #gt_obj2 = shapely.geometry.box(-boxBEV[2] / 2.0, -boxBEV[3] / 2.0, boxBEV[2] / 2.0, boxBEV[3] / 2.0)
    gt_obj = shapely.affinity.rotate(gt_obj, -boxBEV[4], use_radians=True)
    #gt_obj2 = shapely.affinity.rotate(gt_obj2, boxBEV[4], use_radians=True)
    gt_obj = shapely.affinity.translate(gt_obj, boxBEV[0], boxBEV[1])
    #gt_obj2 = shapely.affinity.translate(gt_obj2, boxBEV[0], boxBEV[1])
    convex_hull = shapely.geometry.MultiPoint(points).convex_hull
    intersect = convex_hull.intersection(gt_obj)
    intersection = intersect.area
    #intersect2 = convex_hull.intersection(gt_obj2)
    #intersection2 = intersect2.area
    iou = intersection/(convex_hull.area + gt_obj.area - intersection)
    return iou


'''

def cal_box3d_iou(boxes3d_a, boxes3d_b, cal_3d=0):
    # Inputs:
    #   boxes3d: (N1, 7) x,y,z,h,w,l,r
    #   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    N1 = len(boxes3d_a)
    N2 = len(boxes3d_b)
    output = np.zeros((N1, N2), dtype=np.float32)

    for idx in range(N1):
        for idy in range(N2):
            if cal_3d:
                output[idx, idy] = float(
                    cal_iou3d(boxes3d_a[idx], boxes3d_b[idy]))
            else:
                output[idx, idy] = float(
                    cal_iou2d(boxes3d_a[idx, [0, 1, 4, 5, 6]], boxes3d_b[idy, [0, 1, 4, 5, 6]]))

    return output
'''

def check_objectness(x, y, w, l, r, radii, threshold = 0.7):

    # parameters are at lidar coordinate
    # if ratio between intersection of sampling area and gt object over gt object area is greater than threshold, then objectness is 1.0
    # if not, then objectness if 0.0
    # if objectness is true, then iou confidence loss and every regression loss should be optimized
    # if objectness is false, then iou confidence should be set to zero.

    gt_obj = shapely.geometry.box(-l / 2.0, -w / 2.0, l / 2.0, w / 2.0)
    gt_obj = shapely.affinity.rotate(gt_obj, -r, use_radians=True)
    gt_obj = shapely.affinity.translate(gt_obj, -y, x)

    sample_area = shapely.geometry.Point(0.0, 0.0).buffer(radii)

    intersect = sample_area.intersection(gt_obj)
    gt_obj_area = l * w

    ratio = intersect.area / gt_obj_area

    inside = 1.0
    if ratio > threshold:
        inside = 1.0
    else:
        inside = 0.0
    return inside

    # print("intersect: {:.2f} gt area: {:.2f} ratio {:.2f}".format(intersect.area, gt_obj_area, ratio))


def check_objectness_v1(x, y, z, radii, height=4.0):
    # currently we only use x and y for objectness
    # if center is within range of radii, then objectness is true, else false
    dist = np.sqrt(x**2 + y**2)
    inside = 1.0
    if dist < radii:
        inside = 1.0
    else:
        inside = 0.0
    return inside


def read_calib_mat(calib_dir, img_idx, tracking=False):
    if tracking:
        calib_f = "{}/{:04d}.txt".format(calib_dir, img_idx)
    else:
        calib_f = "{}/{:06d}.txt".format(calib_dir, img_idx)
    if os.path.exists(calib_f):
        with open(calib_f) as fi:
            lines = fi.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        P2 = np.reshape(P2, (3, 4))

        P2 = np.vstack((P2, np.array([0,0,0,0])))

        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        R0 = np.reshape(R0, (3, 3))
        R0 = np.hstack((R0, np.array([[0],[0],[0]])))
        R0 = np.vstack((R0, np.array([0,0,0,1])))

        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        Tr_velo_to_cam = np.reshape(Tr_velo_to_cam, (3, 4))
        Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, np.array([0,0,0,1])))

        # warn("P2: {} R0 :{} Tr: {}".format(P2, R0, Tr_velo_to_cam))
    else:
        P2 = np.eye(4)
        P2[3,3] = 0
        R0 = np.eye(4)
        Tr_velo_to_cam = np.eye(4)

    return {'P2' : P2,
            'R0' : R0,
            'Tr_velo2cam' : Tr_velo_to_cam}

def read_calib_mat_paul(calib_f):
    with open(calib_f) as fi:
        lines = fi.readlines()

    calib = {}
    for line in lines:
        obj = line.strip().split(' ')
        if len(obj) < 2: continue
        name = obj[0][:-1]
        P = np.array(obj[1:], dtype=np.float32)
        P = np.reshape(P, (3, -1))
        calib[name] = P
    return calib


if __name__ == '__main__':
    x = 0.0
    y = 0.0
    w = 200.0
    l = 400.0
    r = 100
    radii = 100.0

    check_objectness(x, y, w, l, r, radii, threshold = 0.8)

    x = 1
    y = 0
    w = 1
    l = 5
    r = np.pi/4
    box = np.array([x,y,w,l,r])

    print(get_sample_radii(x,y,w,l,r))







# ---------------------------------- For 3d-bbox visualization
"""
3D box representation:
    bboxes = 8 x 3 matrix.
           = np.array([[-length, 0, -width],          # lower in y
                       [-length, 0,  width],
                       [ length, 0,  width],
                       [ length, 0, -width],
                       [-length, -2*height, -width],  # higher in y
                       [-length, -2*height,  width],
                       [ length, -2*height,  width],
                       [ length, -2*height, -width]])/2
"""


def draw_3dbox(bbox, ax):
    """
        Note that ax should be defined as a 3d plot.
        example)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    """
    ax.scatter(xs=bbox[:,0], ys=bbox[:,1], zs=bbox[:,2], marker='o')
    ax.plot(bbox[[0,1,5,4,0],0], bbox[[0,1,5,4,0],1], bbox[[0,1,5,4,0],2])  # front
    ax.plot(bbox[[2,3,7,6,2],0], bbox[[2,3,7,6,2],1], bbox[[2,3,7,6,2],2])  # back
    ax.plot(bbox[[0,3],0], bbox[[0,3],1], bbox[[0,3],2])
    ax.plot(bbox[[1,2],0], bbox[[1,2],1], bbox[[1,2],2])
    ax.plot(bbox[[4,7],0], bbox[[4,7],1], bbox[[4,7],2])
    ax.plot(bbox[[5,6],0], bbox[[5,6],1], bbox[[5,6],2])


def draw_2dbox(bbox, ax):
    face_front = Polygon(bbox[[2, 3, 7, 6], :], True, lw=0.2, alpha=0.4, color='C1')
    face_back = Polygon(bbox[[0, 1, 5, 4], :], True, lw=0.2, alpha=0.4, color='C0')
    ax.plot(bbox[[2, 3, 7, 6, 2], 0], bbox[[2, 3, 7, 6, 2], 1], color='C1')
    ax.plot(bbox[[0, 1, 5, 4, 0], 0], bbox[[0, 1, 5, 4, 0], 1], color='C0')
    ax.plot(bbox[[0, 3], 0], bbox[[0, 3], 1], lw=1, c='C2', linestyle='-')
    ax.plot(bbox[[1, 2], 0], bbox[[1, 2], 1], lw=1, c='C3', linestyle='-')
    ax.plot(bbox[[4, 7], 0], bbox[[4, 7], 1], lw=1, c='C2', linestyle='-')
    ax.plot(bbox[[5, 6], 0], bbox[[5, 6], 1], lw=1, c='C3', linestyle='-')
    ax.add_patch(face_back)
    ax.add_patch(face_front)


def rotationM(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))


def draw_3dbox_in_2d(obj, P, ax):
    """ This function draws all the 6 faces of a 3D box """
    bbox2 = get_coords_2d(obj, P)
    draw_2dbox(bbox2, ax)

def draw_3dbox_in_2d_cv2(image, obj, P):
    """
    This function draws all the 6 faces of a 3D box on the given image
    using OPENCV functions.
    """
    import cv2
    bbox2 = get_coords_2d(obj, P).astype(np.int)
    bbox2 = tuple(map(tuple, bbox2))
    cv2.line(image, bbox2[2], bbox2[3], (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, bbox2[3], bbox2[7], (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, bbox2[7], bbox2[6], (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, bbox2[6], bbox2[2], (255, 0, 0), 1, cv2.LINE_AA)

    cv2.line(image, bbox2[0], bbox2[1], (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(image, bbox2[1], bbox2[5], (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(image, bbox2[5], bbox2[4], (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(image, bbox2[4], bbox2[0], (0, 255, 0), 1, cv2.LINE_AA)

    cv2.line(image, bbox2[0], bbox2[3], (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(image, bbox2[4], bbox2[7], (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(image, bbox2[1], bbox2[2], (255, 255, 0), 1, cv2.LINE_AA)
    cv2.line(image, bbox2[5], bbox2[6], (255, 255, 0), 1, cv2.LINE_AA)
    return image

def clip(subjectPolygon, clipPolygon):
    # polygon intersection
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []

        if len(inputList) == 0: return outputList
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
    return(outputList)


def get_coords_3d(obj):
    # Rl = obj.Rl
    height = obj.height
    width = obj.width
    length = obj.length
    x = obj.x
    y = obj.y
    z = obj.z
    bbox = np.array([[-length, 0, -width],
                     [-length, 0,  width],
                     [ length, 0,  width],
                     [ length, 0, -width],
                     [-length, -2*height, -width],
                     [-length, -2*height,  width],
                     [ length, -2*height,  width],
                     [ length, -2*height, -width]])/2
    if abs(z) < 1e-6: z = np.sign(z) * 1e-6
    M = rotationM([0, 1, 0], obj.R)
    bbox = np.transpose(np.dot(M, np.transpose(bbox)))
    bbox3 = bbox + np.array([x, y, z])
    return bbox3

def get_coords_2d(obj, P):
    """
    calculate the image (2d) coordinates of the 3d bounding box
    of an object.
    """
    bbox3 = get_coords_3d(obj)
    # Rl = obj.Rl
    # height = obj.height
    # width = obj.width
    # length = obj.length
    # x = obj.x
    # y = obj.y
    # z = obj.z
    # bbox = np.array([[-length, 0, -width],
    #                  [-length, 0,  width],
    #                  [ length, 0,  width],
    #                  [ length, 0, -width],
    #                  [-length, -2*height, -width],
    #                  [-length, -2*height,  width],
    #                  [ length, -2*height,  width],
    #                  [ length, -2*height, -width]])/2
    # if abs(z) < 1e-6: z = np.sign(z) * 1e-6
    # M = rotationM([0, 1, 0], Rl + math.atan(x/z))
    # bbox = np.transpose(np.dot(M, np.transpose(bbox)))
    # bbox3 = bbox + np.array([x, y, z])
    bbox2 = project_pts3_to_image(bbox3, P)
    return bbox2


def get_coord_2d(obj, P, image_size):
    bbox2 = get_coords_2d(obj, P)
    # ignore outside image part.
    hull = ConvexHull(bbox2)
    bbox2 = clip(bbox2[hull.vertices, :].tolist(), [[0, 0], [image_size[1], 0], [image_size[1], image_size[0]], [0, image_size[0]]])
    if not bbox2: return None
    # bounding box
    bbox2 = np.array(bbox2)
    bbox2 = np.array([np.min(bbox2[:, 0]), np.min(bbox2[:, 1]), np.max(bbox2[:, 0]), np.max(bbox2[:, 1])])
    return bbox2


def measure(box1, box2):
    return np.sqrt(np.sum(np.square(box1[:2] - box2[:2]))) + np.sqrt(np.sum(np.square(box1[2:] - box2[2:])))


def wrapToPi(rad):
    rad = rad % (2 * np.pi)
    if rad > np.pi: rad -= 2 * np.pi
    return rad
