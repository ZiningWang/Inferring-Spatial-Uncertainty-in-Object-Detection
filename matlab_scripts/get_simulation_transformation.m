%% add paths
addpath(genpath('../../IROS/kitti_visualizer/matlab_scripts/'))
%% parameters
data_path = 'D:\Berkeley\Kitti_data\object\training\';
base_dir = [data_path,'velodyne/'];
calib_dir = [data_path,'calib/'];
label_dir = [data_path,'label_2/'];
path_ima = [data_path,'image_2/'];

%% inputs
frame = 2341;
ego_transform = [-17270 , 5400, 10, 20]';

%% calculate absolute position of other vehicle labels
ry_ego = ego_transform(4)/180*pi;
labels_stat=gt_statistics(label_dir,[frame]);
n = size(labels_stat{1}.pos,1);
rotmat = [cos(ry_ego), -sin(ry_ego), 0; sin(ry_ego), cos(ry_ego), 0; 0,0,1];
fprintf('frame: %d\n', frame)
for i = 1:n
    gt_idx = labels_stat{1}.gt_idx(i);
    pos = labels_stat{1}.pos(i,[3,1,2])'*1e2;
    transform = round(ego_transform(1:3)+rotmat*pos);
    ry = round(labels_stat{1}.pos(i,4)*180/pi-90+(ego_transform(4)-180));
    if ry< -180
        ry = ry+360;
    end
    fprintf('%d, %d, %d, %d, %d\n',gt_idx, transform(1), transform(2), ego_transform(3), ry);
end