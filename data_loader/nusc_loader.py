import os
import random
import numpy as np
import torch.utils.data as data

from PIL import Image
import csv

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from data_loader.loader_utils import *
from data_loader.nusc_utils import *

__all__ = ['NUSC']


class NUSC(data.Dataset):
    """
    Args:
        mode:
        process_data (callable):
        generate_data (callable):
        args:
    """

    def __init__(self, mode, args):
        self.mode = mode
        self.process_data = ProcessNUSC(args)       
        self.data_path = args['data_root']
        
        self.accumulation_frame_num = args['accumulation_frame_num']
        self.accumulation_frame_skip = args['accumulation_frame_skip']

        if mode == 'train':                       
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True) # v1.0-trainval, v1.0-test
            self.num_samples = args['train_samples']            
            self.samples = self.make_sample_dataset()
        elif mode == 'valid':
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True) # v1.0-trainval, v1.0-test
            self.num_samples = args['val_samples']
            self.samples = self.make_sample_dataset()
        elif mode == 'test' :            
            self.nusc = NuScenes(version='v1.0-test', dataroot=self.data_path, verbose=True)
            self.num_samples = -1            
            self.rand_init_params = {}
            f = open(args['rand_init'], 'r')
            rdr = csv.reader(f)
            for line in rdr:
                self.rand_init_params[line[0]] = [float(i) for i in line[1:]] 
            f.close()           
            self.samples = self.make_test_sample_dataset()
        
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        lidar_token, nearby_camera_token = self.samples[index]
        pcd, img, calibs = self.file_reader(lidar_token, nearby_camera_token) 
        if self.mode != 'test': rand_init = None                 
        else: rand_init = self.rand_init_params[lidar_token + '_'+ nearby_camera_token]
        return self.process_data(pcd, img, calibs, lidar_token + '_'+ nearby_camera_token, rand_init=rand_init)

    def make_test_sample_dataset(self):  
        sample_list = []
        for k in self.rand_init_params.keys():
            lidar_token, camera_token = k.split("_")[0], k.split("_")[1]
            sample_list.append([lidar_token, camera_token])   
        return sample_list

    def make_sample_dataset(self):  
        sample_list = make_nuscenes_dataset(self.nusc, frame_skip=20, max_translation=10, mode=self.mode)      
        random.shuffle(sample_list)
        if self.num_samples > 0:
            sample_list = sample_list[:self.num_samples]
        else:
            self.num_samples = len(sample_list)
        return sample_list

    def get_lidar_pc_intensity_by_token(self, lidar_token):
        lidar = self.nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar['filename']))
        pc_np = pc.points[0:3, :]

        # remove point falls on egocar
        x_inside = np.logical_and(pc_np[0, :] < 0.8, pc_np[0, :] > -0.8)
        y_inside = np.logical_and(pc_np[1, :] < 2.7, pc_np[1, :] > -2.7)
        inside_mask = np.logical_and(x_inside, y_inside)
        outside_mask = np.logical_not(inside_mask)
        pc_np = pc_np[:, outside_mask]

        P_oi = get_sample_data_ego_pose_P(self.nusc, lidar)

        return pc_np, P_oi

    def lidar_frame_accumulation(self, lidar, P_io, P_lidar_vehicle, P_vehicle_lidar,
                                 direction, pc_np_list):
        counter = 1
        accumulated_counter = 0
        while accumulated_counter < self.accumulation_frame_num:
            if lidar[direction] == '':
                break

            if counter % self.accumulation_frame_skip != 0:
                counter += 1
                lidar = self.nusc.get('sample_data', lidar[direction])
                continue

            pc_np_j, P_oj = self.get_lidar_pc_intensity_by_token(lidar[direction])
            P_ij = np.dot(P_io, P_oj)
            P_ij_trans = np.dot(np.dot(P_lidar_vehicle, P_ij), P_vehicle_lidar)
            pc_np_j_transformed = transform_pc_np(P_ij_trans, pc_np_j)
            pc_np_list.append(pc_np_j_transformed)

            counter += 1
            lidar = self.nusc.get('sample_data', lidar[direction])
            accumulated_counter += 1

        # print('accumulation %s %d' % (direction, counter))
        return pc_np_list


    def accumulate_lidar_points(self, lidar):
        pc_np_list = []
        # load itself
        pc_np_i, P_oi = self.get_lidar_pc_intensity_by_token(lidar['token'])
        pc_np_list.append(pc_np_i)
        P_io = np.linalg.inv(P_oi)

        P_vehicle_lidar = get_calibration_P(self.nusc, lidar)
        P_lidar_vehicle = np.linalg.inv(P_vehicle_lidar)

        # load next
        pc_np_list = self.lidar_frame_accumulation(lidar, P_io, P_lidar_vehicle, P_vehicle_lidar,
                                                   'next', pc_np_list)

        # load prev
        pc_np_list = self.lidar_frame_accumulation(lidar, P_io, P_lidar_vehicle, P_vehicle_lidar,
                                                   'prev', pc_np_list)

        pc_np = np.concatenate(pc_np_list, axis=1)

        return pc_np

    def file_reader(self, lidar_token, camera_token):
        """
        :param sample_path:
        :return:
        """
        # load point cloud
        pointsensor = self.nusc.get('sample_data', lidar_token)
        pc_np = self.accumulate_lidar_points(pointsensor)
        pcd = pc_np[:3, :].T
        # pcl_path = os.path.join(self.data_path, pointsensor['filename'])
        # pc = LidarPointCloud.from_file(pcl_path)
        # pcd = pc.points[:3, :].T

        lidar_calib_P = get_calibration_P(self.nusc, pointsensor)
        lidar_pose_P = get_sample_data_ego_pose_P(self.nusc, pointsensor)

        cam = self.nusc.get('sample_data', camera_token)        
        img = Image.open(os.path.join(self.data_path, cam['filename'])) 
        img = np.array(img, dtype='uint8')
        K = get_camera_K(self.nusc, cam)

        camera_calib_P = get_calibration_P(self.nusc, cam)
        camera_pose_P = get_sample_data_ego_pose_P(self.nusc, cam)

        camera_pose_P_inv = np.linalg.inv(camera_pose_P)
        camera_calib_P_inv = np.linalg.inv(camera_calib_P)

        T_cam_velo = K @ camera_calib_P_inv[:3, :]
        posej_T_posei = camera_pose_P_inv @ lidar_pose_P @ lidar_calib_P

        # # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        # cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        # r, t = np.eye(4), np.eye(4)
        # r[:3, :3] = np.array(Quaternion(cs_record['rotation']).rotation_matrix)
        # t[:3, 3] = np.array(cs_record['translation'])
        # ego1_T_sensor1 = t @ r

        # # Second step: transform from ego to the global frame.
        # poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        # r, t = np.eye(4), np.eye(4)
        # r[:3, :3] = np.array(Quaternion(poserecord['rotation']).rotation_matrix)
        # t[:3, 3] = np.array(poserecord['translation'])
        # global_T_ego1 = t @ r

        # # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        # poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        # r, t = np.eye(4), np.eye(4)
        # r[:3, :3] = np.array(Quaternion(poserecord['rotation']).rotation_matrix.T)
        # t[:3, 3] = -np.array(poserecord['translation'])
        # ego2_T_global = r @ t

        # # Fourth step: transform from ego into the camera.
        # cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        # r, t = np.eye(4), np.eye(4)
        # r[:3, :3] = np.array(Quaternion(cs_record['rotation']).rotation_matrix.T)
        # t[:3, 3] = -np.array(cs_record['translation'])
        # sensor2_T_ego2 = r @ t

        # view = np.array(cs_record['camera_intrinsic'])
        # intrinsic_sensor2 = np.eye(4)
        # intrinsic_sensor2[:view.shape[0], :view.shape[1]] = view 

        # sensor2_T_sensor1 = intrinsic_sensor2 @ sensor2_T_ego2 @ ego2_T_global @ global_T_ego1 @ ego1_T_sensor1
        # sensor2_T_sensor1 = sensor2_T_sensor1[:3, :]

        # posej_T_posei = sensor2_T_ego2 @ ego2_T_global @ global_T_ego1 @ ego1_T_sensor1

        calibs = {
            # 'ego1_T_sensor1': ego1_T_sensor1,
            # 'global_T_ego1': global_T_ego1,
            # 'ego2_T_global': ego2_T_global,
            # 'sensor2_T_ego2': sensor2_T_ego2,
            'T_cam_velo': T_cam_velo,
            # 'sensor2_T_sensor1': sensor2_T_sensor1,
            'posej_T_posei': posej_T_posei
        }
        return pcd, img, calibs

class ProcessNUSC(object):

    def __init__(self, args):         
        self.raw_cam_img_size = args['raw_cam_img_size']
        self.num_points = args['num_points']
        self.is_test = args['test']
        if args['test'] == False:
            self.l_rot_range = args['dclb']['l_rot_range']
            self.l_trs_range = args['dclb']['l_trs_range']
            self.c_rot_range = args['dclb']['c_rot_range']
        else:
            self.l_rot_range, self.l_trs_range, self.c_rot_range = None, None, None
        return

    def __call__(self, pcd, img, calibs, tokeni_tokenj, rand_init=None):

        rr, rp, ry, tx, ty, tz, rt = rand_init_params(rand_init, self.l_rot_range, self.l_trs_range, self.c_rot_range)
        
        gts = preproc_gt(rr, rp, ry, tx, ty, tz, rt, calibs['posej_T_posei'])
        imgs = preproc_img(img, gts, self.raw_cam_img_size)
        pc = preproc_pcd(pcd, gts, self.num_points)

        img = imgs['in']
        gts['img_raw'] = imgs['raw']
        gts['img_rot'] = imgs['rot']  
        gts['img_mask'] = imgs['img_mask']   

        A = np.array([[1, 0, -self.raw_cam_img_size[1] / 2],
                      [0, 1, -self.raw_cam_img_size[0] / 2],
                      [0, 0, 1]])

        calib = calibs['T_cam_velo']

        gts['cam_T_velo'] = np.linalg.inv(A) @ gts['intrinsic_sensor2'] @ A @ calib @ gts['sensor2_T_sensor1']    

        return pc[:3, :], img, calib, A, gts, tokeni_tokenj