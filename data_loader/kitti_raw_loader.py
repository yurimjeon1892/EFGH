import os
import random
import numpy as np
import torch.utils.data as data

from collections import namedtuple
import csv

import data_loader.pykitti_utils as pu
from data_loader.loader_utils import *

__all__ = ['KITTI_RAW']

class KITTI_RAW(data.Dataset):
    """
    Args:
        mode:
        process_data (callable):
        generate_data (callable):
        args:
    """

    def __init__(self, mode, args):
        self.mode = mode
        self.process_data = ProcessKITTIRAW(args)
        self.data_path = args['data_root']
        self.cams = ['image_02']
        if mode == 'train':
            self.dates = ['2011_09_26']           
            self.drive_list = ['0001', '0002', '0009', '0011', '0013', 
                            '0014', '0015', '0017', '0018', '0019',
                            '0020', '0022', '0023', '0027', '0028',
                            '0029', '0032', '0035', '0036', '0039',
                            '0046', '0048', '0051', '0052', '0056',
                            '0057', '0059', '0060', '0061', '0064',
                            '0079', '0084', '0086', '0087', '0091',
                            '0093', '0095', '0096', '0101', '0104',
                            '0106', '0113', '0117', '0119',
                            ] 
            self.num_samples = args['train_samples']
            self.samples = self.make_sample_dataset()
        elif mode == 'valid':
            self.dates = ['2011_09_26']   
            self.drive_list = ['0005','0070']     
            self.num_samples = args['val_samples']
            self.samples = self.make_sample_dataset()
        elif mode == 'test':
            self.dates = ['2011_09_30']       
            self.drive_list = ['0028']     
            self.num_samples = args['val_samples']
            self.samples = self.make_sample_dataset()
            self.rand_init = {}
            f = open(args['rand_init'], 'r')
            rdr = csv.reader(f)
            for line in rdr:
                self.rand_init[line[0]] = [float(i) for i in line[1:]] 
            f.close()               
        else:
            print('worng mode: ', mode)
            exit()
        
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pcd, img, calibs, cam, fname = \
            self.file_reader(self.samples[index])         
        if self.mode != 'test': rand_init = None                 
        else: 
            k = fname.split("/")[-1]
            rand_init = self.rand_init[k]  
        return self.process_data(pcd, img, calibs, cam, fname, rand_init=rand_init)

    def make_sample_dataset(self):  
        sample_list = []
        for date in self.dates:               
            for dnum in self.drive_list:
                drive = date + '_drive_' + dnum + '_sync'
                file2_list = os.listdir(os.path.join(self.data_path, date, drive, 'velodyne_points', 'data'))
                for fname in file2_list:    
                    fname = fname[:-4]      
                    for cam in self.cams:     
                        indiv_sample = {'image': os.path.join(self.data_path, date, drive, 
                                                            cam, 'data', fname + '.png'),
                                        'velodyne_raw': os.path.join(self.data_path, date, drive, 
                                                                    'velodyne_points', 'data', fname + '.bin'),
                                        'calib_path': os.path.join(self.data_path, date),
                                        'fname': os.path.join(date, drive, fname),
                                        'cam': cam}
                        sample_list.append(indiv_sample)
        if self.mode == 'train': 
            random.shuffle(sample_list)        
        elif self.mode == 'test': 
            int_ids = []
            for sample in sample_list:
                str_ids = sample['image'].split('/')[-1][:-4]
                int_ids.append(int(str_ids))
            sortids = sorted(range(len(int_ids)), key=lambda k: int_ids[k])
            sorted_sample_list = []
            for _id in list(sortids):
                sorted_sample_list.append(sample_list[_id])
            sample_list = sorted_sample_list

        if self.num_samples > 0:
            sample_list = sample_list[:self.num_samples]
        else:
            self.num_samples = len(sample_list)
        return sample_list

    def calib_read(self, calib_path):

        def _load_calib_rigid(calib_path, filename):
            """Read a rigid transform calibration file as a numpy.array."""
            filepath = os.path.join(calib_path, filename)
            data = pu.read_calib_file(filepath)
            return pu.transform_from_rot_trans(data['R'], data['T'])

        def _load_calib_cam_to_cam(calib_path, velo_to_cam_file, cam_to_cam_file):
            # We'll return the camera calibration as a dictionary
            data = {}

            # Load the rigid transformation from velodyne coordinates
            # to unrectified cam0 coordinates
            T_cam0unrect_velo = _load_calib_rigid(calib_path, velo_to_cam_file)
            data['T_cam0_velo_unrect'] = T_cam0unrect_velo

            # Load and parse the cam-to-cam calibration data
            cam_to_cam_filepath = os.path.join(calib_path, cam_to_cam_file)
            filedata = pu.read_calib_file(cam_to_cam_filepath)

            # Create 3x4 projection matrices
            P_rect_00 = np.reshape(filedata['P_rect_00'], (3, 4))
            P_rect_10 = np.reshape(filedata['P_rect_01'], (3, 4))
            P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
            P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))

            data['P_rect_00'] = P_rect_00
            data['P_rect_10'] = P_rect_10
            data['P_rect_20'] = P_rect_20
            data['P_rect_30'] = P_rect_30

            # Create 4x4 matrices from the rectifying rotation matrices
            R_rect_00 = np.eye(4)
            R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))
            R_rect_10 = np.eye(4)
            R_rect_10[0:3, 0:3] = np.reshape(filedata['R_rect_01'], (3, 3))
            R_rect_20 = np.eye(4)
            R_rect_20[0:3, 0:3] = np.reshape(filedata['R_rect_02'], (3, 3))
            R_rect_30 = np.eye(4)
            R_rect_30[0:3, 0:3] = np.reshape(filedata['R_rect_03'], (3, 3))

            data['R_rect_00'] = R_rect_00
            data['R_rect_10'] = R_rect_10
            data['R_rect_20'] = R_rect_20
            data['R_rect_30'] = R_rect_30

            # Compute the rectified extrinsics from cam0 to camN
            T0 = np.eye(4)
            T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
            T1 = np.eye(4)
            T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
            T2 = np.eye(4)
            T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
            T3 = np.eye(4)
            T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

            # Compute the velodyne to rectified camera coordinate transforms
            data['T_cam0_velo'] = P_rect_00.dot(R_rect_00.dot(T_cam0unrect_velo))
            data['T_cam1_velo'] = P_rect_10.dot(R_rect_00.dot(T_cam0unrect_velo))
            data['T_cam2_velo'] = P_rect_20.dot(R_rect_00.dot(T_cam0unrect_velo))
            data['T_cam3_velo'] = P_rect_30.dot(R_rect_00.dot(T_cam0unrect_velo))

            # # Compute the camera intrinsics
            # data['K_cam0'] = P_rect_00[0:3, 0:3]
            # data['K_cam1'] = P_rect_10[0:3, 0:3]
            # data['K_cam2'] = P_rect_20[0:3, 0:3]
            # data['K_cam3'] = P_rect_30[0:3, 0:3]

            # Compute the stereo baselines in meters by projecting the origin of
            # each camera frame into the velodyne frame and computing the distances
            # between them
            # p_cam = np.array([0, 0, 0, 1])
            # p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
            # p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
            # p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
            # p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

            # data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
            # data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

            return data

        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the rigid transformation from IMU to velodyne
        data['T_velo_imu'] = _load_calib_rigid(calib_path, 'calib_imu_to_velo.txt')

        # Load the camera intrinsics and extrinsics
        data.update(_load_calib_cam_to_cam(calib_path, 
            'calib_velo_to_cam.txt', 'calib_cam_to_cam.txt'))

        # # Pre-compute the IMU to rectified camera coordinate transforms
        # data['T_cam0_imu'] = data['T_cam0_velo'].dot(data['T_velo_imu'])
        # data['T_cam1_imu'] = data['T_cam1_velo'].dot(data['T_velo_imu'])
        # data['T_cam2_imu'] = data['T_cam2_velo'].dot(data['T_velo_imu'])
        # data['T_cam3_imu'] = data['T_cam3_velo'].dot(data['T_velo_imu'])

        calibs = namedtuple('CalibData', data.keys())(*data.values())
        return calibs

    def file_reader(self, sample_one):
        """
        :param sample_path:
        :return:
        """
        pcd = pcd_read(sample_one['velodyne_raw'])
        img = rgb_read(sample_one['image'])
        calibs = self.calib_read(sample_one['calib_path'])
        return pcd[:, :3], img, calibs, sample_one['cam'], sample_one['fname']

class ProcessKITTIRAW(object):

    def __init__(self, args):
        self.raw_cam_img_size = args['raw_cam_img_size']
        self.lidar_line = args['lidar_line']
        self.num_points = args['num_points']
        if args['test'] == False:
            self.l_rot_range = args['dclb']['l_rot_range']
            self.l_trs_range = args['dclb']['l_trs_range']
            self.c_rot_range = args['dclb']['c_rot_range']
        else:
            self.l_rot_range, self.l_trs_range, self.c_rot_range = None, None, None
        return    

    def __call__(self, pcd, img, calibs, cam, fname, rand_init=None):

        rr, rp, ry, tx, ty, tz, rt = rand_init_params(rand_init, self.l_rot_range, self.l_trs_range, self.c_rot_range)

        gts = preproc_gt(rr, rp, ry, tx, ty, tz, rt)
        imgs = preproc_img(img, gts, self.raw_cam_img_size)
        pc = preproc_pcd(pcd, gts, self.num_points, self.lidar_line)

        img = imgs['in']
        gts['img_raw'] = imgs['raw']
        gts['img_rot'] = imgs['rot']  
        gts['img_mask'] = imgs['img_mask']  

        if cam == 'image_02': calib = calibs.T_cam2_velo
        elif cam == 'image_03': calib = calibs.T_cam3_velo

        A = np.array([[1, 0, -self.raw_cam_img_size[1] / 2],
                      [0, 1, -self.raw_cam_img_size[0] / 2],
                      [0, 0, 1]])

        gts['cam_T_velo'] = np.linalg.inv(A) @ gts['intrinsic_sensor2'] @ A @ calib @ gts['sensor2_T_sensor1']    

        return pc[:3, :], img, calib, A, gts, fname