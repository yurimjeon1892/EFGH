import os
import random
import numpy as np
import torch.utils.data as data
import csv

from data_loader.loader_utils import *

__all__ = ['KITTI_ODOM']

class KITTI_ODOM(data.Dataset):
    """
    Args:
        mode:
        process_data (callable):
        generate_data (callable):
        args:
    """

    def __init__(self, mode, args):
        self.mode = mode
        self.process_data = ProcessKITTIODOM(args)
        self.data_path = os.path.join(args['data_root'], 'dataset')
        self.accumulation_frame_num = args['accumulation_frame_num']
        self.accumulation_frame_skip = args['accumulation_frame_skip']       
        
        if mode == 'train':
            self.num_samples = args['train_samples']
            self.sequences = args['sequences']['train']
            self.delta_ij_max = args['delta_ij_max']
            self.translation_max = args['translation_max']
            self.samples = self.make_sample_dataset()
        elif mode == 'valid':
            self.num_samples = args['val_samples']
            self.sequences = args['sequences']['valid']
            self.delta_ij_max = args['delta_ij_max']
            self.translation_max = args['translation_max']
            self.samples = self.make_sample_dataset()
        elif mode == 'test':
            self.num_samples = args['val_samples']
            self.sequences = args['sequences']['test']     
            self.rand_init_params = {}
            f = open(args['rand_init'], 'r')
            rdr = csv.reader(f)
            for line in rdr:
                self.rand_init_params[line[0]] = [float(i) for i in line[1:]] 
            f.close()        
            self.samples = self.make_test_sample_dataset(self.rand_init_params)
        else:
            print('worng mode: ', mode)
            exit()
        
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pcd, img, calibs, posej_T_posei, fname = self.file_reader(self.samples[index]) 
        if self.mode != 'test': rand_init = None                 
        else: rand_init = self.rand_init_params[fname]  
        return self.process_data(pcd, img, calibs, posej_T_posei, fname, rand_init=rand_init)

    def get_sequence_j(self, poses, calibs, seq_i):
        seq_sample_num = len(poses)
        # get the max and min of possible j
        seq_j_min = max(seq_i - self.delta_ij_max, 0)
        seq_j_max = min(seq_i + self.delta_ij_max, seq_sample_num - 1)

        # pose of i
        Pi = pose_read(poses[seq_i])

        while True:
            seq_j = random.randint(seq_j_min, seq_j_max)
            # get the pose, if the pose is too large, ignore and re-sample
            Pj = pose_read(poses[seq_j])
            posej_T_posei = calibs["Tr_inv"] @ np.linalg.inv(Pj) @ Pi @ calibs["Tr"] # 4x4
            t_ji = posej_T_posei[0:3, 3]  # 3
            t_ji_norm = np.linalg.norm(t_ji)  # scalar
            if t_ji_norm < self.translation_max: break
            else: continue

        return seq_j, posej_T_posei

    def make_test_sample_dataset(self, rand_init):  
        sample_list = []
        for seq in self.sequences:
            seq = '{0:02d}'.format(int(seq))
            print("parsing seq {}".format(seq))            

            calib_fn = os.path.join(self.data_path, "sequences", seq, "calib.txt")
            calibs = calib_read(calib_fn)

            f = open(os.path.join(self.data_path, 'poses', seq + '.txt'), 'r')
            poses = f.readlines()
            f.close()

            for k in rand_init.keys():
                seq1, seq_i, seq_j = k.split("_")[0], int(k.split("_")[1]), int(k.split("_")[2])
                if seq1 != seq : continue     

                Pi = pose_read(poses[seq_i])
                Pj = pose_read(poses[seq_j])
                posej_T_posei = calibs["Tr_inv"] @ np.linalg.inv(Pj) @ Pi @ calibs["Tr"] # 4x4

                str_seq_i = str(seq_i).zfill(6)
                str_seq_j = str(seq_j).zfill(6)  

                indiv_sample = {'image': os.path.join(self.data_path, 'sequences', seq, 'image_2', str_seq_j + '.png'),
                                'velodyne_raw': os.path.join(self.data_path, 'sequences', seq, 'velodyne', str_seq_i + '.bin'),
                                'calib': calibs,
                                'posej_T_posei': posej_T_posei,
                                'fname': k}
                sample_list.append(indiv_sample)

        if self.num_samples > 0:
            sample_list = sample_list[:self.num_samples]
        else:
            self.num_samples = len(sample_list)
        return sample_list

    def make_sample_dataset(self):  
        sample_list = []
        for seq in self.sequences:
            seq = '{0:02d}'.format(int(seq))
            print("parsing seq {}".format(seq))            

            calib_fn = os.path.join(self.data_path, "sequences", seq, "calib.txt")
            calibs = calib_read(calib_fn)

            f = open(os.path.join(self.data_path, 'poses', seq + '.txt'), 'r')
            poses = f.readlines()
            f.close()

            file_list = os.listdir(os.path.join(self.data_path, 'sequences', seq, 'velodyne'))
            for seq_i in range(0, len(file_list)):

                seq_j, posej_T_posei = self.get_sequence_j(poses, calibs, seq_i)

                str_seq_i = str(seq_i).zfill(6)
                str_seq_j = str(seq_j).zfill(6)  

                indiv_sample = {'image': os.path.join(self.data_path, 'sequences', seq, 'image_2', str_seq_j + '.png'),
                                'velodyne_raw': os.path.join(self.data_path, 'sequences', seq, 'velodyne', str_seq_i + '.bin'),
                                'calib': calibs,
                                'posej_T_posei': posej_T_posei,
                                'fname': seq + '_' + str_seq_i + '_' + str_seq_j}
                sample_list.append(indiv_sample)
            # self.samples = self.samples[:10]
        if self.mode == 'train': 
            random.shuffle(sample_list)      

        if self.num_samples > 0:
            sample_list = sample_list[:self.num_samples]
        else:
            self.num_samples = len(sample_list)
        return sample_list

    def search_for_accumulation(self, pcd_dir, seq, seq_i, seq_sample_num, calibs, P_oi, stride):
        f = open(os.path.join(self.data_path, 'poses', seq + '.txt'), 'r')
        poses = f.readlines()
        f.close()        

        P_io = np.linalg.inv(P_oi)

        pc_np_list = []

        counter = 0
        while len(pc_np_list) < self.accumulation_frame_num:
            counter += 1
            seq_j = seq_i + stride * counter
            if seq_j < 0 or seq_j >= seq_sample_num:
                break

            str_seq_j = str(seq_j).zfill(6) 
            # print('str_seq_ij: ', seq_i, str_seq_j)

            pc_j = pcd_read(os.path.join(pcd_dir, str_seq_j + '.bin'))
            pc_j = pc_j.T
            P_oj = pose_read(poses[seq_j])
            P_ij = P_io @ P_oj

            pc_j = np.concatenate((pc_j[:3, :], np.ones((1, pc_j.shape[1]), dtype=pc_j.dtype)), axis=0)
            pc_j = calibs['Tr_inv'] @ P_ij @ calibs['Tr'] @ pc_j
            pc_j = pc_j[:3, :]

            pc_np_list.append(pc_j)

        return pc_np_list

    def get_accumulated_pc(self, pcd_path, seq, seq_i, calibs):

        pc_np = pcd_read(pcd_path)
        pc_np = pc_np.T
        # shuffle the point cloud data, this is necessary!
        pc_np = pc_np[:, np.random.permutation(pc_np.shape[1])]
        pc_np = pc_np[:3, :]  # 3xN

        if self.accumulation_frame_num <= 0.5: return pc_np.T

        pc_np_list = [pc_np]

        # pose of i
        f = open(os.path.join(self.data_path, 'poses', seq + '.txt'), 'r')
        poses = f.readlines()
        f.close()
        seq_sample_num = len(poses)

        P_oi = pose_read(poses[seq_i])   

        pcd_paths = pcd_path.split('/')[:-1]
        pcd_dir = os.path.join('/', *pcd_paths)
        # search for previous
        prev_pc_np_list = self.search_for_accumulation(pcd_dir, seq, seq_i, seq_sample_num,
                                                       calibs, P_oi, -self.accumulation_frame_skip)
        # search for next
        next_pc_np_list = self.search_for_accumulation(pcd_dir, seq, seq_i, seq_sample_num,
                                                       calibs, P_oi, self.accumulation_frame_skip)

        pc_np_list = pc_np_list + prev_pc_np_list + next_pc_np_list

        pc_np = np.concatenate(pc_np_list, axis=1)

        return pc_np.T

    def file_reader(self, sample_one):
        """
        :param sample_path:
        :return:
        """
        seq, str_seq_i = sample_one['fname'].split('_')[0], sample_one['fname'].split('_')[1]
        pcd = self.get_accumulated_pc(sample_one['velodyne_raw'], seq, int(str_seq_i), sample_one['calib'])
        img = rgb_read(sample_one['image'])
        return pcd, img, sample_one['calib'], sample_one['posej_T_posei'], sample_one['fname']

class ProcessKITTIODOM(object):

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

    def __call__(self, pcd, img, calibs, posej_T_posei, fname, rand_init=None):

        rr, rp, ry, tx, ty, tz, rt = rand_init_params(rand_init, self.l_rot_range, self.l_trs_range, self.c_rot_range)

        gts = preproc_gt(rr, rp, ry, tx, ty, tz, rt, posej_T_posei)
        imgs = preproc_img(img, gts, self.raw_cam_img_size)
        pc = preproc_pcd(pcd, gts, self.num_points, self.lidar_line)

        img = imgs['in']
        gts['img_raw'] = imgs['raw']
        gts['img_rot'] = imgs['rot']  
        gts['img_mask'] = imgs['img_mask']     

        A = np.array([[1, 0, -self.raw_cam_img_size[1] / 2],
                      [0, 1, -self.raw_cam_img_size[0] / 2],
                      [0, 0, 1]])

        calib = calibs['P2'] @ calibs['Tr']
        calib = calib[:3, :]

        gts['cam_T_velo'] = np.linalg.inv(A) @ gts['intrinsic_sensor2'] @ A @ calib @ gts['sensor2_T_sensor1']    

        return pc[:3, :], img, calib, A, gts, fname