import os
import random
import numpy as np

import yaml
from scipy.spatial.transform import Rotation
from PIL import Image
from math import pi, sin, cos

from common.numpy_utils import rpy_to_matrix, xyz_to_matrix, crop_image, resize_image, zero_pad_image, rotate_image_from_rotation_matrix_numpy, image_valid_mask

def pose_read(line1):
    pose1 = line1.split(' ')
    pose1 = [float(p) for p in pose1]
    pose1 = np.array(pose1, dtype=float)
    pose1 = pose1.reshape((3, 4))
    pose1_eye = np.eye(4)
    pose1_eye[:3, :] = pose1
    # pose1_inv = np.linalg.inv(pose1_eye)
    return pose1_eye   

def calib_read(calib_path):
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    # P0 = np.reshape(data['P0'], (3, 4))
    # P1 = np.reshape(data['P1'], (3, 4))
    P2 = np.reshape(data['P2'], (3, 4))
    # P3 = np.reshape(data['P3'], (3, 4))
    Tr = np.reshape(data['Tr'], (3, 4))

    P2_eye, Tr_eye = np.eye(4), np.eye(4)
    P2_eye[:3, :] = P2
    Tr_eye[:3, :] = Tr   

    calibs = {
        'Tr': Tr_eye,
        'Tr_inv' : np.linalg.inv(Tr_eye),            
        'P2': P2_eye,
        'P2_inv': np.linalg.inv(P2_eye),     
    }

    return calibs

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    return rgb_png

def pcd_read(filename):
    scan = np.fromfile(filename, dtype=np.float32)
    return scan.reshape((-1, 4))

def rand_init_params(rand_init, rpy_range, xyz_range, t_range):

    if rand_init is not None:
        rr, rp, ry, tx, ty, tz, rt = rand_init
    elif rpy_range != None and xyz_range != None and t_range != None:
        rr = (random.random() * 2. - 1.) * pi * rpy_range
        rp = (random.random() * 2. - 1.) * pi * rpy_range 
        ry = (random.random() * 2. - 1.) * pi * rpy_range             
        tx = (random.random() * 2. - 1.) * xyz_range
        ty = (random.random() * 2. - 1.) * xyz_range
        tz = (random.random() * 2. - 1.) * xyz_range            
        rt = (random.random() * 2. - 1.) * pi * t_range
    else:
        print('rand_init_params error')
    return rr, rp, ry, tx, ty, tz, rt

def preproc_gt(rr, rp, ry, tx, ty, tz, rt, posej_T_posei=np.eye(4)):

    lrot = rpy_to_matrix(rr, rp, ry)
    ltrs = xyz_to_matrix(tx, ty, tz)

    rand_init_l = np.array(lrot @ ltrs)
    rand_init_l_inv = np.array(np.linalg.inv(rand_init_l))

    rand_init_c = np.array([[cos(rt), -sin(rt), 0],
                            [sin(rt), cos(rt), 0],
                            [0, 0, 1]])     
    rand_init_c_inv = np.array(np.linalg.inv(rand_init_c))
    
    sensor2_T_sensor1 = posej_T_posei @ rand_init_l_inv
    intrinsic_sensor2 = rand_init_c_inv
    
    gts = {
        'rand_init_l': rand_init_l, # for E
        'rand_init_c': rand_init_c, # for H

        'sensor2_T_sensor1': sensor2_T_sensor1,
        'intrinsic_sensor2': intrinsic_sensor2,
        }

    return gts

def preproc_img(img, gts, raw_cam_img_size):

    img_raw = crop_image(img, raw_cam_img_size, init=True)
    img_rot = rotate_image_from_rotation_matrix_numpy(img, gts['rand_init_c'])    
    img_rot = crop_image(img_rot, raw_cam_img_size)   

    img_input = resize_image(img_rot, (int(img_rot.shape[0] / 2), int(img_rot.shape[1] / 2))) 
    img_input = zero_pad_image(img_input, (int(raw_cam_img_size[0] / 2), int(raw_cam_img_size[1] / 2)))
    img_input = np.array(img_input, dtype='uint8')  # in the range [0,255]
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.ascontiguousarray(img_input, dtype=np.float32)  

    img_mask = image_valid_mask(img_rot, raw_cam_img_size)
    img_mask = np.array(img_mask, dtype='uint8')  # in the range [0,255]
    img_mask = np.transpose(img_mask, (2, 0, 1))
    img_mask = np.ascontiguousarray(img_mask)        

    img_raw = np.transpose(img_raw, (2, 0, 1))    
    img_rot = np.transpose(img_rot, (2, 0, 1))

    imgs = {
        'in': img_input,
        'raw': img_raw,
        'rot': img_rot,
        'img_mask': img_mask
    }

    return imgs

def preproc_img_rellis(img, gts, raw_cam_img_size):

    img_raw = resize_image(img, raw_cam_img_size)
    img_rot = rotate_image_from_rotation_matrix_numpy(img, gts['rand_init_c'])    
    img_rot = crop_image(img_rot, raw_cam_img_size)   

    img_input = resize_image(img_rot, (int(img_rot.shape[0] / 2), int(img_rot.shape[1] / 2))) 
    img_input = zero_pad_image(img_input, (int(raw_cam_img_size[0] / 2), int(raw_cam_img_size[1] / 2)))
    img_input = np.array(img_input, dtype='uint8')  # in the range [0,255]
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.ascontiguousarray(img_input, dtype=np.float32)  

    img_mask = image_valid_mask(img_rot, raw_cam_img_size)
    img_mask = np.array(img_mask, dtype='uint8')  # in the range [0,255]
    img_mask = np.transpose(img_mask, (2, 0, 1))
    img_mask = np.ascontiguousarray(img_mask)        

    img_raw = np.transpose(img_raw, (2, 0, 1))    
    img_rot = np.transpose(img_rot, (2, 0, 1))

    imgs = {
        'in': img_input,
        'raw': img_raw,
        'rot': img_rot,
        'img_mask': img_mask
    }

    return imgs

def preproc_pcd(pcd, gts, num_points, lidar_line=None, radius=50.):

    def reduce_lidar_line(xyz_intensity, reduce_lidar_line_to):
        OringLines = 64
        velo_down = []
        pt_num = xyz_intensity.shape[0]
        down_Rate = OringLines / reduce_lidar_line_to
        line_num = int(pt_num / OringLines)

        for i in range(64):
            if i % down_Rate == 0:
                for j in range(int(-line_num/2), int(line_num/2)):
                    velo_down.append(xyz_intensity[i*line_num+j])
        data_reduced = np.array(velo_down)
        return data_reduced

    if lidar_line is not None:
        pcd = reduce_lidar_line(pcd, lidar_line)

    if radius is not None:
        logic_x = np.logical_and(pcd[:, 0] >= -radius, pcd[:, 0] < radius)
        logic_y = np.logical_and(pcd[:, 1] >= -radius, pcd[:, 1] < radius)    
        mask = np.logical_and(logic_x, logic_y)
        indices = np.where(mask)[0]
        pcd = pcd[indices]

    if num_points < pcd.shape[0]:
        sampled_indices1 = \
            np.random.choice(range(pcd.shape[0]),
                                size=num_points, replace=False, p=None)
        pcd_ = pcd[sampled_indices1].T
    else:
        pcd_ = np.zeros(shape=(3, num_points))
        pcd_[:3, :pcd.shape[0]] = pcd[:, :3].T

    pc = np.ones((4, pcd_.shape[1]))
    pc[:3, :] = pcd_[:3, :]
    pc = np.array(gts['rand_init_l'] @ pc) 

    return pc

# for Rellis-3D

def get_lidar2cam_mtx(filepath):
    with open(filepath,'r') as f:
        data = yaml.load(f,Loader= yaml.Loader)
    q = data['os1_cloud_node-pylon_camera_node']['q']
    q = np.array([q['x'],q['y'],q['z'],q['w']])
    t = data['os1_cloud_node-pylon_camera_node']['t']
    t = np.array([t['x'],t['y'],t['z']])
    R_vc = Rotation.from_quat(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4,4)
    RT[:3,:3] = R_vc
    RT[:3,-1] = t
    RT = np.linalg.inv(RT)
    return RT

def get_cam_mtx(filepath):
    data = np.loadtxt(filepath)
    P = np.zeros((3,3))
    P[0,0] = data[0]
    P[1,1] = data[1]
    P[2,2] = 1
    P[0,2] = data[2]
    P[1,2] = data[3]
    return P