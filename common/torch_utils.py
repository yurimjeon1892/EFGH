import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import pi, sin, cos
from scipy.spatial.transform import Rotation 

from common.numpy_utils import crop_image

def range_img_from_cartesian_pc_torch(pc, range_img_size, lidar_fov_rad, device):
    """
    :param pc: point clouds (numpy array, B x 3 x N)
    :param range_img_size: range image size (tuple, 2)
    :param lidar_fov_rad: LiDAR FoV in radian / pi (tuple, 2)
    :param device: device (string)
    :return imgs: range images (torch tensor, B x 4 x range_img_size)
    """
    fov_up = lidar_fov_rad[0] * pi
    fov_down = lidar_fov_rad[1] * pi

    xyz = pc.clone().float()    
    batch_size = pc.size(0)
    
    x_ = xyz[:, 0, :]
    y_ = xyz[:, 1, :]
    z_ = xyz[:, 2, :]  

    r_ = torch.sqrt(torch.sum(torch.pow(xyz, 2), 1))
    pitch_ = torch.asin(z_ / r_)
    yaw_ = torch.atan2(y_, x_)

    imgs = []
    for b in range(batch_size):

        x_b, y_b, z_b, r_b, pitch_b, yaw_b = x_[b, :], y_[b, :], z_[b, :], r_[b, :], pitch_[b, :], yaw_[b, :]

        mask = (pitch_b < fov_up) * (pitch_b > fov_down)  
        x_b = x_b[mask].view(1, -1) 
        y_b = y_b[mask].view(1, -1) 
        z_b = z_b[mask].view(1, -1) 
        r_b = r_b[mask].view(1, -1)
        pitch_b = pitch_b[mask].view(1, -1)
        yaw_b = yaw_b[mask].view(1, -1)
        values = torch.cat([x_b, y_b, z_b, r_b], 0)

        # normalizing and scaling
        u = ((fov_up - pitch_b) / (fov_up - fov_down)) * (range_img_size[0] - 1)
        v = ((-yaw_b + pi) / (2 * pi)) * (range_img_size[1] - 1)
        indices = torch.cuda.LongTensor(torch.cat([u.long(), v.long()], 0)) # (2, N) 
        values = torch.cuda.FloatTensor(values).t() # (N, C)
        img = torch.zeros((range_img_size[0], range_img_size[1], values.size(-1))).to(device)
        img[indices[0].tolist(), indices[1].tolist()] = values
        img = img.permute(2, 0, 1).unsqueeze(0)   
        imgs.append(img)

    imgs = torch.cat(imgs, 0) 

    return imgs

def depth_img_from_cartesian_pc_torch(pc, cam_T_velo, cam_img_size, device):
    """
    :param pc: point cloud (torch tensor, B x 3 x N)
    :param cam_T_velo: extrinsic calibration matrix from cam to lidar (torch tensor, B x 3 x 4)
    :param cam_img_size: camera image size (tuple, 2)
    :param device: device (string)
    :return imgs: depth images (torch tensor, B x 4 x cam_img_size)
    """
    pc = pc.to(device).float()
    cam_T_velo = cam_T_velo.to(device).float()
    imgs = []
    for b in range(pc.size(0)):

        pc_b = torch.cat([pc[b, :3, :], torch.ones((1, pc.size(-1))).to(device).float()], 0)
        xyw_b = torch.mm(cam_T_velo[b], pc_b)

        w = xyw_b[2, :] 
        x = xyw_b[0, :] / w
        y = xyw_b[1, :] / w

        mask = (x < cam_img_size[1]) * (x > 0) * (y < cam_img_size[0]) * (y > 0) * (w > 0)
        x = x[mask].view(1, -1)   
        y = y[mask].view(1, -1)        
        w = w[mask].view(-1)

        px, py, pz = pc_b[0, :], pc_b[1, :], pc_b[2, :]
        px = px[mask].view(-1)   
        py = py[mask].view(-1)        
        pz = pz[mask].view(-1)

        indices = torch.cat([y, x], 0).to(device).long() # (2, N)
        depth = torch.zeros((cam_img_size[0], cam_img_size[1], 4)).to(device)    
        depth[indices[0].tolist(), indices[1].tolist(), 0] = px
        depth[indices[0].tolist(), indices[1].tolist(), 1] = py
        depth[indices[0].tolist(), indices[1].tolist(), 2] = pz
        depth[indices[0].tolist(), indices[1].tolist(), 3] = w
        depth = depth.permute(2, 0, 1).unsqueeze(0)    

        imgs.append(depth)

    imgs= torch.cat(imgs, 0)

    return imgs

def normal_vector_2d_from_abs_sign_torch(abs, sign, device):
    """
    :param abs: normal vector absolute value in xy order (torch tensor, B x 2 x 1)
    :param sign: one-hot encoded normal vector sign value for xy (torch tensor, B x 4)
    :param device: device (string)
    :return norms: 2d normal vector (torch tensor, B x 2 x 1)
    """
    softmax = nn.Softmax(dim=0).cuda()
    norms = []
    for b in range(abs.size(0)):
        abs_b, sign_b = abs[b, :, 0], sign[b, :]  
        sign_b = torch.argmax(softmax(sign_b)).data.tolist()
        sign_b, y_sign = divmod(sign_b, 2)
        sign_b, x_sign = divmod(sign_b, 2) 
        sgn = torch.tensor([x_sign, y_sign])
        sgn = torch.where(sgn == 0, -torch.ones_like(sgn), sgn).to(device) 
        norm = torch.unsqueeze(abs_b * sgn, 0)
        norms.append(norm)
    norms = torch.unsqueeze(torch.cat(norms, 0), -1) 
    return norms

def normal_vector_3d_from_abs_sign_torch(abs, sign, device):
    """
    :param abs: normal vector absolute value in xyz order (torch tensor, B x 3 x 1)
    :param sign: one-hot encoded normal vector sign value for xyz (torch tensor, B x 8)
    :param device: device (string)
    :return norms: 3d normal vector (torch tensor, B x 3 x 1)
    """
    softmax = nn.Softmax(dim=0).cuda()
    norms = []
    for b in range(abs.size(0)):
        abs_b, sign_b = abs[b, :, 0], sign[b, :]  
        sign_b = torch.argmax(softmax(sign_b)).data.tolist()
        sign_b, z_sign = divmod(sign_b, 2)
        sign_b, y_sign = divmod(sign_b, 2)
        sign_b, x_sign = divmod(sign_b, 2) 
        sgn = torch.tensor([x_sign, y_sign, z_sign])
        sgn = torch.where(sgn == 0, -torch.ones_like(sgn), sgn).to(device) 
        norm = torch.unsqueeze(abs_b * sgn, 0)
        norms.append(norm)
    norms = torch.unsqueeze(torch.cat(norms, 0), -1) 
    return norms

def quaternion_from_abs_sign_torch(abs, sign, device):
    """
    :param abs: quaternion absolute value in xyzw order (torch tensor, B x 4 x 1)
    :param sign: one-hot encoded quaternion sign value for xyz (torch tensor, B x 8)
    :param device: device (string)
    :return quats: quaternion (torch tensor, B x 4 x 1)
    """
    softmax = nn.Softmax(dim=0).cuda()
    quats = []
    for b in range(abs.size(0)):
        abs_b, sign_b = abs[b, :, 0], sign[b, :]    
        sign_b = torch.argmax(softmax(sign_b)).data.tolist()
        sign_b, z_sign = divmod(sign_b, 2)
        sign_b, y_sign = divmod(sign_b, 2)
        sign_b, x_sign = divmod(sign_b, 2)
        sgn = torch.tensor([x_sign, y_sign, z_sign, 1])
        sgn = torch.where(sgn == 0, -torch.ones_like(sgn), sgn).to(device) 
        quat = torch.unsqueeze(abs_b * sgn, 0)
        quats.append(quat)
    quats = torch.unsqueeze(torch.cat(quats, 0), -1)    
    return quats

def rotation_matrix_between_two_vectors_torch(srce, dest, device):
    """
    :param vec1: Input source vector (torch tensor, B x 3 x 1)
    :param vec2: Input destination vector (torch tensor, B x 3 x 1)
    :param device: device (string)
    :return quats: Output rotation matrix which when applied to vec1, aligns it with vec2. (torch tensor, B x 4 x 4)
    """
    rotation_matrices = []
    for b in range(srce.size(0)):
        vec1_b = srce[b, :, 0].to(device) 
        vec2_b = dest[b, :, 0].to(device)               
        v = torch.cross(vec1_b, vec2_b)
        c = torch.dot(vec1_b, vec2_b)
        s = torch.sqrt(torch.sum(torch.pow(v, 2))) 
        kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]).to(device)
        if (1 - c) == 0: 
            rotation_matrix = torch.eye(4)
        elif (1 + c) == 0: 
            rotation_matrix = -torch.eye(4)
            if vec1_b[0].item() == 0.0 and vec2_b[0].item() == 0.0:
                rotation_matrix[0, 0] = 1
            elif vec1_b[2].item() == 0.0 and vec2_b[2].item() == 0.0:
                rotation_matrix[2, 2] = 1        
        else:
            rot3 = torch.eye(3).to(device) + kmat + torch.mm(kmat, kmat) * ((1 - c) / (s ** 2))
            rotation_matrix = torch.eye(4).to(device)
            rotation_matrix[:3, :3] = rot3
        rotation_matrix = torch.unsqueeze(rotation_matrix, 0)        
        rotation_matrices.append(rotation_matrix)
    rotation_matrices = torch.cat(rotation_matrices, 0).to(device)
    return rotation_matrices

def rotation_matrix_from_quaternion_torch(quat_xyzw, device):
    """
    :param quat_xyzw: quaternion in xyzw order (torch tensor, B x 4 x 1)
    :param device: device (string)
    :return rotation_matrices: rotation matrix (torch tensor, B x 4 x 4)
    """
    quat_xyzw = quat_xyzw.detach().cpu().numpy()
    rotation_matrices = []
    for b in range(quat_xyzw.shape[0]):
        quat_b = quat_xyzw[b, :, 0]
        r = Rotation.from_quat(quat_b) 
        rotation_matrix_4x4 = np.eye(4)
        rotation_matrix_4x4[:3, :3] = r.as_matrix()
        rotation_matrix_4x4 = torch.unsqueeze(torch.tensor(rotation_matrix_4x4), 0)
        rotation_matrices.append(rotation_matrix_4x4)
    rotation_matrices = torch.cat(rotation_matrices, 0).to(device)
    return rotation_matrices

def translation_matrix_from_vector_torch(vec, device):
    """
    :param vec: translation vector (torch tensor, B x 3 x 1)
    :param device: device (string)
    :return translation_matrices: translation matrix (torch tensor, B x 4 x 4)
    """
    translation_matrices = []
    for b in range(vec.size(0)):
        vec_ = vec[b, :]
        t = torch.tensor([[1, 0, 0, vec_[0]], [0, 1, 0, vec_[1]], [0, 0, 1, vec_[2]], [0, 0, 0, 1]])
        t = torch.unsqueeze(t, 0)
        translation_matrices.append(t)
    translation_matrices = torch.cat(translation_matrices, 0).to(device)
    return translation_matrices

def rotate_image_from_rotation_matrix_torch(img, mat, device):
    """
    :param img: image (torch tensor, B x 3 x H x W)
    :param mat: rotation matrix (torch tensor, B x 3 x 3)
    :param device: device (string)
    :return rot_images: rotated images (torch tensor, B x 3 x H x W)
    """
    from PIL import Image
    rot_images = []
    for b in range(img.size(0)):
        rot_deg = torch.rad2deg(torch.atan2(mat[b, 1, 0], mat[b, 0, 0]))
        img_b = img.cpu().detach().numpy()[b]
        oh, ow = img_b.shape[1], img_b.shape[2]
        img_b = np.transpose(img_b, (1, 2, 0))
        img_rot = Image.fromarray(np.array(img_b, dtype='uint8'))        
        img_rot = img_rot.rotate(rot_deg)      
        img_crop = crop_image(img_rot, (oh, ow)) 
        rot_images.append(torch.unsqueeze(torch.tensor(img_crop).permute((2, 0, 1)), 0))    
    rot_images = torch.cat(rot_images, 0).to(device).float() 
    return rot_images

def compute_cam_T_velo_matrix(c_T, l_T, calib, A, device):
    """
    :param c_T: camera transform matrix (torch tensor, B x 3 x 3)
    :param l_T: LiDAR transform matrix (torch tensor, B x 4 x 4)
    :param calib: initital extrinsic calibration matrix (torch tensor, B x 3 x 4)
    :param A: camera related matrix for rotation (torch tensor, B x 3 x 3)
    :param device: device (string)
    :return mat1: cam_T_velo matrix (torch tensor, B x 3 x 4)
    """
    mat1 = torch.bmm(calib, l_T)
    mat1 = torch.bmm(A, mat1)
    mat1 = torch.bmm(c_T, mat1)
    mat1 = torch.bmm(torch.inverse(A), mat1)
    return mat1

def circular_assign_torch(feat, offset, device):
    """
    :param feat: feature map (torch tensor, B x C x H x W)
    :param offset: offset value (int)
    :param device: device (string)
    :return mat1: feature map (torch tensor, B x C x H x W')
    """    
    right_end = feat[:, :, :, :offset]
    left_end = feat[:, :, :, -offset:]
    left_end_flip = torch.zeros(left_end.size())
    for i in range(offset):
        left_end_flip[:, :, :, i] = left_end[:,:, :, offset - 1 - i]
    feat_circular_assign = torch.cat([left_end_flip.to(device), feat, right_end.to(device)], -1)
    return feat_circular_assign

def vector_from_radian_torch(rad, device):
    """
    :param rad: radian (torch tensor, B)
    :param device: device (string)
    :return vecs: vector (torch tensor, B x 3 x 1)
    """  
    vecs = []
    for b in range(rad.size(0)):
        vecs.append(torch.unsqueeze(torch.unsqueeze(torch.tensor([cos(rad[b]), sin(rad[b]), 0.]), 0), -1)) 
    vecs = torch.cat(vecs, 0).to(device)
    return vecs

def matrix_3x3_to_4x4(mat, device):
    """
    :param mat: matrix (torch tensor, B x 3 x 3)
    :param device: device (string)
    :return mat4: matrix (torch tensor, B x 4 x 4)
    """ 
    mat4 = torch.zeros((mat.size(0), 4, 4))
    mat4[:, :3, :3] = mat
    mat4[:, 3, 3] = 1
    return mat4.to(device)

def concat_tensors(t1, t2):
    """
    :param t1: tensor 1 (torch tensor, B x C x H x W)
    :param t2: tensor 2 (torch tensor, B x C' x H' x W')
    :return tcat: concate tensor (torch tensor, B x (C+C') x H x W)
    """ 
    if t2.size(2) != t1.size(2):        
        p1 = int((t2.size(2) - t1.size(2)) / 2)
        t2 = t2[:, :, p1:p1 + t1.size(2), :]
    tcat = torch.cat((t1, t2), 1)
    return tcat