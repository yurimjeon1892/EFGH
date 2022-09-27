import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image
from math import pi

def image_draw(pcd, img, calib, A, gt, pred, raw_cam_img_size, lidar_fov_rad, cmap=plt.cm.plasma):

    in_pcd = pcd.cpu().detach().numpy()[0]
    in_img = img.cpu().detach().numpy()[0]    
    calib = calib.cpu().detach().numpy()[0]

    net_input_img_size = (int(raw_cam_img_size[0] / 2), int(raw_cam_img_size[1] / 2))
    range_img_size = (int(raw_cam_img_size[0]/ 2), int(raw_cam_img_size[1] * 2))

    in_img = in_img.astype('uint8') 
    in_img = crop_image(in_img, net_input_img_size)

    cam_raw = gt['img_raw'].cpu().detach().numpy()[0].astype('uint8')  
    cam_raw = resize_image(cam_raw, raw_cam_img_size)

    cam_raw_rot = gt['img_rot'].cpu().detach().numpy()[0].astype('uint8')  
    cam_raw_rot = resize_image(cam_raw_rot, raw_cam_img_size)

    in_depth = depth_img_from_cartesian_pc_numpy(in_pcd, calib, raw_cam_img_size) 
    in_depth, valid_mask = minmax_color_img_from_img_numpy(in_depth, cmap=cmap, px=2, valid_mask=True)
    in_depth = depth_img_with_cam_img(in_depth, valid_mask, cam_raw_rot, raw_cam_img_size)

    gt_e_l = gt['e_l'].cpu().detach().numpy()[0] 
    gt_f_l = gt['f_l'].cpu().detach().numpy()[0] 
    gt_g_l = gt['g_l'].cpu().detach().numpy()[0] 
    gt_h_c = gt['h_c'].cpu().detach().numpy()[0] 

    A_ = A.cpu().detach().numpy()[0] 

    gt_intrinsic_sensor2 = gt_h_c
    gt_sensor2_T_sensor1 = gt_g_l @ gt_f_l @ gt_e_l

    gt_cam_T_velo = np.linalg.inv(A_) @ gt_intrinsic_sensor2 @ A_ @ calib @ gt_sensor2_T_sensor1      
    gt_depth = depth_img_from_cartesian_pc_numpy(in_pcd, gt_cam_T_velo, raw_cam_img_size) 
    gt_depth, valid_mask = minmax_color_img_from_img_numpy(gt_depth, cmap=cmap, px=2, valid_mask=True)
    gt_depth = depth_img_with_cam_img(gt_depth, valid_mask, cam_raw_rot, raw_cam_img_size)

    # gt_cam_T_velo = calib @ gt_sensor2_T_sensor1
    # gt_raw = depth_img_from_cartesian_pc_numpy(in_pcd, gt_cam_T_velo, raw_cam_img_size) 
    # gt_raw, valid_mask = minmax_color_img_from_img_numpy(gt_raw, cmap=cmap, px=2, valid_mask=True)
    # gt_raw = depth_img_with_cam_img(gt_raw, valid_mask, cam_raw, raw_cam_img_size)
    # raw = np.concatenate([cam_raw, gt_raw],0)

    gt_img = rotate_image_from_rotation_matrix_numpy(in_img, gt_h_c)  
    gt_img = crop_image(gt_img, net_input_img_size)

    in_range = range_img_from_cartesian_pc_numpy(in_pcd, np.eye(4), range_img_size, lidar_fov_rad)  
    in_range = minmax_color_img_from_img_numpy(in_range, cmap=cmap, px=2)

    gt_range = range_img_from_cartesian_pc_numpy(in_pcd, gt_sensor2_T_sensor1, range_img_size, lidar_fov_rad)  
    gt_range = minmax_color_img_from_img_numpy(gt_range, cmap=cmap, px=2)

    summary_img = {
            # 'raw': raw,  
            }

    # e_range = range_img_from_cartesian_pc_numpy(in_pcd, gt_e_l, range_img_size, lidar_fov_rad)  
    # e_range = minmax_color_img_from_img_numpy(e_range, cmap=cmap, px=2)
    # ef_range = range_img_from_cartesian_pc_numpy(in_pcd, gt_f_l @ gt_e_l, range_img_size, lidar_fov_rad)  
    # ef_range = minmax_color_img_from_img_numpy(ef_range, cmap=cmap, px=2)
    # all_in_one = np.concatenate([in_range, e_range, ef_range, gt_range], 0)
    # summary_img['range_gt']= all_in_one

    # gt_cam_T_velo = np.linalg.inv(A_) @ gt_intrinsic_sensor2 @ A_ @ calib @ gt_e_l   
    # img_depth_EH = depth_img_from_cartesian_pc_numpy(in_pcd, gt_cam_T_velo, raw_cam_img_size)
    # img_depth_EH, valid_mask = minmax_color_img_from_img_numpy(img_depth_EH, cmap=cmap, px=2, valid_mask=True)
    # img_depth_EH = depth_img_with_cam_img(img_depth_EH, valid_mask, cam_raw_rot, raw_cam_img_size)
    # gt_cam_T_velo = np.linalg.inv(A_) @ gt_intrinsic_sensor2 @ A_ @ calib @ gt_f_l @ gt_e_l   
    # img_depth_EFH = depth_img_from_cartesian_pc_numpy(in_pcd, gt_cam_T_velo, raw_cam_img_size)
    # img_depth_EFH, valid_mask = minmax_color_img_from_img_numpy(img_depth_EFH, cmap=cmap, px=2, valid_mask=True)
    # img_depth_EFH = depth_img_with_cam_img(img_depth_EFH, valid_mask, cam_raw_rot, raw_cam_img_size)
    # all_in_one = np.concatenate([in_depth, img_depth_EH, img_depth_EFH, gt_depth], 0)
    # summary_img['depth_gt'] = all_in_one

    if 'E' in pred['network']: 
        
        e_l = pred['e_l'].cpu().detach().numpy()[0]
        img_range_E = range_img_from_cartesian_pc_numpy(in_pcd, e_l, range_img_size, lidar_fov_rad)  
        img_range_E = minmax_color_img_from_img_numpy(img_range_E, cmap=cmap, px=2)
        summary_img['pred_range_E'] = img_range_E  

    if 'E' in pred['network'] and 'H' in pred['network']:

        eh_cam_T_velo = pred['eh_cam_T_velo'].cpu().detach().numpy()[0]   
        img_depth_EH = depth_img_from_cartesian_pc_numpy(in_pcd, eh_cam_T_velo, raw_cam_img_size)
        img_depth_EH, valid_mask = minmax_color_img_from_img_numpy(img_depth_EH, cmap=cmap, px=2, valid_mask=True)
        img_depth_EH = depth_img_with_cam_img(img_depth_EH, valid_mask, cam_raw_rot, raw_cam_img_size)
        summary_img['pred_depth_EH'] = img_depth_EH

    if 'H' in pred['network']: 
        
        h_c = pred['h_c'].cpu().detach().numpy()[0]    
        img_cam_H = rotate_image_from_rotation_matrix_numpy(in_img, h_c)  
        img_cam_H = crop_image(img_cam_H, net_input_img_size)
        cam = np.concatenate([in_img, img_cam_H, gt_img], 0)
        summary_img['cam'] = cam        

    if 'F' in pred['network']: 

        e_l = pred['e_l'].cpu().detach().numpy()[0]     
        f_l = pred['f_l'].cpu().detach().numpy()[0]  
        ef_l = f_l @ e_l
        img_range_EF = range_img_from_cartesian_pc_numpy(in_pcd, ef_l, range_img_size, lidar_fov_rad)  
        img_range_EF = minmax_color_img_from_img_numpy(img_range_EF, cmap=cmap, px=2)
        summary_img['pred_range_EF'] = img_range_EF 

        efh_cam_T_velo = pred['efh_cam_T_velo'].cpu().detach().numpy()[0]         
        img_depth_EFH = depth_img_from_cartesian_pc_numpy(in_pcd, efh_cam_T_velo, raw_cam_img_size)    
        img_depth_EFH, valid_mask = minmax_color_img_from_img_numpy(img_depth_EFH, cmap=cmap, px=2, valid_mask=True)
        img_depth_EFH = depth_img_with_cam_img(img_depth_EFH, valid_mask, cam_raw_rot, raw_cam_img_size)        
        summary_img['pred_depth_EFH'] = img_depth_EFH

        pred_f_score = pred['f_score'].cpu().detach().numpy()[0]  
        gt_f_score = gt['f_score'].cpu().detach().numpy()[0]
        img_pred_f_score = score_image(pred_f_score, range_img_size, cmap=cmap)
        img_gt_f_score = score_image(gt_f_score, range_img_size, cmap=cmap)
        img_f_score = np.concatenate([img_gt_f_score, img_pred_f_score], 0)
        summary_img['score'] = img_f_score        

    if 'G' in pred['network']:

        e_l = pred['e_l'].cpu().detach().numpy()[0]        
        f_l = pred['f_l'].cpu().detach().numpy()[0]
        g_l = pred['g_l'].cpu().detach().numpy()[0]  
        efg_l = g_l @ f_l @ e_l
        img_range_EFG = range_img_from_cartesian_pc_numpy(in_pcd, efg_l, range_img_size, lidar_fov_rad)  
        img_range_EFG = minmax_color_img_from_img_numpy(img_range_EFG, cmap=cmap, px=2)
        summary_img['pred_range_EFG'] = img_range_EFG 

        efgh_cam_T_velo = pred['efgh_cam_T_velo'].cpu().detach().numpy()[0]  
        img_depth_EFGH = depth_img_from_cartesian_pc_numpy(in_pcd, efgh_cam_T_velo, raw_cam_img_size)  
        img_depth_EFGH, valid_mask = minmax_color_img_from_img_numpy(img_depth_EFGH, cmap=cmap, px=2, valid_mask=True)
        img_depth_EFGH = depth_img_with_cam_img(img_depth_EFGH, valid_mask, cam_raw_rot, raw_cam_img_size)         
        summary_img['pred_depth_EFGH'] = img_depth_EFGH       

        dimage = pred['g_depth'].cpu().detach().numpy()[0][0]   
        img_dimage = minmax_color_img_from_img_numpy(dimage, cmap=cmap)
        gt_dimage = gt['g_depth'].cpu().detach().numpy()[0][0]   
        img_gt_dimage = minmax_color_img_from_img_numpy(gt_dimage, cmap=cmap, px=2)
        all_in_one = np.concatenate([img_dimage, img_gt_dimage], 0)
        summary_img['dimage'] = all_in_one  

        g_mask = pred['g_mask'].cpu().detach().numpy()[0][0]   
        img_g_mask = minmax_color_img_from_img_numpy(g_mask, cmap=cmap)
        gt_mask = gt['g_mask'].cpu().detach().numpy()[0][0]   
        img_gt_mask = minmax_color_img_from_img_numpy(gt_mask, cmap=cmap)
        all_in_one = np.concatenate([img_g_mask, img_gt_mask], 0)
        summary_img['mask'] = all_in_one               

    if 'E' in pred['network'] and 'F' in pred['network']:
        if 'G' in pred['network']: 
            all_in_one = np.concatenate([in_range, summary_img['pred_range_E'],summary_img['pred_range_EF'],summary_img['pred_range_EFG'], gt_range],0)
            summary_img['range'] = all_in_one
            del(summary_img['pred_range_E'])
            del(summary_img['pred_range_EF'])
            del(summary_img['pred_range_EFG'])
            all_in_one = np.concatenate([in_depth, summary_img['pred_depth_EH'],summary_img['pred_depth_EFH'],summary_img['pred_depth_EFGH'], gt_depth],0)
            summary_img['depth'] = all_in_one
            del(summary_img['pred_depth_EH'])
            del(summary_img['pred_depth_EFH'])
            del(summary_img['pred_depth_EFGH'])
        else:
            all_in_one = np.concatenate([in_range, summary_img['pred_range_E'],summary_img['pred_range_EF'], gt_range],0)
            summary_img['range'] = all_in_one
            del(summary_img['pred_range_E'])
            del(summary_img['pred_range_EF'])
            all_in_one = np.concatenate([in_depth, summary_img['pred_depth_EH'],summary_img['pred_depth_EFH'], gt_depth],0)
            summary_img['depth'] = all_in_one
            del(summary_img['pred_depth_EH'])
            del(summary_img['pred_depth_EFH'])
    
    return summary_img

def eval_image_draw(pcd, img, calib, A, gt, pred, raw_cam_img_size, lidar_fov_rad, px, cmap=plt.cm.jet):

    in_pcd = pcd.cpu().detach().numpy()[0]
    in_img = img.cpu().detach().numpy()[0]    
    calib = calib.cpu().detach().numpy()[0]

    net_input_img_size = (int(raw_cam_img_size[0] / 2), int(raw_cam_img_size[1] / 2))
    range_img_size = (int(raw_cam_img_size[0]/ 2), int(raw_cam_img_size[1] * 2))

    eval_img = {}

    ###### initial state image ######

    in_img = in_img.astype('uint8') 
    in_img = crop_image(in_img, net_input_img_size)

    cam_raw = gt['img_raw'].cpu().detach().numpy()[0].astype('uint8')  
    cam_raw = resize_image(cam_raw, raw_cam_img_size)

    cam_raw_rot = gt['img_rot'].cpu().detach().numpy()[0].astype('uint8') # input image without resize 
    cam_raw_rot = resize_image(cam_raw_rot, raw_cam_img_size)

    in_depth = depth_img_from_cartesian_pc_numpy(in_pcd, calib, raw_cam_img_size) 
    in_depth, valid_mask = minmax_color_img_from_img_numpy(in_depth, cmap=cmap, px=px, valid_mask=True)
    in_depth = depth_img_with_cam_img(in_depth, valid_mask, cam_raw_rot, raw_cam_img_size)

    # in_range = range_img_from_cartesian_pc_numpy(in_pcd, np.eye(4), range_img_size, lidar_fov_rad)  
    # in_range = minmax_color_img_from_img_numpy(in_range, cmap=cmap, px=2)

    ###### ground truth image ######

    # A_ = A.cpu().detach().numpy()[0] 
    # gt_intrinsic_sensor2 = gt['intrinsic_sensor2'].cpu().detach().numpy()[0] 
    # gt_sensor2_T_sensor1 = gt['sensor2_T_sensor1'].cpu().detach().numpy()[0] 

    # gt_cam_T_velo = np.linalg.inv(A_) @ gt_intrinsic_sensor2 @ A_ @ calib @ gt_sensor2_T_sensor1      
    # gt_depth = depth_img_from_cartesian_pc_numpy(in_pcd, gt_cam_T_velo, raw_cam_img_size) 
    # gt_depth, valid_mask = minmax_color_img_from_img_numpy(gt_depth, cmap=cmap, px=2, valid_mask=True)
    # gt_depth = depth_img_with_cam_img(gt_depth, valid_mask, cam_raw_rot, raw_cam_img_size)

    # gt_cam_T_velo = calib @ gt_sensor2_T_sensor1
    # gt_raw = depth_img_from_cartesian_pc_numpy(in_pcd, gt_cam_T_velo, raw_cam_img_size) 
    # gt_raw, valid_mask = minmax_color_img_from_img_numpy(gt_raw, cmap=cmap, px=2, valid_mask=True)
    # gt_raw = depth_img_with_cam_img(gt_raw, valid_mask, cam_raw, raw_cam_img_size)
    # raw = np.concatenate([cam_raw, gt_raw],0)

    # gt_img = rotate_image_from_rotation_matrix_numpy(in_img, gt_intrinsic_sensor2)  
    # gt_img = crop_image(gt_img, net_input_img_size)    

    # gt_range = range_img_from_cartesian_pc_numpy(in_pcd, gt_sensor2_T_sensor1, range_img_size, lidar_fov_rad)  
    # gt_range = minmax_color_img_from_img_numpy(gt_range, cmap=cmap, px=2)    
        
    ###### RGB rotated image ######

    eh_cam_T_velo = pred['eh_cam_T_velo'].cpu().detach().numpy()[0]   
    img_depth_EH = depth_img_from_cartesian_pc_numpy(in_pcd, eh_cam_T_velo, raw_cam_img_size)
    img_depth_EH, valid_mask = minmax_color_img_from_img_numpy(img_depth_EH, cmap=cmap, px=px, valid_mask=True)
    img_depth_EH = depth_img_with_cam_img(img_depth_EH, valid_mask, cam_raw_rot, raw_cam_img_size)
    eval_img['pred_depth_EH'] = img_depth_EH    

    efh_cam_T_velo = pred['efh_cam_T_velo'].cpu().detach().numpy()[0]         
    img_depth_EFH = depth_img_from_cartesian_pc_numpy(in_pcd, efh_cam_T_velo, raw_cam_img_size)    
    img_depth_EFH, valid_mask = minmax_color_img_from_img_numpy(img_depth_EFH, cmap=cmap, px=px, valid_mask=True)
    img_depth_EFH = depth_img_with_cam_img(img_depth_EFH, valid_mask, cam_raw_rot, raw_cam_img_size)        
    eval_img['pred_depth_EFH'] = img_depth_EFH    
    

    efgh_cam_T_velo = pred['efgh_cam_T_velo'].cpu().detach().numpy()[0]  
    img_depth_EFGH = depth_img_from_cartesian_pc_numpy(in_pcd, efgh_cam_T_velo, raw_cam_img_size)  
    img_depth_EFGH, valid_mask = minmax_color_img_from_img_numpy(img_depth_EFGH, cmap=cmap, px=2, valid_mask=True)
    img_depth_EFGH = depth_img_with_cam_img(img_depth_EFGH, valid_mask, cam_raw_rot, raw_cam_img_size)         
    eval_img['pred_depth_EFGH'] = img_depth_EFGH       

    ###### Horizon network image ######

    # h_c = pred['h_c'].cpu().detach().numpy()[0]    
    # img_cam_H = rotate_image_from_rotation_matrix_numpy(in_img, h_c)  
    # img_cam_H = crop_image(img_cam_H, net_input_img_size)
    # eval_img['pred_img_H'] = img_cam_H 

    ###### range image ######

    # e_l = pred['e_l'].cpu().detach().numpy()[0]        
    # f_l = pred['f_l'].cpu().detach().numpy()[0]
    # g_l = pred['g_l'].cpu().detach().numpy()[0] 

    # img_range_E = range_img_from_cartesian_pc_numpy(in_pcd, e_l, range_img_size, lidar_fov_rad)  
    # img_range_E = minmax_color_img_from_img_numpy(img_range_E, cmap=cmap, px=2)
    # eval_img['pred_range_E'] = img_range_E  

    # ef_l = f_l @ e_l
    # img_range_EF = range_img_from_cartesian_pc_numpy(in_pcd, ef_l, range_img_size, lidar_fov_rad)  
    # img_range_EF = minmax_color_img_from_img_numpy(img_range_EF, cmap=cmap, px=2)
    # eval_img['pred_range_EF'] = img_range_EF 

    # efg_l = g_l @ f_l @ e_l
    # img_range_EFG = range_img_from_cartesian_pc_numpy(in_pcd, efg_l, range_img_size, lidar_fov_rad)  
    # img_range_EFG = minmax_color_img_from_img_numpy(img_range_EFG, cmap=cmap, px=2)
    # eval_img['pred_range_EFG'] = img_range_EFG 

    ###### depth pred ######

    # dimage = pred['g_depth'].cpu().detach().numpy()[0][0]   
    # img_dimage = minmax_color_img_from_img_numpy(dimage, cmap=cmap)
    # eval_img['pred_dimage'] = img_dimage      

    # g_mask = pred['g_mask'].cpu().detach().numpy()[0][0]   
    # img_g_mask = minmax_color_img_from_img_numpy(g_mask, cmap=cmap)
    # eval_img['pred_gmask'] = img_g_mask    

    for k in eval_img.keys():
        h_c = pred['h_c'].cpu().detach().numpy()[0]  
        if "depth" in k:
            eval_img[k] = rotate_image_from_rotation_matrix_numpy(eval_img[k], h_c)  
            eval_img[k] = crop_image(eval_img[k], net_input_img_size)

    return eval_img

def range_img_from_cartesian_pc_numpy(pc, calib, range_img_size, lidar_fov_rad):
    """
    :param pc: point cloud (numpy array, 3 x N)
    :param calib: extrinsic calibration matrix (numpy array, 4 x 4)
    :param range_img_size: range image size (tuple, 2)
    :param lidar_fov_rad: LiDAR FoV in radian / pi (tuple, 2)
    :return img: range image (numpy array, H x W)
    """

    fov_up = lidar_fov_rad[0] * pi
    fov_down = lidar_fov_rad[1] * pi

    pc = pc.copy()
    pc = np.concatenate([pc, np.ones((1, pc.shape[1]))], 0)
    pc = calib @ pc

    pc = pc.copy()
    pc = np.concatenate([pc[:3, :], np.ones((1, pc.shape[1]))], 0)

    r = np.sqrt(np.sum(np.power(pc, 2), 0))
    x = pc[0, :]
    y = pc[1, :]
    z = pc[2, :]  
    pitch = np.arcsin(z / r)
    yaw = np.arctan2(y, x)

    mask = (pitch < fov_up) * (pitch > fov_down)        
    pitch = pitch[mask]
    yaw = yaw[mask]

    u = ((fov_up - pitch) / (fov_up - fov_down)) * (range_img_size[0] - 1)
    v = ((-yaw + pi) / (2 * pi)) * (range_img_size[1] - 1)
    r = r[mask].reshape(-1, 1) # (N, C)

    img = np.zeros((range_img_size[0], range_img_size[1]))
    for i in range(u.shape[-1]):
        img[int(u[i]), int(v[i])] = r[i]
    return img
    
def depth_img_from_cartesian_pc_numpy(pc, cam_T_velo, raw_cam_img_size):  
    """
    :param pc: point cloud (numpy array, 3 x N)
    :param cam_T_velo: extrinsic calibration matrix (numpy array, 3 x 4)
    :param raw_cam_img_size: camera image size (tuple, 2)
    :return depth_img: depth image (numpy array, H x W)
    """
    pc = np.concatenate([pc[:3, :], np.ones((1, pc.shape[1]))], 0)
    pc = cam_T_velo @ pc    
    
    depth_img = np.zeros(shape=raw_cam_img_size)
    for idx, xyw in enumerate(pc.T):        
        x = xyw[0]
        y = xyw[1] 
        w = xyw[2] 
        is_in_img = (
            w > 0 and 0 <= x < w * raw_cam_img_size[1] and 0 <= y < w * raw_cam_img_size[0]
        )        
        if is_in_img:
            depth_img[int(y / w), int(x / w)] = w
    return depth_img.astype('uint8')

def depth_img_with_cam_img(minmax_d_img, valid_mask, cam_img, raw_cam_img_size):
    """
    :param minmax_d_img: minmax colored depth image (numpy array, H x W x 3)
    :param valid mask: 0 for emtpy pixel, 1 for occupied pixel in depth image (numpy array, H x W)
    :param cam_img: camera image (numpy array, H x W x 3)
    :param raw_cam_img_size: raw camera image size (tuple, 2)
    :return minmax_d_img: minmax colored depth image + camera image (numpy array, H x W x 3)
    """
    cam_img = resize_image(cam_img, raw_cam_img_size)  
    cam_img = np.array(cam_img)
    if cam_img.shape[2] != 3:
        cam_img = np.transpose(cam_img, (1, 2, 0))
    for h in range(raw_cam_img_size[0]):
        for w in range(raw_cam_img_size[1]):
            if not valid_mask[h, w]: minmax_d_img[h, w, :] = cam_img[h, w, :]
    return minmax_d_img.astype('uint8')

def minmax_color_img_from_img_numpy(img, cmap, px=2, valid_mask=False):
    """
    :param img: Input image (numpy array, H x W)
    :param cmap: plt color map
    :param px: pixel size (int)
    :param valid_mask: return valida mask? (bool)
    :return img: minmax colored image (numpy array, H x W x 3)
    """
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  
    height, width = img.shape[0], img.shape[1]
    minmax_img = np.zeros(shape=(height, width))
    for y in range(height):
        for x in range(width):
            if img[y, x] > 0:
                y_min, y_max = np.maximum(0, y - px), np.minimum(height - 1, y + px + 1)
                x_min, x_max = np.maximum(0, x - px), np.minimum(width - 1, x + px + 1)
                max_depth = np.max(minmax_img[y_min:y_max, x_min:x_max])
                if max_depth < img[y, x]:
                    minmax_img[y_min:y_max, x_min:x_max] = img[y, x]
    v_mask = (minmax_img != 0).reshape(minmax_img.shape)
    minmax_img = 255 * cmap(minmax_img)[:, :, :3]
    minmax_img = minmax_img.astype('uint8')
    if valid_mask: return minmax_img, v_mask
    else: return minmax_img

def score_image(img, range_img_size, cmap):
    """
    :param img: score image (numpy array, W)
    :param range_img_size: range image size (tuple, 2)
    :return img: minmax colored score images (numpy array, H x W' x 3)
    """
    img = np.tile(np.expand_dims(img, 0), [8, 1]) 
    minmax_img = minmax_color_img_from_img_numpy(img, cmap=cmap)
    minmax_img = Image.fromarray(minmax_img)
    resize_img = minmax_img.resize((range_img_size[1], range_img_size[0]))
    resize_img = np.array(resize_img, dtype='uint8')  # in the range [0,255]
    return resize_img

def save_image(img, fname):
    """
    :param img: image (numpy array, H x W x 3)
    :param fname: file name (string)
    """
    img = np.array(img).astype('uint8')
    if img.ndim == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))
    im = Image.fromarray(img)
    im.save(fname)

def rotate_image_from_rotation_matrix_numpy(img, mat):
    """
    :param img: image (numpy array, H x W x 3)
    :param mat: rotation matrix (numpy array, 3 x 3)
    :return img_rot: rotated image (numpy array, H' x W' x 3)
    """
    img = np.array(img)
    if img.shape[2] != 3:
        img = np.transpose(np.array(img), (1, 2, 0)).astype('uint8')

    rot_deg = math.degrees(np.arctan2(mat[1, 0], mat[0, 0]))

    img_rot = Image.fromarray(img)  
    img_rot = img_rot.rotate(rot_deg, expand=True)
    img_rot = np.array(img_rot)

    if img_rot.shape[2] != 3:
        img_rot = np.transpose(np.array(img_rot), (1, 2, 0))  
    rot_img = img_rot.astype('uint8') 
    return rot_img

def crop_image(img, target_size, init=False): 
    """ 
    :param img: image (numpy array, H x W x 3)
    :param target_size: crop size (tuple, 2)
    :return cropped_img: cropped image (numpy array, H' x W' x 3)
    """
    img = np.array(img)    
    if img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))    
    pad_size_h, pad_size_w = img.shape[0], img.shape[1]
    if pad_size_h < target_size[0]:
        pad_size_h = target_size[0]
    if pad_size_w < target_size[1]:
        pad_size_w = target_size[1]
    img = zero_pad_image(img, (pad_size_h, pad_size_w))
    h = img.shape[0]
    w = img.shape[1]
    i = int(math.floor((h - target_size[0]) / 2.))
    j = int(math.floor((w - target_size[1]) / 2.))
    if init : i, j = 0, 0
    if img.ndim == 3:
        img = img[i:i + target_size[0], j:j + target_size[1], :]
    elif img.ndim == 2:
        img = img[i:i + target_size[0], j:j + target_size[1]]   
    cropped_img = img.astype('uint8') 
    return cropped_img

def resize_image(img, target_size):   
    """ 
    :param img: image (numpy array, H x W x 3)
    :param target_size: target size (tuple, 2)
    :return resized_img: resized image (numpy array, H' x W' x 3)
    """
    img = np.array(img)    
    if img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))
    img_resize = Image.fromarray(img)    
    resized_img = img_resize.resize((target_size[1], target_size[0])) 
    resized_img = np.array(resized_img)
    return resized_img

def zero_pad_image(img, target_size):
    """ 
    :param img: image (numpy array, H x W x 3)
    :param target_size: zero-padded image size (tuple, 2)
    :return padded_img: zero-padded image (numpy array, H' x W' x 3)
    """
    img = np.array(img)    
    if img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))
    h = img.shape[0]
    w = img.shape[1]        
    i = int(math.floor((target_size[0] - h) / 2.))
    j = int(math.floor((target_size[1] - w) / 2.))
    padded_img = np.zeros((target_size[0], target_size[1], 3))
    padded_img[i:i + h, j:j + w, :] = img
    return padded_img.astype('uint8') 

def image_valid_mask(img, target_size):
    """ 
    :param img: image (numpy array, H x W x 3)
    :param target_size: valid mask size (tuple, 2)
    :return valid_mask: valid_mask (numpy array, H' x W' x 1)
    """
    img = np.array(img)    
    if img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))    
    valid_mask = np.ones((target_size[0], target_size[1], 1))
    zero_mask = (img[:, :, 0] == 0) * (img[:, :, 1] == 0) * (img[:, :, 2] == 0)
    valid_mask[zero_mask] = 0
    return valid_mask.astype('uint8') 

def rpy_to_matrix(roll, pitch, yaw):
    """ 
    :param roll, pitch, yaw: roll, pitch, yaw values in radian (float)
    :return R4: Rotation marix (numpy array, 4 x 4)
    """

    yawMatrix = np.array([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.array([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.array([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix @ pitchMatrix @ rollMatrix
    R4 = np.eye(4)
    R4[:3, :3] = R

    return R4

def xyz_to_matrix (tx, ty, tz):
    """ 
    :param tx, ty, tz: tx, ty, tz values (float)
    :return t: Translation marix (numpy array, 4 x 4)
    """
    t = np.array([[1, 0, 0, tx], 
                  [0, 1, 0, ty], 
                  [0, 0, 1, tz],
                  [0, 0, 0, 1]])
    return t   

def rotation_matrix_from_two_vectors_numpy(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1 = vec1 / np.sqrt(np.sum(np.power(vec1, 2)))
    v = np.cross(vec1, vec2)
    c = np.dot(vec1, vec2)
    s = np.sqrt(np.sum(np.power(v, 2))) 
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if (1 - c) == 0: return np.eye(3)
    if (1 + c) == 0: return -np.eye(3)
    rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

import open3d as o3d
import copy

def draw_registration_result(source_, target_):
    print(source_.shape, target_.shape)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.transpose(source_))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(np.transpose(target_))

    source_temp = copy.deepcopy(source_pcd)
    target_temp = copy.deepcopy(target_pcd)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def draw_single_registration_result(source_):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.transpose(source_))

    source_temp = copy.deepcopy(source_pcd)
    source_temp.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([source_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])