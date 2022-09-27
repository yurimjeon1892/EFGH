# helper functions for training
import os, sys
import shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from .numpy_utils import image_draw

def update_summary(summary, mode, iter, losses, errors, pcd, img, calib, A, gt, pred, raw_cam_img_size, lidar_fov_rad):

    for k in list(losses.keys()):
        summary.add_scalar(mode + '_loss/' + k, losses[k].avg, iter) 

    for k in list(errors.keys()):
        summary.add_scalar(mode + '_error/' + k, errors[k], iter)   

    summary_img = image_draw(pcd, img, calib, A, gt, pred, raw_cam_img_size, lidar_fov_rad)  
    for k in list(summary_img.keys()):
        summary_img_k = summary_img[k]
        if summary_img_k.shape[2] == 3:
            summary_img_k = np.transpose(summary_img[k], (2, 0, 1))
        summary.add_image(mode + '_image/' + k, summary_img_k, iter)    

    return

def adjust_learning_rate(lr_init, optimizer, iter):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.7**(iter // 50000))
    for param_group in optimizer.param_groups:        
        if lr < param_group["lr"]:
            print("[>] old lr: ", param_group["lr"], end=" --> ")
            param_group["lr"] = lr
            print("new lr:", param_group["lr"])
        else:
            lr = param_group["lr"]
    return lr

def save_checkpoint(state, is_best, ckpt_dir,
                    filename='checkpoint.pth.tar',
                    iter_iterval=1000):
    torch.save(state, os.path.join(ckpt_dir, filename))
    if state['iter'] % iter_iterval == 0:
        shutil.copyfile(
            os.path.join(ckpt_dir, filename),
            os.path.join(ckpt_dir, 'checkpoint_' + str(state['iter'])+'.pth.tar'))
        print(state['iter'], 'iter checkpoint saved')

    if is_best:
        shutil.copyfile(
            os.path.join(ckpt_dir, filename),
            os.path.join(ckpt_dir, 'model_best.pth.tar'))

    if state['iter'] > 5 * iter_iterval:
        prev_checkpoint_filename = os.path.join(
            ckpt_dir, 'checkpoint_' + str(state['iter'] - 5 * iter_iterval) + '.pth.tar')
        # print('prev_checkpoint_filename', prev_checkpoint_filename)
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)
    print("[i] Saved ckpt >>> ", os.path.join(ckpt_dir, filename))

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class Lss():
    def __init__(self, keys):
        self.keys = keys
        self.dict = {}
        self.flush()  

    def flush(self):
        for k in self.keys:
            self.dict[k] = AverageMeter()      

    def update(self, losses, batch):
        for k in list(losses.keys()):
            self.dict[k].update(losses[k].item(), batch) 
        return 
    
class Err():
    def __init__(self, dataset):
        self.dataset = dataset
        self.error_dict = {}    
        self.dict = {}

    def flush(self, keys=None):
        if keys != None:
            for k in list(keys):
                self.error_dict[k] = []
        else:
            for k in list(self.error_dict.keys()):
                self.error_dict[k] = []

    def update(self, gt, pred):

        gt_sensor2_T_sensor1 = gt['sensor2_T_sensor1'].cpu().detach().numpy()[0]
        pred_sensor2_T_sensor1 = pred['sensor2_T_sensor1'].cpu().detach().numpy()[0]

        if self.dataset == "KITTI_RAW": # error metric for camera-LiDAR extrinsic calibration.            
            rot_error, trs_error = self.calc_error_raw_np(gt_sensor2_T_sensor1, pred_sensor2_T_sensor1)
        else: # error metric for image based localization.
            rot_error, trs_error = self.calc_error_odom_np(gt_sensor2_T_sensor1, pred_sensor2_T_sensor1)

        errors = {
            'rot': rot_error,
            'trs': trs_error 
        } 

        if not self.error_dict: self.flush(errors.keys())        
        
        for k in list(errors.keys()):
            self.error_dict[k].append(errors[k]) 
            self.dict[k + '_mean'] = np.mean(self.error_dict[k])
            self.dict[k + '_std'] = np.std(self.error_dict[k]) 
              
        return    

    def calc_error_raw_np(self, gt, pred):

        gt_rot = R.from_matrix(gt[:3, :3])
        pred_rot = R.from_matrix(pred[:3, :3])

        gt_rot_as_quat = gt_rot.as_quat()        
        pred_rot_as_quat = pred_rot.as_quat()    
        rot_error = self.quaternion_distance(gt_rot_as_quat, pred_rot_as_quat)
        
        gt_trs = gt[:3, 3]
        pred_trs = pred[:3, 3]
        trs_error = np.mean(np.fabs(gt_trs - pred_trs))
        
        return rot_error, trs_error       

    def quaternion_distance(self, q1, q2):
        """
        Batch quaternion distances, used as loss
        Args:
            q (torch.Tensor): shape=[4]
            r (torch.Tensor): shape=[4]
        Returns:
            torch.Tensor: shape=[1]
        """
        q1 = Quaternion(q1[3], q1[0], q1[1], q1[2])
        q2 = Quaternion(q2[3], q2[0], q2[1], q2[2])
        t = q1 * q2.inverse
        tt = np.array([t[0], t[1], t[2], t[3]])
        tt_norm = np.linalg.norm(tt[1:])
        tt_abs = np.abs(tt[0])
        tt_ret = 2 * (np.arctan2(tt_norm, tt_abs)) * (180 / np.pi)
        return tt_ret

    def calc_error_odom_np(self, gt, pred):
        gt_R, gt_t = gt[:3, :3], gt[:3, 3]
        pred_R, pred_t = pred[:3, :3], pred[:3, 3]
        tmp = (np.trace(pred_R.transpose().dot(gt_R))-1)/2
        tmp = np.clip(tmp, -1.0, 1.0)
        L_rot = np.arccos(tmp)
        L_rot = 180 * L_rot / np.pi
        L_trans = np.linalg.norm(pred_t - gt_t)
        return L_rot, L_trans