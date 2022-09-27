import torch.optim
import torch.utils.data
from tqdm import tqdm
import numpy as np
import os
import sys
import gc
import traceback

from common.numpy_utils import eval_image_draw, save_image
from common.helper import Err

def test_odom(loader, model, args):

    err = Err(args['dataset'])

    ckpt_name = args['ckpt_path'].split('/')[-2]
    rand_init_name = args['rand_init'].split('/')[-1]
    save_dir = os.path.join('../../test/preds', ckpt_name)      
    os.makedirs(save_dir, mode=0o777, exist_ok=True) 
    print('Save directory: ', save_dir)

    pred_name = rand_init_name.replace("rand_init", "pred")
    pred_dir = os.path.join(save_dir, pred_name)  
    print("Prediction csv: ", pred_dir)
    f = open(pred_dir, 'w')
    f.close()

    model.eval()
    with torch.no_grad():
        description = '[i] Test '
        for i, (pcd, img, calib, A, gt, fname) in \
                enumerate(tqdm(loader, desc=description, unit='batches')):
            try:
                # Convert data type
                pcd = pcd.to(args['DEVICE']).float()
                img = img.to(args['DEVICE']).float()
                calib = calib.to(args['DEVICE']).float()
                A = A.to(args['DEVICE']).float()

                # run model
                pred = model(pcd, img, calib, A)    

                err.update(gt, pred)
                #####################################  
                pred_sensor2_T_sensor1 = pred["sensor2_T_sensor1"].detach().cpu().numpy()[0, :3, :].flatten()

                f = open(pred_dir, 'a')
                f.write(fname[0] + ',')
                for i in range(pred_sensor2_T_sensor1.shape[0]):
                    f.write(str(pred_sensor2_T_sensor1[i]) + ',')
                f.write('\n')
                f.close()
                #####################################     
                
                if args['save_image'] == True:
                    imgs = eval_image_draw(pcd, img, calib, A, gt, pred, args['raw_cam_img_size'], args['lidar_fov_rad'])  
                    # for k in imgs.keys():
                    for k in ['raw']:
                        image_file_name = os.path.join(save_dir, fname[0] + '_' + k + '.png')
                        save_image(imgs[k], image_file_name)                    

                # del pcd, img, gt
                # torch.cuda.empty_cache()

            except RuntimeError as ex:
                print("in VAL, RuntimeError " + repr(ex))
                # traceback.print_tb(ex.__traceback__, file=logger.out_fd)
                traceback.print_tb(ex.__traceback__)

                if "CUDA out of memory" in str(ex) or "cuda runtime error" in str(ex):
                    print("out of memory, continue")
                    del pcd, img, gt
                    torch.cuda.empty_cache()
                    gc.collect()
                    print('remained objects after OOM crash')
                else:
                    sys.exit(1)
    
    print('Error; ', end=" ")
    for k in list(err.dict.keys()):
        print(k + ' {:.4f}'.format(err.dict[k]), end=" ")
    print()
    print('[i] Test finished.')    
    return 

def test_kitti_raw(loader, model, args):

    err = Err(args['dataset'])

    T_cam0unrect_velo = np.array(
    [[ 7.027555e-03, -9.999753e-01,  2.599616e-05, -7.137748e-03],
    [-2.254837e-03, -4.184312e-05, -9.999975e-01, -7.482656e-02],
    [ 9.999728e-01,  7.027479e-03, -2.255075e-03, -3.336324e-01],
    [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
    R_rect_00 = np.array(
    [[ 0.999928   , 0.00808599, -0.0088668 ,  0.        ],
    [-0.0081232  , 0.9999583 , -0.00416975,  0.        ],
    [ 0.00883271 , 0.00424148,  0.999952  ,  0.        ],
    [ 0.         , 0.        ,  0.        ,  1.        ]])

    ckpt_name = args['ckpt_path'].split('/')[-2]
    rand_init_name = args['rand_init'].split('/')[-1][20:-4]
    save_dir = os.path.join('../../test/preds', ckpt_name)            
    os.makedirs(save_dir, mode=0o777, exist_ok=True) 
    print('Save directory: ', save_dir)

    pred_dir = os.path.join(save_dir, 'kitti_raw_pred_' + rand_init_name + '.csv')  
    print("Prediction csv: ", pred_dir)
    f = open(pred_dir, 'w')
    f.close()

    model.eval()
    with torch.no_grad():
        description = '[i] Test '
        for i, (pcd, img, calib, A, gt, fname) in \
                enumerate(tqdm(loader, desc=description, unit='batches')):
            try:
                # Convert data type
                pcd = pcd.to(args['DEVICE']).float()
                img = img.to(args['DEVICE']).float()
                calib = calib.to(args['DEVICE']).float()
                A = A.to(args['DEVICE']).float()           

                # run model
                pred = model(pcd, img, calib, A)     

                err.update(gt, pred)
                #####################################
                pred_sensor2_T_sensor1 = pred['sensor2_T_sensor1'].cpu().detach().numpy()[0]
                pred_sensor2_T_sensor1 = R_rect_00 @ T_cam0unrect_velo @ pred_sensor2_T_sensor1
                pred_sensor2_T_sensor1 = pred_sensor2_T_sensor1[:3, :].flatten()

                fn_key = fname[0].split("/")[-1]
                f = open(pred_dir, 'a')
                f.write(fn_key + ',')
                for i in range(pred_sensor2_T_sensor1.shape[0]):
                    f.write(str(pred_sensor2_T_sensor1[i]) + ',')
                f.write('\n')
                f.close()
                #####################################
                
                if args['save_image'] == True:
                    imgs = eval_image_draw(pcd, img, calib, A, gt, pred, args['raw_cam_img_size'], args['lidar_fov_rad'])  
                    fns = fname[0].split('/')
                    for k in imgs.keys():
                        fname = os.path.join(save_dir, fns[1] + '_' + fns[2] + '_' + k + '.png')
                        save_image(imgs[k], fname)                       

                # del pcd, img, gt
                # torch.cuda.empty_cache()

            except RuntimeError as ex:
                print("in VAL, RuntimeError " + repr(ex))
                # traceback.print_tb(ex.__traceback__, file=logger.out_fd)
                traceback.print_tb(ex.__traceback__)

                if "CUDA out of memory" in str(ex) or "cuda runtime error" in str(ex):
                    print("out of memory, continue")
                    del pcd, img, gt
                    torch.cuda.empty_cache()
                    gc.collect()
                    print('remained objects after OOM crash')
                else:
                    sys.exit(1)
    f.close()
    print('[i] Test finished.')    
    return 