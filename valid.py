import sys
import gc
import traceback

import torch.nn.parallel
import torch.optim
import torch.utils.data

from tqdm import tqdm
from common.helper import update_summary, Lss, Err


def validate(loader, model, criterion, args, summary, it):
    
    lss = Lss(criterion.loss_name)
    err = Err(args['dataset'])

    model.eval()
    with torch.no_grad():
        description = '[i] Valid iter {}'.format(it)
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

                # compute loss
                losses, gt = criterion.compute_loss(pcd, img, calib, A, gt, pred)

                lss.update(losses, pcd.size(0))
                err.update(gt, pred)

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

    update_summary(summary, 'valid', it, lss.dict, err.dict, pcd, img, calib, A, gt, pred, args['raw_cam_img_size'], args['lidar_fov_rad'])

    print('[i] Valid iter {}; '.format(it))
    print('Loss; ', end=" ")
    for k in list(lss.keys):
        print(k + ' {:.2f}'.format(lss.dict[k].avg), end=" ")
    print()
    print('Error; ', end=" ")
    for k in list(err.dict.keys()):
        print(k + ' {:.4f}'.format(err.dict[k]), end=" ")
    print()

    return lss.dict