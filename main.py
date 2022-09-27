import os, sys
import gc

import torch.nn.parallel
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter
import numpy as np

import shutil

import data_loader
import nets
import losses

from common.helper import query_yes_no
from iterater import iterater
from test import test_odom, test_kitti_raw

import yaml

def main():

    # ensure numba JIT is on
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    # parse arguments
    global args
    with open(sys.argv[1], 'r') as stream:
        args = yaml.safe_load(stream)

    cuda = torch.cuda.is_available()
    if cuda :
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("=> using '{}' for computation.".format(device))

    # -------------------- logging args --------------------
    print("=> checking ckpt dir...")
    if args['test'] is False and os.path.exists(args['ckpt_dir']):
        if args['resume_path'] is False:
            to_continue = query_yes_no(
                '=> Attention!!! ckpt_dir {' + args['ckpt_dir'] + '} already exists!\n' 
                + '=> Whether to continue?',
                default=None)
            if to_continue:
                for root, dirs, files in os.walk(args['ckpt_dir'], topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
            else:
                sys.exit(1)
        elif args['pretrained_path'] is not False:
            to_continue = query_yes_no(
                '=> Attention!!! ckpt_dir {' + args['ckpt_dir'] +
                '} already exists! Whether to continue?',
                default=None)
            if to_continue:
                for root, dirs, files in os.walk(args['ckpt_dir'], topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
            else:
                sys.exit(1)
    if args['test'] is False :
        os.makedirs(args['ckpt_dir'], mode=0o777, exist_ok=True)
        shutil.copyfile(sys.argv[1], os.path.join(args['ckpt_dir'], 'config.yaml'))
        summary = SummaryWriter(args['ckpt_dir'])        

    # -------------------- dataset & loader --------------------
    loader = {}
    if args['test'] is False :
        train_dataset = data_loader.__dict__[args['dataset']](
            mode='train',
            args=args
        )
        # print('train_dataset: ' + str(train_dataset))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=args['workers'],
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )
        loader['train'] = train_loader

        val_dataset = data_loader.__dict__[args['dataset']](
            mode='valid',
            args=args
        )
        # print('val_dataset: ' + str(val_dataset))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=args['workers'],
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )
        loader['validate'] = val_loader

    else:
        test_dataset = data_loader.__dict__[args['dataset']](
            mode=args['test'],
            args=args
        )
        # print('val_dataset: ' + str(val_dataset))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=args['workers'],
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )    
    # -------------------- create model --------------------
    print("=> creating model and optimizer... ")
    model = nets.__dict__[args['arch'] + 'Backbone'](args).to(device)
    model = torch.nn.DataParallel(model)
    if args['test'] is False :
        criterion = losses.__dict__[args['arch'] + 'Criterion'](args)

    # -------------------- resume --------------------
    if args['test']:
        if os.path.isfile(args['ckpt_path']):
            print("=> loading checkpoint '{}'".format(args['ckpt_path']))
            checkpoint = torch.load(args['ckpt_path'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("=> completed.")
            print("=> start iter {}, min loss {}"
                  .format(checkpoint['iter'], checkpoint['min_loss']))         
        else:
            print("=> no checkpoint found at '{}'".format(args['ckpt_path']))
            return
        if args['dataset'] == 'KITTI_ODOM': test_odom(test_loader, model, args)        
        elif args['dataset'] == 'KITTI_RAW': test_kitti_raw(test_loader, model, args)
        elif args['dataset'] == 'NUSC': test_odom(test_loader, model, args)
        elif args['dataset'] == 'RELLIS_3D': test_odom(test_loader, model, args)
        return

    elif args['resume_path']:
        if os.path.isfile(args['resume_path']):
            print("=> loading checkpoint '{}'".format(args['resume_path']))
            checkpoint = torch.load(args['resume_path'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            if args['test'] is False : model = grad_false_keys_filter(model, args['grad_false_keys'])
            print("=> completed.")
            print("=> start iter {}, min loss {}"
                  .format(checkpoint['iter'], checkpoint['min_loss']))         
        else:
            print("=> no checkpoint found at '{}'".format(args['resume_path']))
            return

    elif args['pretrained_path']:
        if os.path.isfile(args['pretrained_path']):
            print("=> loading checkpoint '{}'".format(args['pretrained_path']))
            checkpoint = torch.load(args['pretrained_path'])            
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['state_dict']
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            update_dict = update_dict_filter(pretrained_dict, args['convert_dict'], model_dict)
            model_dict.update(update_dict)
            model.load_state_dict(update_dict, strict=False)
            model = grad_false_keys_filter(model, args['grad_false_keys'])
            print("=> completed.")
        else:
            print("=> no checkpoint found at '{}'".format(args['pretrained_path']))
            return

    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(model_named_params,
                                    lr=args['lr'],
                                    weight_decay=args['weight_decay'])        

    print("=> total model parameters: {:.3f}M".format(
        sum(p.numel() for p in model.parameters())/1000000.0))

    # -------------------- main loop --------------------
    it_dict = {}
    if args['resume_path']:
        it_dict['iter'] = checkpoint['iter'] + 1
        it_dict['min_train_loss'] = None
        it_dict['best_train_iter'] = None
        it_dict['min_val_loss'] = None
        it_dict['best_val_iter'] = None
        # it_dict['min_val_loss'] = checkpoint['min_loss']
        # it_dict['best_val_iter'] = checkpoint['iter']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        it_dict['iter'] = 0
        it_dict['min_train_loss'] = None
        it_dict['best_train_iter'] = None
        it_dict['min_val_loss'] = None
        it_dict['best_val_iter'] = None

    while it_dict['iter'] < args['epochs'] * len(loader['train']):
        it_dict = \
            iterater(loader, model, criterion, optimizer, args, summary, it_dict)
        gc.collect()
    return

def update_dict_filter(pretrained_dict, convert_dict, model_dict):
    update_dict = {}
    for pretrainedk in pretrained_dict.keys():
        converted = False
        for cvtk in convert_dict.keys():   
            if cvtk in pretrainedk: 
                newk = pretrainedk.replace(cvtk, convert_dict[cvtk])
                update_dict[newk] = pretrained_dict[pretrainedk]
                converted = True
                print(pretrainedk , '-->', newk)
        if converted == False:
            update_dict[pretrainedk] = pretrained_dict[pretrainedk]
    update_dict = {k: v for k, v in update_dict.items() if k in model_dict}
    return update_dict

def grad_false_keys_filter(model, grad_false_keys):
    for k, p in model.named_parameters():
        k_requires_grad = True
        for grad_false_key in grad_false_keys:
            if grad_false_key in k:
                p.requires_grad = False
                k_requires_grad = False
        if k_requires_grad: print(k)
    return model

if __name__ == '__main__':
    main()
