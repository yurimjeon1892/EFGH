import torch
import torch.nn as nn

from math import pi
from common.torch_utils import depth_img_from_cartesian_pc_torch, rotation_matrix_between_two_vectors_torch, translation_matrix_from_vector_torch, matrix_3x3_to_4x4

class Eloss(nn.Module):

    def __init__(self, args):
        super(Eloss, self).__init__()
        
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.cos_sim = nn.CosineSimilarity(dim=1).cuda() 

        self.device = args['DEVICE']      

        self.lambda_e_gn = args['lambda']['e_gn']  
        self.lambda_abs = 10.
        self.lambda_sgn = 1.                    

        self.target_e3_vector = torch.unsqueeze(torch.unsqueeze(torch.tensor([0., 0., 1.]), 0), -1).float()

        self.loss_name = ['e_gn','e_gn_sgn', 'e_gn_abs']     

    def compute(self, gt, pred):

        gt_rand_init_l = gt['rand_init_l'][:, :3, :3].clone().detach().float()      
        gt_e_gn = torch.bmm(gt_rand_init_l, self.target_e3_vector)
        gt_e_gn = gt_e_gn / torch.sqrt(torch.sum(torch.pow(gt_e_gn, 2), 1, keepdim=True))
        gt['e_gn'] = gt_e_gn.to(self.device).float()
        gt['e_l'] = rotation_matrix_between_two_vectors_torch(gt['e_gn'], self.target_e3_vector, device=self.device)    
       
        gt_e_gn_abs = torch.abs(gt['e_gn']) 
        gt_e_gn_sgn = torch.sign(gt['e_gn'])
        gt_e_gn_sgn[gt_e_gn_sgn==-1] = 0 # make sign of '0' as -1 (use -1 instead of 1 just because z<=0)
        gt_e_gn_sgn_as_cls = []
        for b in range(gt_e_gn_sgn.size(0)):
            e_gn_sgn_b = gt_e_gn_sgn[b, :, 0]
            gt_e_gn_sgn_cls = torch.unsqueeze((e_gn_sgn_b[0] * (2 ** 2) + e_gn_sgn_b[1] * (2 ** 1) + e_gn_sgn_b[2]).long(), 0)      
            gt_e_gn_sgn_as_cls.append(gt_e_gn_sgn_cls)  
        gt_e_gn_sgn_as_cls = torch.cat(gt_e_gn_sgn_as_cls, 0)
        gt['e_gn_abs'] = gt_e_gn_abs.to(self.device).float()
        gt['e_gn_sgn'] = gt_e_gn_sgn_as_cls.to(self.device).long()

        # Rotation Loss
        ## Cos_Proximity_Loss
        cos_sim = self.cos_sim(pred['e_gn_abs'], gt['e_gn_abs']) 
        loss_e_gn_abs = torch.mean( 1 - cos_sim ) * self.lambda_abs # use 1-cos(theta) to make loss as positive. 
        ## Cross_Entropy_Loss              
        loss_e_gn_sgn = self.cross_entropy_loss(pred['e_gn_sgn'], gt['e_gn_sgn']) * self.lambda_sgn
        loss_e_gn = loss_e_gn_abs + loss_e_gn_sgn 

        losses = {
            'e_gn': loss_e_gn * self.lambda_e_gn, 
            'e_gn_abs': loss_e_gn_abs * self.lambda_e_gn, 
            'e_gn_sgn': loss_e_gn_sgn * self.lambda_e_gn, 
        }
        return losses, gt

class Floss(nn.Module):

    def __init__(self, args):
        super(Floss, self).__init__()

        self.dataset = args["dataset"]
        self.bce = nn.BCELoss(reduction='none')    

        self.device = args['DEVICE']
        self.positive_num = args['fov_pos_num']        
        self.neg_ratio = args['fov_neg_ratio']    
        self.lambda_fov = args['lambda']['fov']  

        self.target_e1_vector = torch.unsqueeze(torch.unsqueeze(torch.tensor([1., 0., 0.]), 0), -1).float().to(self.device)  
    
        self.loss_name = ['fov']

    def compute(self, gt, pred):

        gt_sensor2_T_sensor1 = gt['sensor2_T_sensor1'][:, :3, :3].clone().detach().float()
        gt_sensor2_T_sensor1_inv = torch.inverse(gt_sensor2_T_sensor1).to(self.device)

        pred_e_l = pred['e_l'][:, :3, :3].clone().detach().float().to(self.device)        
        gt_f_l_inv = torch.bmm(pred_e_l, gt_sensor2_T_sensor1_inv)
        
        gt_f_axis = torch.bmm(gt_f_l_inv, self.target_e1_vector)
        gt['f_score'] = self.gt_fov(gt_f_axis, pred['f_score'].size(-1))

        gt_e_l = gt['e_l'][:, :3, :3].clone().detach().float().to(self.device)
        gt_f_l_inv = torch.bmm(gt_e_l, gt_sensor2_T_sensor1_inv)
        gt['f_l'] = matrix_3x3_to_4x4(torch.inverse(gt_f_l_inv), self.device)

        pos = gt['f_score'] > 0
        batch = gt['f_score'].size(0)

        # Compute max conf across batch for hard negative mining
        loss_c = self.bce(pred['f_score'], gt['f_score'])

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(batch, -1)
        _, loss_idx = loss_c.sort(1, descending=True) # B, N, C
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        weighted = (pos + neg).gt(0)
        conf_p = pred['f_score'][weighted].view(batch, -1)
        targets_weighted = gt['f_score'][weighted].view(batch, -1)  
        loss_fov = self.bce(conf_p, targets_weighted)
        loss_fov = torch.mean(loss_fov) 

        losses = {
            'fov': loss_fov * self.lambda_fov,    
        }
        return losses, gt

    def gt_fov(self, gt_f_axis, fov_width):

        f_yaw_rad = []
        for b in range(gt_f_axis.size(0)):
            rad = torch.tensor([torch.atan2(gt_f_axis[b, 1, 0], gt_f_axis[b, 0, 0])])
            f_yaw_rad.append(torch.unsqueeze(rad, 0)) 
        f_yaw_rad = torch.cat(f_yaw_rad, 0)
        f_indices = ((-f_yaw_rad + pi) / (2 * pi)) * fov_width

        zz = torch.zeros((gt_f_axis.size(0), fov_width))
        oo = torch.ones((self.positive_num))
        for b in range(gt_f_axis.size(0)):
            f_idx = f_indices[b]
            xmin = int(f_idx) - int(self.positive_num / 2)
            xmax = xmin + self.positive_num
            if xmin >=0 and xmax < fov_width:
                zz[b, xmin:xmax] = oo
            elif xmin < 0:
                zz[b, 0:xmax] = oo[0:xmax]
                zz[b, xmin:] = oo[xmin:]
            else:
                zz[b, xmin:] = oo[0:fov_width-xmin]
                zz[b, 0:xmax-fov_width] = oo[fov_width-xmin:]
        f_score = zz.to(self.device).float()

        return f_score

class Gloss(nn.Module):

    def __init__(self, args):
        super(Gloss, self).__init__()

        self.smooth_l1 = torch.nn.SmoothL1Loss()
        self.bce = nn.BCELoss(reduction='mean')  
        
        self.device = args['DEVICE']  
        self.raw_cam_img_size = args['raw_cam_img_size']

        self.lambda_g_trs = args['lambda']['g_trs']
        self.lambda_g_depth = args['lambda']['g_depth']
        self.lambda_g_mask = args['lambda']['g_mask']

        self.target_origin = torch.unsqueeze(torch.unsqueeze(torch.tensor([0., 0., 0., 1.]), 0), -1).float().to(self.device)
        
        self.loss_name = ['g_trs', 'g_depth', 'g_mask']  

    def compute(self, gt, pred, pc):

        gt_sensor2_T_sensor1 = gt['sensor2_T_sensor1'].clone().detach().float().to(self.device)
        # gt_sensor2_T_sensor1_inv = torch.inverse(gt_sensor2_T_sensor1).to(self.device)

        pred_ef_l = torch.bmm(pred['f_l'], pred['e_l'])
        pred_ef_l_inv = torch.inverse(pred_ef_l).to(self.device)
        
        gt_g_l = torch.bmm(gt_sensor2_T_sensor1, pred_ef_l_inv)  
        gt_g_cp = torch.bmm(gt_g_l, self.target_origin)
        gt['g_trs'] = gt_g_cp[:, :3, :].to(self.device).float()

        gt_ef_l = torch.bmm(gt['f_l'], gt['e_l'])
        gt_ef_l_inv = torch.inverse(gt_ef_l)
        gt_g_l = torch.bmm(gt_sensor2_T_sensor1, gt_ef_l_inv)  
        gt_g_cp = torch.bmm(gt_g_l, self.target_origin)
        gt['g_l'] = translation_matrix_from_vector_torch(gt_g_cp, self.device)
  
        gt_depth = depth_img_from_cartesian_pc_torch(pc, gt['cam_T_velo'], self.raw_cam_img_size, self.device)
        gt['g_depth'] = torch.unsqueeze(gt_depth[:, -1, :, :], 1).to(self.device).float()
        gt['g_mask'] = (gt['g_depth'] > 0).float()
        gt['img_mask'] = gt['img_mask'].to(self.device)
        depth_loss_valid_mask = (gt['g_depth'] > 0).detach() * ( gt['img_mask'] > 0).detach()       

        ## Translation Loss
        loss_g_trs = self.smooth_l1(gt['g_trs'], pred['g_trs'])

        ## Depth loss   
        diff = gt['g_depth'] - pred['g_depth']
        diff = diff[depth_loss_valid_mask]
        loss_g_depth = (diff**2).mean()

        ## Mask loss   
        b = pred['g_mask'].size(0)
        loss_g_mask = self.bce(pred['g_mask'][:, 0, :, :].view((b, -1)), gt['g_mask'].view((b, -1))) * self.lambda_g_mask
        
        losses = {
            'g_trs': loss_g_trs * self.lambda_g_trs,
            'g_depth': loss_g_depth * self.lambda_g_depth,
            'g_mask': loss_g_mask * self.lambda_g_depth,
        }

        return losses, gt

class Hloss(nn.Module):

    def __init__(self, args):
        super(Hloss, self).__init__()

        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.cos_sim = nn.CosineSimilarity(dim=1).cuda()   

        self.device = args['DEVICE']  

        self.lambda_h_hrzn = args['lambda']['h_hrzn']  
        self.lambda_abs = 10.
        self.lambda_sgn = 1.         

        self.target_e2_vector = torch.unsqueeze(torch.unsqueeze(torch.tensor([0., 1., 0.]), 0), -1)

        self.loss_name = ['h_hrzn', 'h_hrzn_abs', 'h_hrzn_sgn']     

    def compute(self, gt, pred):

        gt_rand_init_c = gt['rand_init_c'][:, :3, :3].clone().detach().float()    
        gt_h_hrzn = torch.bmm(gt_rand_init_c, self.target_e2_vector)
        gt_h_hrzn = gt_h_hrzn / torch.sqrt(torch.sum(torch.pow(gt_h_hrzn, 2), 1, keepdim=True))
        gt['h_hrzn'] = gt_h_hrzn.to(self.device).float()
        h_c = rotation_matrix_between_two_vectors_torch(gt['h_hrzn'], self.target_e2_vector, device=self.device)    
        gt['h_c'] = h_c[:, :3, :3]
       
        gt_h_hrzn_abs = torch.abs(gt['h_hrzn'])[:, :2, :] 
        gt_h_hrzn_sgn = torch.sign(gt['h_hrzn'])
        gt_h_hrzn_sgn[gt_h_hrzn_sgn==-1] = 0 # make sign of '0' as -1 (use -1 instead of 1 just because z<=0)
        gt_h_hrzn_sgn_as_cls = []
        for b in range(gt_h_hrzn_sgn.size(0)):
            h_hrzn_sgn_b = gt_h_hrzn_sgn[b, :, 0]
            gt_h_hrzn_sgn_cls = torch.unsqueeze((h_hrzn_sgn_b[0] * (2 ** 1) + h_hrzn_sgn_b[1]).long(), 0)      
            gt_h_hrzn_sgn_as_cls.append(gt_h_hrzn_sgn_cls)  
        gt_h_hrzn_sgn_as_cls = torch.cat(gt_h_hrzn_sgn_as_cls, 0)
        gt['h_hrzn_abs'] = gt_h_hrzn_abs.to(self.device).float()
        gt['h_hrzn_sgn'] = gt_h_hrzn_sgn_as_cls.to(self.device).long()

        # Rotation Loss
        ## Cos_Proximity_Loss
        cos_sim = self.cos_sim(pred['h_hrzn_abs'], gt['h_hrzn_abs']) 
        loss_h_hrzn_abs = torch.mean( 1 - cos_sim ) * self.lambda_abs # use 1-cos(theta) to make loss as positive. 
        ## Cross_Entropy_Loss              
        loss_h_hrzn_sgn = self.cross_entropy_loss(pred['h_hrzn_sgn'], gt['h_hrzn_sgn']) * self.lambda_sgn
        loss_h_hrzn = loss_h_hrzn_abs + loss_h_hrzn_sgn 

        losses = {
            'h_hrzn': loss_h_hrzn * self.lambda_h_hrzn, 
            'h_hrzn_abs': loss_h_hrzn_abs * self.lambda_h_hrzn, 
            'h_hrzn_sgn': loss_h_hrzn_sgn * self.lambda_h_hrzn, 
        }

        return losses, gt