import torch
import torch.nn as nn
import torch.nn.functional as F

from math import pi

from nets.vgg import vgg11_bn_modified
from nets.net_utils import conv_bn_relu, convt_bn_relu
from common.torch_utils import range_img_from_cartesian_pc_torch, rotation_matrix_between_two_vectors_torch, circular_assign_torch, vector_from_radian_torch

__all__ = ['F']

class Fnet(nn.Module):

    def __init__(self, args):
        super(Fnet, self).__init__()

        self.device = args['DEVICE']    
        self.range_img_size = (int(args['raw_cam_img_size'][0] / 2),  int(args['raw_cam_img_size'][1] * 2))
        self.lidar_fov_rad = args['lidar_fov_rad']

        self.vgg_camera = vgg11_bn_modified()
        self.vgg_5_1_camera = convt_bn_relu(512, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.vgg_5_2_camera = convt_bn_relu(128, 32, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.vgg_5_3_camera = convt_bn_relu(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.conv_range = conv_bn_relu(4, 3, kernel_size=(1, 2), stride=(1, 1), padding=0)
        self.vgg_range = vgg11_bn_modified()
        self.vgg_5_1_range = convt_bn_relu(512, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.vgg_5_2_range = convt_bn_relu(128, 32, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.vgg_5_3_range = convt_bn_relu(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=1)

        if args['DEVICE'] == "RELLIS_3D":
            self.target_e1_vector = torch.unsqueeze(torch.unsqueeze(torch.tensor([-1., 0., 0.]), 0), -1)
        else:
            self.target_e1_vector = torch.unsqueeze(torch.unsqueeze(torch.tensor([1., 0., 0.]), 0), -1)

        return

    def forward(self, pc, ret, check=False):

        pc1 = pc.clone()
        pc1 = torch.cat([pc1, torch.ones(pc1.size(0), 1, pc1.size(2)).to(self.device)], 1)
        e_pc = torch.bmm(ret['e_l'], pc1)
        e_range = range_img_from_cartesian_pc_torch(e_pc, self.range_img_size, self.lidar_fov_rad, device=self.device)
        h_img = ret["h_img"]

        if check:                        
            print("[F] e_pc              ", e_pc.size())
            print("[F] e_range           ", e_range.size())
            print("[F] h_img             ", h_img.size())

        conv_cam0 = self.vgg_camera(h_img)
        conv_cam1 = self.vgg_5_1_camera(conv_cam0)
        conv_cam2 = self.vgg_5_2_camera(conv_cam1)
        conv_cam3 = self.vgg_5_3_camera(conv_cam2)
        cam_feat = (conv_cam3) / (torch.max(conv_cam3) - torch.min(conv_cam3))    

        conv_rng0 = self.conv_range(e_range)
        conv_rng0 = self.vgg_range(conv_rng0)   
        conv_rng1 = self.vgg_5_1_range(conv_rng0)
        conv_rng2 = self.vgg_5_2_range(conv_rng1)
        conv_rng3 = self.vgg_5_3_range(conv_rng2)    
        rng_feat = (conv_rng3) / (torch.max(conv_rng3) - torch.min(conv_rng3)) 

        if check:
            print("[F] conv_cam0         ", conv_cam0.size())            
            print("[F] conv_cam1         ", conv_cam1.size())
            print("[F] conv_cam2         ", conv_cam2.size())
            print("[F] conv_cam3         ", conv_cam3.size())
            print("[F] cam_feat          ", cam_feat.size())
            print("[F] conv_rng0         ", conv_rng0.size())  
            print("[F] conv_rng1         ", conv_rng1.size())   
            print("[F] conv_rng2         ", conv_rng2.size())   
            print("[F] conv_rng3         ", conv_rng3.size())   
            print("[F] rng_feat          ", rng_feat.size()) 

        rng_feat = circular_assign_torch(rng_feat, int(rng_feat.size(-1) / 8), self.device)
        f_score = F.conv2d(rng_feat, cam_feat, stride=1, padding=0)
        f_score = f_score / (cam_feat.size(0) * cam_feat.size(1))
        f_score = torch.sigmoid(f_score.view(-1)).view(f_score.size(0), -1)

        if check:            
            print("[F] rng_feat2         ", rng_feat.size())
            print("[F] f_score           ", f_score.size())

        f_idx = torch.argmax(f_score, dim=1, keepdim=True).float()
        f_rad = - (f_idx / (f_score.size(-1) - 1)) * 2 * pi + pi
        f_fwd = vector_from_radian_torch(f_rad, self.device)
        # f_T = rotation_matrix_between_two_vectors_torch(self.target_e1_vector, f_fwd, device=self.device)
        f_T = rotation_matrix_between_two_vectors_torch(f_fwd, self.target_e1_vector, device=self.device)

        if check:
            print("[F] f_idx             ", f_idx.size())        
            print("[F] f_rad             ", f_rad.size())            
            print("[F] f_fwd             ", f_fwd.size())
            print("[F] f_T               ", f_T.size())

        ret['f_score'] = f_score 
        ret['f_l'] = f_T
        ret['sensor2_T_sensor1'] = torch.bmm(ret['f_l'], ret['sensor2_T_sensor1'])
        ret['network'] = ret['network'] + 'F'

        if check:            
            print("[F] f_score           ", ret['f_score'].size())
            print("[F] f_l               ", ret['f_l'].size())
            print("[F] sensor2_T_sensor1 ", ret['sensor2_T_sensor1'].size())
            print("[F] network           ", ret['network'])        
        
        return ret