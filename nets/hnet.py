import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.vgg import vgg11_bn
from common.torch_utils import normal_vector_2d_from_abs_sign_torch, rotation_matrix_between_two_vectors_torch, rotate_image_from_rotation_matrix_torch

__all__ = ['H']

class Hnet(nn.Module):

    def __init__(self, args):
        super(Hnet, self).__init__()

        self.device = args['DEVICE']    

        self.vgg = vgg11_bn()
        
        self.conv_hrzn_1 = torch.nn.Conv1d(512, 256, 1)
        self.conv_hrzn_2 = torch.nn.Conv1d(256, 128, 1)
        self.conv_hrzn_3 = torch.nn.Conv1d(128, 128, 1)

        self.bn_hrzn_1 = nn.BatchNorm1d(256)
        self.bn_hrzn_2 = nn.BatchNorm1d(128)
        self.bn_hrzn_3 = nn.BatchNorm1d(128)

        self.lin_hrzn_1 = nn.Linear(128, 128)
        self.lin_hrzn_2 = nn.Linear(128, 128)
        self.lin_hrzn_3 = nn.Linear(128, 32)
        self.lin_hrzn_abs = nn.Linear(32, 2)
        self.lin_hrzn_sgn = nn.Linear(32, 4) 

        self.softmax = nn.Softmax(dim=1).cuda()

        self.target_e2_vector = torch.unsqueeze(torch.unsqueeze(torch.tensor([0., 1., 0.]), 0), -1)

        return

    def forward(self, img, check=False):

        conv0 = self.vgg(img)
        conv1 = conv0.view((conv0.size(0), conv0.size(1), -1))

        if check:            
            print("[H] img               ", img.size())
            print("[H] conv0             ", conv0.size())
            print("[H] conv1             ", conv1.size())
        
        hrzn_cv_1 = F.relu(self.bn_hrzn_1(self.conv_hrzn_1(conv1)))
        hrzn_cv_2 = F.relu(self.bn_hrzn_2(self.conv_hrzn_2(hrzn_cv_1)))
        hrzn_cv_3 = F.relu(self.bn_hrzn_3(self.conv_hrzn_3(hrzn_cv_2)))   

        hrzn_fc_0 = torch.max(hrzn_cv_3, 2, keepdim=True)[0]
        hrzn_fc_0 = hrzn_fc_0.view(hrzn_fc_0.size(0), -1)
        hrzn_fc_1 = F.relu(self.lin_hrzn_1(hrzn_fc_0))
        hrzn_fc_2 = F.relu(self.lin_hrzn_2(hrzn_fc_1))
        hrzn_fc_3 = F.relu(self.lin_hrzn_3(hrzn_fc_2))
        
        hrzn_sgn = self.lin_hrzn_sgn(hrzn_fc_3)
        hrzn_abs_0 = self.lin_hrzn_abs(hrzn_fc_3)  
        hrzn_abs_0 = self.softmax(hrzn_abs_0)
        hrzn_abs_denom = torch.sqrt(torch.sum(torch.pow(hrzn_abs_0, 2), 1, keepdim=True))
        hrzn_abs = torch.unsqueeze(hrzn_abs_0 / hrzn_abs_denom, -1)

        if check:            
            print("[H] hrzn_cv_1         ", hrzn_cv_1.size())
            print("[H] hrzn_cv_2         ", hrzn_cv_2.size())
            print("[H] hrzn_cv_3         ", hrzn_cv_3.size())
            print("[H] hrzn_fc_0         ", hrzn_fc_0.size())
            print("[H] hrzn_fc_1         ", hrzn_fc_1.size())
            print("[H] hrzn_fc_2         ", hrzn_fc_2.size())
            print("[H] hrzn_sgn          ", hrzn_sgn.size())
            print("[H] hrzn_abs          ", hrzn_abs.size())            

        h_hrzn = normal_vector_2d_from_abs_sign_torch(hrzn_abs, hrzn_sgn, device=self.device)        
        h_hrzn_3d = torch.cat([h_hrzn, torch.zeros(h_hrzn.size(0), 1, 1).to(self.device)], 1)
        h_T = rotation_matrix_between_two_vectors_torch(h_hrzn_3d, self.target_e2_vector, device=self.device) 
        h_T = h_T[:, :3, :3]
        h_img = rotate_image_from_rotation_matrix_torch(img, h_T, self.device)        

        ret = {
            'h_hrzn_abs': hrzn_abs,
            'h_hrzn_sgn': hrzn_sgn,
            'h_hrzn': h_hrzn,
            'h_img': h_img,

            'h_c': h_T,
            'intrinsic_sensor2': h_T,
            'network': 'H'
        }

        if check:
            print("[H] h_hrzn_abs        ", ret['h_hrzn_abs'].size())
            print("[H] h_hrzn_sgn        ", ret['h_hrzn_sgn'].size())            
            print("[H] h_hrzn            ", ret['h_hrzn'].size())
            print("[H] h_img             ", ret['h_img'].size())
            print("[H] h_c               ", ret['h_c'].size())
            print("[H] intrinsic_sensor2 ", ret['intrinsic_sensor2'].size())
            print("[H] network           ", ret['network'])
        
        return ret