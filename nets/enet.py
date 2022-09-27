import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.generate_data import GenerateData
from nets.bilateralNN import BilateralConvFlex
from nets.net_utils import conv_1x1
from common.torch_utils import normal_vector_3d_from_abs_sign_torch, rotation_matrix_between_two_vectors_torch

__all__ = ['E']

class Enet(nn.Module):

    def __init__(self, args):
        super(Enet, self).__init__()

        dim = args['dim']
        scales_filter_map = args['scale_map']             
        chunk_size = -1

        self.device = args['DEVICE']         
        self.generate_data = GenerateData(dim, scales_filter_map, self.device)        

        self.conv_in = nn.Sequential(
            conv_1x1(dim, 32, use_leaky=args['use_leaky']),
            conv_1x1(32, 32, use_leaky=args['use_leaky']),
            conv_1x1(32, 32, use_leaky=args['use_leaky']),
        )

        self.bcn1 = BilateralConvFlex(dim, scales_filter_map[0][1],
                                      32 + dim + 1, [32, 32],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=chunk_size)

        self.bcn2 = BilateralConvFlex(dim, scales_filter_map[1][1],
                                      32 + dim + 1, [64, 64],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=chunk_size)

        self.bcn3 = BilateralConvFlex(dim, scales_filter_map[2][1],
                                      64 + dim + 1, [128, 128],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=chunk_size)

        self.bcn4 = BilateralConvFlex(dim, scales_filter_map[3][1],
                                      128 + dim + 1, [256, 256],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=chunk_size)
        
        self.bcn5 = BilateralConvFlex(dim, scales_filter_map[4][1],
                                      256 + dim + 1, [256, 256],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=chunk_size)

        self.conv_gn_1 = nn.Conv1d(256, 128, 1)
        self.conv_gn_2 = nn.Conv1d(128, 128, 1)
        self.conv_gn_3 = nn.Conv1d(128, 128, 1)

        self.bn_gn_1 = nn.BatchNorm1d(128)
        self.bn_gn_2 = nn.BatchNorm1d(128)
        self.bn_gn_3 = nn.BatchNorm1d(128)

        self.lin_gn_1 = nn.Linear(128, 128)
        self.lin_gn_2 = nn.Linear(128, 128)
        self.lin_gn_3 = nn.Linear(128, 32)
        self.lin_gn_abs = nn.Linear(32, 3)
        self.lin_gn_sgn = nn.Linear(32, 8)

        self.softmax = nn.Softmax(dim=1).cuda()

        self.target_e3_vector = torch.unsqueeze(torch.unsqueeze(torch.tensor([0., 0., 1.]), 0), -1)

    def forward(self, pc, check=False):

        pc1 = pc.clone()

        pc1, generated_data = self.generate_data(pc1[0, :, :])

        pc1 = torch.unsqueeze(pc1, 0).to(self.device)

        pc1_out0 = self.conv_in(pc1[:, :3, :])

        pc1_out1 = self.bcn1(torch.cat((generated_data[0]['pc1_el_minus_gr'], pc1_out0), dim=1),
                             in_barycentric=generated_data[0]['pc1_barycentric'],
                             in_lattice_offset=generated_data[0]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[0]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc1_out2 = self.bcn2(torch.cat((generated_data[1]['pc1_el_minus_gr'], pc1_out1), dim=1),
                             in_barycentric=generated_data[1]['pc1_barycentric'],
                             in_lattice_offset=generated_data[1]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[1]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc1_out3 = self.bcn3(torch.cat((generated_data[2]['pc1_el_minus_gr'], pc1_out2), dim=1),
                             in_barycentric=generated_data[2]['pc1_barycentric'],
                             in_lattice_offset=generated_data[2]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[2]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc1_out4 = self.bcn4(torch.cat((generated_data[3]['pc1_el_minus_gr'], pc1_out3), dim=1),
                             in_barycentric=generated_data[3]['pc1_barycentric'],
                             in_lattice_offset=generated_data[3]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[3]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)
        
        pc1_out5 = self.bcn5(torch.cat((generated_data[4]['pc1_el_minus_gr'], pc1_out4), dim=1),
                             in_barycentric=generated_data[4]['pc1_barycentric'],
                             in_lattice_offset=generated_data[4]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[4]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        if check:
            print("[E] pc1_out1          ", pc1_out1.size())
            print("[E] pc1_out2          ", pc1_out2.size())
            print("[E] pc1_out3          ", pc1_out3.size())
            print("[E] pc1_out4          ", pc1_out4.size())
            print("[E] pc1_out5          ", pc1_out5.size())
        
        gn_cv_1 = F.relu(self.bn_gn_1(self.conv_gn_1(pc1_out5)))
        gn_cv_2 = F.relu(self.bn_gn_2(self.conv_gn_2(gn_cv_1)))
        gn_cv_3 = F.relu(self.bn_gn_3(self.conv_gn_3(gn_cv_2)))   

        gn_fc_0 = torch.max(gn_cv_3, 2, keepdim=True)[0]
        gn_fc_0 = gn_fc_0.view(gn_fc_0.size(0), -1)
        gn_fc_1 = F.relu(self.lin_gn_1(gn_fc_0))
        gn_fc_2 = F.relu(self.lin_gn_2(gn_fc_1))
        gn_fc_3 = F.relu(self.lin_gn_3(gn_fc_2))
        
        gn_sgn = self.lin_gn_sgn(gn_fc_3)
        gn_abs_0 = self.lin_gn_abs(gn_fc_3)  
        gn_abs_0 = self.softmax(gn_abs_0)
        gn_abs_denom = torch.sqrt(torch.sum(torch.pow(gn_abs_0, 2), 1, keepdim=True))
        gn_abs = torch.unsqueeze(gn_abs_0 / gn_abs_denom, -1)

        if check:            
            print("[E] gn_cv_1           ", gn_cv_1.size())
            print("[E] gn_cv_2           ", gn_cv_2.size())
            print("[E] gn_cv_3           ", gn_cv_3.size())
            print("[E] gn_fc_0           ", gn_fc_0.size())
            print("[E] gn_fc_1           ", gn_fc_1.size())
            print("[E] gn_fc_2           ", gn_fc_2.size())
            print("[E] gn_sgn            ", gn_sgn.size())
            print("[E] gn_abs            ", gn_abs.size())            

        e_gn = normal_vector_3d_from_abs_sign_torch(gn_abs, gn_sgn, device=self.device)
        e_T = rotation_matrix_between_two_vectors_torch(e_gn, self.target_e3_vector, device=self.device)      
   
        ret = {
            'e_gn_abs': gn_abs,
            'e_gn_sgn': gn_sgn,            
            'e_gn': e_gn,      

            'e_l': e_T,
            'sensor2_T_sensor1': e_T,
            'network': 'E'            
            }

        if check:            
            print("[E] e_gn_abs          ", ret['e_gn_abs'].size())
            print("[E] e_gn_sgn          ", ret['e_gn_sgn'].size())
            print("[E] e_gn              ", ret['e_gn'].size())
            print("[E] e_l               ", ret['e_l'].size())
            print("[E] sensor2_T_sensor1 ", ret['sensor2_T_sensor1'].size())
            print("[E] network           ", ret['network'])        

        return ret
