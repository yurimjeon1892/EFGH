import torch.nn as nn

from nets.enet import Enet
from nets.hnet import Hnet
from nets.fnet import Fnet
from nets.gnet import Gnet
from common.torch_utils import compute_cam_T_velo_matrix

__all__ = ['EFGHBackbone']

class EFGHBackbone(nn.Module):

    def __init__(self, args):
        super(EFGHBackbone, self).__init__()
        
        self.E = Enet(args)     
        self.H = Hnet(args)          
        self.F = Fnet(args)    
        self.G = Gnet(args)   

        self.device = args['DEVICE']  

    def forward(self, pc, img, calib, A, check=False):

        rete = self.E(pc, check)
        reth = self.H(img, check)

        ret = {}
        for key, value in rete.items(): 
            ret[key] = value  
        for key, value in reth.items():  
            ret[key] = value 
        ret['network'] = rete['network'] + reth['network']
        ret['eh_cam_T_velo'] = compute_cam_T_velo_matrix(ret['intrinsic_sensor2'], ret['sensor2_T_sensor1'], calib, A, self.device)

        ret = self.F(pc, ret, check)
        ret['efh_cam_T_velo'] = compute_cam_T_velo_matrix(ret['intrinsic_sensor2'], ret['sensor2_T_sensor1'], calib, A, self.device)

        ret = self.G(pc, img, ret, check)
        ret['efgh_cam_T_velo'] = compute_cam_T_velo_matrix(ret['intrinsic_sensor2'], ret['sensor2_T_sensor1'], calib, A, self.device)

        ret['cam_T_velo'] = ret['efgh_cam_T_velo']

        return ret