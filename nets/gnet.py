import torch
import torch.nn as nn

import nets.resnet as RESNET
from nets.net_utils import conv_bn_relu, convt_bn_relu, init_weights
from common.torch_utils import depth_img_from_cartesian_pc_torch, translation_matrix_from_vector_torch, concat_tensors

__all__ = ['Gnet']

class Gnet(nn.Module):

    def __init__(self, args):
        super(Gnet, self).__init__()

        self.device = args['DEVICE']   
        self.raw_cam_img_size = args['raw_cam_img_size']
        self.range_img_size = (int(args['raw_cam_img_size'][0] / 2),  int(args['raw_cam_img_size'][1] * 2))
        self.lidar_fov_rad = args['lidar_fov_rad']

        self.conv_i0 = conv_bn_relu(3,
                                    64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        # self.conv_d0 = conv_bn_relu(4,
        #                             32,
        #                             kernel_size=3,
        #                             stride=2,
        #                             padding=1)
        # pretrained_model = RESNET.__dict__['resnet34'](pretrained=False)
        pretrained_model = RESNET.__dict__['resnet18'](pretrained=False)
        pretrained_model.apply(init_weights)
        self.conv_img2 = pretrained_model._modules['layer1']
        self.conv_img3 = pretrained_model._modules['layer2']
        self.conv_img4 = pretrained_model._modules['layer3']
        self.conv_img5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory
        self.convt_img4 = convt_bn_relu(in_channels=512,
                                        out_channels=256,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1)
        self.convt_img3 = convt_bn_relu(in_channels=(256 + 256),
                                        out_channels=128,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1)
        self.convt_img2 = convt_bn_relu(in_channels=(128 + 128),
                                        out_channels=64,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1)
        self.convt_dimg = convt_bn_relu(in_channels=(64 + 64),
                                        out_channels=1,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1)
        self.convt_mask = convt_bn_relu(in_channels=(64 + 64),
                                        out_channels=2,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1)
        self.softmax = nn.Softmax(dim=1).cuda()
        ## -----

        self.conv_i1 = conv_bn_relu(64,
                                    32,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.conv_d1 = conv_bn_relu(4,
                                    32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        # pretrained_model = RESNET.__dict__['resnet50'](pretrained=False)
        pretrained_model = RESNET.__dict__['resnet18'](pretrained=False)
        pretrained_model.apply(init_weights)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory        
        # self.conv_trs_1 = conv_bn_relu(2048, 512, 1)
        self.conv_trs_1 = conv_bn_relu(512, 512, 1)
        self.conv_trs_2 = conv_bn_relu(512, 512, 1)
        self.conv_trs_3 = conv_bn_relu(512, 512, 1)
        self.conv_trs_4 = nn.Conv1d(512, 3, 1) 

        return

    def forward(self, pc, img, ret, check=False):
        
        if check:                         
            print("[G] img               ", img.size()) 

        # conv_img1 = self.conv_img1(img)
        conv_img1 = self.conv_i0(img)        
        conv_img2 = self.conv_img2(conv_img1)  
        conv_img3 = self.conv_img3(conv_img2) 
        conv_img4 = self.conv_img4(conv_img3) 
        conv_img5 = self.conv_img5(conv_img4)  

        if check:             
            print("[G] conv_img1         ", conv_img1.size())
            print("[G] conv_img2         ", conv_img2.size())
            print("[G] conv_img3         ", conv_img3.size())   
            print("[G] conv_img4         ", conv_img4.size())   
            print("[G] conv_img5         ", conv_img5.size())   

        convt_img4 = self.convt_img4(conv_img5)
        convt_img3 = concat_tensors(conv_img4, convt_img4)
        convt_img3 = self.convt_img3(convt_img3)
        convt_img2 = concat_tensors(conv_img3, convt_img3)
        convt_img2 = self.convt_img2(convt_img2)
        convt = torch.cat((convt_img2, conv_img2), 1)
        dimg = self.convt_dimg(convt)
        mask = self.convt_mask(convt)
        mask = self.softmax(mask)

        if check:         
            print("[G] convt_img4        ", convt_img4.size())
            print("[G] convt_img3        ", convt_img3.size())
            print("[G] convt_img2        ", convt_img2.size())
            print("[G] convt_img1        ", convt.size())
            print("[G] dimg              ", dimg.size())
            print("[G] mask              ", mask.size())

        f_pc = torch.cat([pc.clone()  , torch.ones(pc.size(0), 1, pc.size(2)).to(self.device)], 1)
        f_pc = torch.bmm(ret['sensor2_T_sensor1'], f_pc)     
        f_depth = depth_img_from_cartesian_pc_torch(pc, ret['efh_cam_T_velo'], self.raw_cam_img_size, device=self.device)    

        if check:      
            print("[G] pc                ", pc.size())                    
            print("[G] f_depth           ", f_depth.size()) 
        
        conv_i1 = self.conv_i1(convt_img2)
        conv_d1 = self.conv_d1(f_depth)
        conv1 = torch.cat((conv_i1, conv_d1), 1)  # batchsize * ? * 352 * 1216

        conv2 = self.conv2(conv1)  # batchsize * ? * 352 * 1216
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152

        if check:         
            print("[G] conv_i1           ", conv_i1.size())
            print("[G] conv_d1           ", conv_d1.size())
            print("[G] conv1             ", conv1.size())
            print("[G] conv2             ", conv2.size())
            print("[G] conv3             ", conv3.size())   
            print("[G] conv4             ", conv4.size())   
            print("[G] conv5             ", conv5.size())

        trs1 = self.conv_trs_1(conv5)
        trs2 = self.conv_trs_2(trs1)
        trs3 = self.conv_trs_3(trs2)
        trs3 = trs3.view(trs3.size(0), trs3.size(1), -1)   
        trs4 = self.conv_trs_4(trs3)
        trs = torch.mean(trs4, 2, keepdim=True)
        g_T = translation_matrix_from_vector_torch(trs, device=self.device) 

        if check:            
            print("[G] trs1              ", trs1.size())
            print("[G] trs2              ", trs2.size())
            print("[G] trs3              ", trs3.size())
            print("[G] trs4              ", trs4.size())
            print("[G] trs               ", trs.size())
            print("[G] g_T               ", g_T.size())

        ret['g_depth'] = dimg
        ret['g_mask'] = mask
        ret['g_trs'] = trs
        ret['g_l'] = g_T  
        ret['sensor2_T_sensor1'] = torch.bmm(ret['g_l'], ret['sensor2_T_sensor1'])
        ret['network'] = ret['network'] + 'G'

        if check:
            print("[G] g_depth           ", ret['g_depth'].size())  
            print("[G] g_trs             ", ret['g_trs'].size())  
            print("[G] g_l               ", ret['g_l'].size())              
            print("[G] sensor2_T_sensor1 ", ret['sensor2_T_sensor1'].size())
            print("[G] network           ", ret['network'])        
        
        return ret