from .loss_utils import *

class EFGHCriterion(nn.Module):

    def __init__(self, args):
        super(EFGHCriterion, self).__init__()
        
        self.eloss = Eloss(args)    
        self.hloss = Hloss(args)        
        self.floss = Floss(args)  
        self.gloss = Gloss(args)      

        self.loss_name = ['total']
        self.loss_name += self.eloss.loss_name
        self.loss_name += self.hloss.loss_name
        self.loss_name += self.floss.loss_name
        self.loss_name += self.gloss.loss_name

        print('=> complute loss list: ', self.loss_name)

    def compute_loss(self, pc, img, calib, A, gt, pred):

        losses = {}
        eloss, gt = self.eloss.compute(gt, pred)
        losses.update(eloss)
        hloss, gt = self.hloss.compute(gt, pred)
        losses.update(hloss)
        floss, gt = self.floss.compute(gt, pred)
        losses.update(floss)
        gloss, gt = self.gloss.compute(gt, pred, pc)
        losses.update(gloss)

        total = 0
        for k in losses.keys():
            total += losses[k]
        losses['total'] = total

        return losses, gt
        
