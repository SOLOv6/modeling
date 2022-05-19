import torch 
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss

class Loss_Functions():
    def __init__(self, focal_alpha = 0.0, focal_gamma = 0.0, both_weight_focal = 0.0, both_weight_bce = 0.0):
        """
        @param 
        - both_weight_focal: weight focal for bothloss
        - both_weight_bce: weight bce for bothloss
        - focal_alpha: prameter alpha for focalloss
        - focal_gamma: prameter gamma for focalloss
        """
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.both_weight_focal = both_weight_focal
        self.both_weight_bce = both_weight_bce
    
    def bce_loss(self, pred, target, batch_size, image_size):
        """
        @param 
        - pred : model ouput 
        - target : ground truth
        - batch_size : batch size 
        - image_size : image size
        """
        # pos weight 
        dent_pos = torch.ones([batch_size, 1, image_size, image_size]) * 8
        scratch_pos = torch.ones([batch_size, 1, image_size, image_size]) * 8
        spacing_pos = torch.ones([batch_size, 1, image_size, image_size]) * 8
        total_pos = torch.cat([dent_pos, scratch_pos, spacing_pos], dim=1)

        pos_weight = (total_pos)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(pred, target)
        
        return loss

    
    def focal_loss(self, pred, target):
        """
        @param 
        - pred : model ouput 
        - target : ground truth
        """
        loss = sigmoid_focal_loss(pred, target,
                                  alpha=self.focal_alpha, gamma=self.focal_gamma,
                                  reduction='mean')
        return loss

    def both_loss(self, pred, target):
        """
        @param 
        - pred : model ouput 
        - target : ground truth
        """
        # calculate bce loss 
        loss_bce = self.bce_loss(pred, target)

        # calculate focal loss
        loss_focal = self.focal_loss(self.focal_alpha, self.focal_gamma)

        # weight loss
        loss = self.both_weight_focal * loss_focal + self.both_weight_bce * loss_bce

        return loss