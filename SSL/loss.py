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
        # pos weight for each class
        if self.class_name == 'dent':
            pos_weight = torch.ones([batch_size, 1, image_size, image_size]) * 12
        elif self.class_name == 'scratch':
            pos_weight = torch.ones([batch_size, 1, image_size, image_size]) * 8
        else:
            pos_weight = torch.ones([batch_size, 1, image_size, image_size]) * 15

        # calculate bce loss
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


    def cam_loss(self, pred, target, cam_img):
        """
        @param 
        - pred : model ouput 
        - target : ground truth
        - cam_img : Class Activation Map
        """
        # calculate both_loss
        loss_both = self.both_loss(pred, target)
        
        # calculate cam loss
        cam_img[cam_img > 0.5] = 1
        cam_img[cam_img < 0.5] = 0

        loss_cam = self.bce_loss(pred, cam_img)

        loss = 0.7 * loss_both + 0.3 * loss_cam

        return loss
