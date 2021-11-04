'''
The loss function is inspired and modified from the DETR paper.
'''

import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment


class SetCriterion(nn.Module):
    '''
    The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    '''
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        
    def loss_points(self, outputs, targets, indices, num_points):
        """Compute the losses related to the dot points, the L1 regression loss
           targets contain the ground truth coordinates [x,y]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        _, src_idx = self._get_src_permutation_idx(indices)
        _, tgt_idx = self._get_tgt_permutation_idx(indices)

        src_points = outputs.view(-1,2)[src_idx]
        src_points = src_points.reshape(1,1,-1,2)
        target_points = targets.view(-1,2)[tgt_idx]
        target_points = target_points.reshape(1,1,-1,2)
        set_loss = F.l1_loss(src_points, target_points, reduction='none')
        losses = set_loss.sum() / num_points
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]) 
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_points = sum(len(t) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs)).device)
        num_points = torch.clamp(num_points, min=1).item()
        losses = self.loss_points(outputs, targets, indices, num_points)

        return losses
