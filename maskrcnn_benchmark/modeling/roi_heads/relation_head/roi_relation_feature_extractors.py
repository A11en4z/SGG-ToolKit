# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
import cv2
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_intersection
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
import numpy as np
import copy
import math
# from mmrotate.core.bbox.transforms import rbbox2roi
import time
from maskrcnn_benchmark.modeling.utils import cat
from torch.autograd import Variable
import random
import os
from maskrcnn_benchmark.config import cfg

def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))

def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates

def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    
    center, w, h, theta = np.split(np.array(obboxes.cpu()), (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4], axis=-1)
    polys = get_best_begin_point(polys)
    return torch.tensor(polys).float().cpu()

        
def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if len(rboxes.tolist()) ==  5:
        rboxes = torch.Tensor([rboxes.tolist()])
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()

def obb2poly_le90_batch(rboxes):
    """Convert oriented bounding boxes to polygons for batch input.

    Args:
        rboxes (torch.Tensor): Shape (nums, 5) representing [x_ctr, y_ctr, w, h, angle].

    Returns:
        polys (torch.Tensor): Shape (nums, 4, 2) representing [x0, y0, x1, y1, x2, y2, x3, y3].
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 4, 2))

    finite_mask = torch.isfinite(rboxes)
    if not finite_mask.all():
        rboxes = torch.where(finite_mask, rboxes, torch.zeros_like(rboxes))

    x_ctr, y_ctr, width, height, angle = rboxes[:, 0], rboxes[:, 1], rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    width = width.clamp(min=1e-6)
    height = height.clamp(min=1e-6)
    angle = torch.where(torch.isfinite(angle), angle, torch.zeros_like(angle))
    tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5

    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y], dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2, N).permute(2, 0, 1)

    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)

    return polys.contiguous().view(N, 4, 2).cpu()

def bbox2roi_HBB(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


  
@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, cfg, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.pooler_resolution = resolution
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separete spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim//2), nn.ReLU(inplace=True),
                                              make_fc(out_dim//2, out_dim), nn.ReLU(inplace=True),
                                            ])

        # union rectangle size
        self.rect_size = resolution * 4 -1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            ])

        self.type = cfg.Type
        # self.RS_Leap = cfg.RS_Leap
        if "OBB" in self.type:
            from mmrotate.core.bbox.transforms import rbbox2roi
            self.rbbox2roi = rbbox2roi
        elif "HBB" in self.type:
            self.rbbox2roi = bbox2roi_HBB

    def encoder_box(self, head_proposal, tail_proposal):
        bsize = 27

        assert tail_proposal.bbox.shape[-1] == 5
        head_boxes = head_proposal.bbox
        tail_boxes = tail_proposal.bbox
        head_finite = torch.isfinite(head_boxes)
        if not head_finite.all():
            head_boxes = torch.where(head_finite, head_boxes, torch.zeros_like(head_boxes))
        tail_finite = torch.isfinite(tail_boxes)
        if not tail_finite.all():
            tail_boxes = torch.where(tail_finite, tail_boxes, torch.zeros_like(tail_boxes))
        head_boxes = torch.stack(
            [
                head_boxes[:, 0],
                head_boxes[:, 1],
                head_boxes[:, 2].clamp(min=1e-6),
                head_boxes[:, 3].clamp(min=1e-6),
                torch.where(torch.isfinite(head_boxes[:, 4]), head_boxes[:, 4], torch.zeros_like(head_boxes[:, 4])),
            ],
            dim=1,
        )
        tail_boxes = torch.stack(
            [
                tail_boxes[:, 0],
                tail_boxes[:, 1],
                tail_boxes[:, 2].clamp(min=1e-6),
                tail_boxes[:, 3].clamp(min=1e-6),
                torch.where(torch.isfinite(tail_boxes[:, 4]), tail_boxes[:, 4], torch.zeros_like(tail_boxes[:, 4])),
            ],
            dim=1,
        )

        h_cx = head_boxes[:, 0].detach().cpu().numpy()
        h_cy = head_boxes[:, 1].detach().cpu().numpy()
        t_cx = tail_boxes[:, 0].detach().cpu().numpy()
        t_cy = tail_boxes[:, 1].detach().cpu().numpy()
        
        # head_boxes1 = copy.deepcopy(head_boxes)
        # tail_boxes1 = copy.deepcopy(tail_boxes)
        # h_poly = np.array([obb2poly_le90(box)[0] for box in head_boxes]).reshape((-1, 4, 2))
        # t_poly = np.array([obb2poly_le90(box)[0] for box in tail_boxes]).reshape((-1, 4, 2))
        h_poly = np.array(obb2poly_le90_batch(head_boxes.detach().cpu()))
        t_poly = np.array(obb2poly_le90_batch(tail_boxes.detach().cpu()))
        '''
        h_poly[0][0][0]
        tensor([16.4138, 10.7253, 16.5100, 10.8948, 16.3681, 10.9753, 16.2719, 10.8059])
        h_poly1[0]
        array([[16.4138, 10.7253],
            [16.51  , 10.8948],
            [16.3681, 10.9753],
            [16.2719, 10.8059]], dtype=float32)
        '''

        h_x = np.where(h_poly[:, :, 0] <= h_cx[:, None], np.floor(h_poly[:, :, 0]), np.ceil(h_poly[:, :, 0]))
        h_y = np.where(h_poly[:, :, 1] <= h_cy[:, None], np.floor(h_poly[:, :, 1]), np.ceil(h_poly[:, :, 1]))
        h_1 = np.stack((h_x, h_y), axis=-1).astype(int)

        t_x = np.where(t_poly[:, :, 0] <= t_cx[:, None], np.floor(t_poly[:, :, 0]), np.ceil(t_poly[:, :, 0]))
        t_y = np.where(t_poly[:, :, 1] <= t_cy[:, None], np.floor(t_poly[:, :, 1]), np.ceil(t_poly[:, :, 1]))
        t_1 = np.stack((t_x, t_y), axis=-1).astype(int)
        
        #''' plot v1
        # head_rect = torch.zeros((len(h_1), self.rect_size, self.rect_size))
        # tail_rect = torch.zeros((len(t_1), self.rect_size, self.rect_size))
        head_rect = torch.zeros((len(h_1), bsize, bsize))
        tail_rect = torch.zeros((len(t_1), bsize, bsize))

        for i in range(len(h_1)):
            head_rect[i] = torch.tensor(cv2.fillPoly(np.zeros((bsize, bsize)), [h_1[i]], 1))
            tail_rect[i] = torch.tensor(cv2.fillPoly(np.zeros((bsize, bsize)), [t_1[i]], 1))
        head_rect = head_rect.float()
        tail_rect = tail_rect.float()

        # '''
        # head_rect = fill_poly_tensor(h_1, self.rect_size).float()
        # tail_rect = fill_poly_tensor(t_1, self.rect_size).float()


        return head_rect, tail_rect
        
    def compute_weight_uni_vis_data(self, fea):

        wei_data = F.softmax(self.liner(fea), 1) *  30 #torch.mean(self.data1)  # self.liner(fea)
        return wei_data

    def forward(self, x, proposals, rel_pair_idxs=None,OBj = None):
        device = x[0].device
        if rel_pair_idxs is None or len(rel_pair_idxs) == 0:
            if self.separate_spatial:
                empty_region = torch.empty((0, self.feature_extractor.out_channels), device=device, dtype=x[0].dtype)
                empty_spatial = torch.empty((0, self.feature_extractor.out_channels), device=device, dtype=x[0].dtype)
                return (empty_region, empty_spatial)
            return torch.empty((0, self.out_channels), device=device, dtype=x[0].dtype)

        total_num_rel = sum(int(r.shape[0]) for r in rel_pair_idxs)
        if total_num_rel == 0:
            if self.separate_spatial:
                empty_region = torch.empty((0, self.feature_extractor.out_channels), device=device, dtype=x[0].dtype)
                empty_spatial = torch.empty((0, self.feature_extractor.out_channels), device=device, dtype=x[0].dtype)
                return (empty_region, empty_spatial)
            return torch.empty((0, self.out_channels), device=device, dtype=x[0].dtype)

        union_proposals = []
        rect_inputs = []
        # start_time = time.time()
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]

            if "OBB" in self.type:
                union_proposal = boxlist_union(head_proposal, tail_proposal,flag2=True)
            else:
                 union_proposal = boxlist_union(head_proposal, tail_proposal)


                 
            union_proposals.append(union_proposal)

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
            # resize bbox to the scale rect_size
            # head_proposal = head_proposal.resize((self.rect_size * 10, self.rect_size * 10),RS = True)
            # tail_proposal = tail_proposal.resize((self.rect_size * 10, self.rect_size * 10),RS = True)

            if "OBB" in self.type:
                head_proposal = head_proposal.resize((self.rect_size, self.rect_size),RS = True)
                tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size),RS = True)
            else:
                head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
                tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))    
            
            head = []
            tail = []
           
            if "OBB" in self.type:  ## RS
                head_rect, tail_rect = self.encoder_box(head_proposal,tail_proposal) 

                '''
                for h_box,t_box in zip(head_proposal.bbox,tail_proposal.bbox):
                    h = np.zeros((self.rect_size, self.rect_size))
                    t = np.zeros((self.rect_size, self.rect_size))
                    h_cx,h_cy = float(h_box[0]),float(h_box[1])
                    t_cx,t_cy = float(t_box[0]),float(t_box[1])
                    h_poly = np.array(obb2poly_le90(h_box)[0]).reshape((4,2))
                    t_poly = np.array(obb2poly_le90(t_box)[0]).reshape((4,2))

                    
                    # head
                    h_1 = []
                    for h_xy in h_poly:
                        x1,y = float(h_xy[0]),float(h_xy[1])
                        if x1 <= h_cx:
                            x_new = int(math.floor(x1))
                        else:
                            x_new = int(math.ceil(x1))

                        if y <= h_cy:
                            y_new = int(math.floor(y))
                        else:
                            y_new = int(math.ceil(y))
                        h_1.append([x_new,y_new])
                    
                    ## tail
                    t_1 = []
                    for t_xy in t_poly:
                        x2,y = float(t_xy[0]),float(t_xy[1])
                        if x2 <= t_cx:
                            x_new = int(math.floor(x2))
                        else:
                            x_new = int(math.ceil(x2))

                        if y <= t_cy:
                            y_new = int(math.floor(y))
                        else:
                            y_new = int(math.ceil(y))
                        t_1.append([x_new,y_new])
                        
                    cv2.fillPoly(h,[np.array(h_1)],1)
                    cv2.fillPoly(t,[np.array(t_1)],1)
 
                    h = torch.tensor(h)
                    t = torch.tensor(t)
                    head.append(h)
                    tail.append(t)
                head_rect = torch.stack(head).float()
                tail_rect = torch.stack(tail).float()
                '''

            else:
            #### 重写resize 
                head_rect = ((dummy_x_range >= head_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                            (dummy_x_range <= head_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                            (dummy_y_range >= head_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                            (dummy_y_range <= head_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()
                tail_rect = ((dummy_x_range >= tail_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                            (dummy_x_range <= tail_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                            (dummy_y_range >= tail_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                            (dummy_y_range <= tail_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()
        
           
        
            ''' check 
            sum(sum(sum(head1 == head_rect))) == head1.shape[0] * head1.shape[1] *  head1.shape[2]
            tensor(True)
            sum(sum(sum(tail1 == tail_rect))) == head1.shape[0] * head1.shape[1] *  head1.shape[2]
            tensor(True)
            '''

            #rect_input = torch.stack((head1, tail1), dim=1).cuda()
            rect_input = torch.stack((head_rect, tail_rect), dim=1).to(device) # (num_rel, 4, rect_size, rect_size)
            rect_inputs.append(rect_input)
       
        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        rect_inputs = torch.cat(rect_inputs, dim=0)
        if rect_inputs.shape[0] == 0:
            rect_features = torch.empty(
                (0, self.rect_conv[4].out_channels, self.pooler_resolution, self.pooler_resolution),
                device=device,
                dtype=x[0].dtype,
            )
        else:
            rect_features = self.rect_conv(rect_inputs)


        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        if "OBB" in self.type or "HBB" in self.type: 
            assert OBj is not None
            rbox = []
            for kk in range(len(union_proposals)):
                bbox = union_proposals[kk].bbox
                if bbox.numel() > 0:
                    finite_mask = torch.isfinite(bbox)
                    if not finite_mask.all():
                        bbox = torch.where(finite_mask, bbox, torch.zeros_like(bbox))
                    if "OBB" in self.type:
                        w = bbox[:, 2].clamp(min=1e-6)
                        h = bbox[:, 3].clamp(min=1e-6)
                        angle = torch.where(torch.isfinite(bbox[:, 4]), bbox[:, 4], torch.zeros_like(bbox[:, 4]))
                        bbox = torch.stack([bbox[:, 0], bbox[:, 1], w, h, angle], dim=1)
                    else:
                        x1 = torch.min(bbox[:, 0], bbox[:, 2])
                        y1 = torch.min(bbox[:, 1], bbox[:, 3])
                        x2 = torch.max(bbox[:, 0], bbox[:, 2])
                        y2 = torch.max(bbox[:, 1], bbox[:, 3])
                        x2 = torch.where(x2 - x1 < 1e-6, x1 + 1e-6, x2)
                        y2 = torch.where(y2 - y1 < 1e-6, y1 + 1e-6, y2)
                        bbox = torch.stack([x1, y1, x2, y2], dim=1)
                rbox.append(bbox.to(device))
            # 提取特征
            rois = self.rbbox2roi(rbox).to(device)
            union_vis_features = OBj._bbox_forward(x, rois, flag = True )

        else:
              union_vis_features = self.feature_extractor.pooler(x, union_proposals)
        if union_vis_features.numel() > 0:
            finite_mask = torch.isfinite(union_vis_features)
            if not finite_mask.all():
                union_vis_features = torch.where(finite_mask, union_vis_features, torch.zeros_like(union_vis_features))
        if rect_features.numel() > 0:
            rect_finite_mask = torch.isfinite(rect_features)
            if not rect_finite_mask.all():
                rect_features = torch.where(rect_finite_mask, rect_features, torch.zeros_like(rect_features))
              
        # merge two parts
        if self.separate_spatial:
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            if region_features.numel() > 0:
                region_mask = torch.isfinite(region_features)
                if not region_mask.all():
                    region_features = torch.where(region_mask, region_features, torch.zeros_like(region_features))
            if spatial_features.numel() > 0:
                spatial_mask = torch.isfinite(spatial_features)
                if not spatial_mask.all():
                    spatial_features = torch.where(spatial_mask, spatial_features, torch.zeros_like(spatial_features))
            union_features = (region_features, spatial_features)
        else:
            union_features = union_vis_features + rect_features

            if "OBB" in self.type or "HBB" in self.type: 
                union_features = OBj.bbox_head(union_features,flag = True)
            else:
                union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels)
            if union_features.numel() > 0:
                union_mask = torch.isfinite(union_features)
                if not union_mask.all():
                    union_features = torch.where(union_mask, union_features, torch.zeros_like(union_features))


        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)
            
        return union_features



def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
