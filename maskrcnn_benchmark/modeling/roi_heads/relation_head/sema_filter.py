from torch.autograd import Variable
import argparse
import copy
import os
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
import json




class sema_sx(nn.Module):
    def __init__(self, cfg=None, flag=None, min_count=1):
        super(sema_sx, self).__init__()

        """
        语义过滤器：根据训练集统计得到的 object-pair -> predicate 先验表过滤候选关系对。

        - 优先从训练集统计自动生成与当前数据集/类别数对齐的 SF_list；
        - 若生成失败，则回退到 relation_head 目录下的 SF_list.json。
        """

        self.min_count = int(min_count)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_sf_path = os.path.join(current_dir, "SF_list.json")

        self.mt = None
        if cfg is not None:
            self.mt = self._try_build_and_load_from_dataset_statistics(cfg)

        if self.mt is None:
            with open(default_sf_path, "r") as f:
                self.mt = json.load(f)

        self.obj_dim = len(self.mt)
        self.rel_dim = len(self.mt[0][0]) if self.obj_dim > 0 else 0

    def _try_build_and_load_from_dataset_statistics(self, cfg):
        """
        从训练集统计构建并缓存语义过滤表（SF_list）。

        返回:
            list: 三维 list，shape = [num_obj, num_obj, num_rel]；失败返回 None。
        """
        try:
            from maskrcnn_benchmark.data.build import get_dataset_statistics

            statistics = get_dataset_statistics(cfg)
            fg_matrix = statistics["fg_matrix"]
            if hasattr(fg_matrix, "detach"):
                fg_matrix = fg_matrix.detach().cpu().numpy()

            mt = (fg_matrix >= self.min_count).astype("int32")
            if mt.shape[2] > 0:
                mt[:, :, 0] = 0

            cache_dir = getattr(cfg, "OUTPUT_DIR", None) or ""
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(
                    cache_dir,
                    f"SF_list_{mt.shape[0]}_{mt.shape[2]}.json",
                )
                with open(cache_path, "w") as f:
                    json.dump(mt.tolist(), f)
                with open(cache_path, "r") as f:
                    return json.load(f)

            return mt.tolist()
        except Exception:
            return None



    def sx(self,rel_pair_idxs,obj, flag_labels = None):
        
        if rel_pair_idxs.numel() == 0:
            return [rel_pair_idxs]

        cp_rel_pair_idxs = copy.deepcopy(rel_pair_idxs)
        heads = obj[rel_pair_idxs[:, 0]].long()
        tails = obj[rel_pair_idxs[:, 1]].long()
        tep = torch.as_tensor(self.mt, device=heads.device)
        mask = torch.ones(len(rel_pair_idxs), dtype=torch.bool, device=rel_pair_idxs.device)
        valid = (
            (heads > 0)
            & (tails > 0)
            & (heads < self.obj_dim)
            & (tails < self.obj_dim)
        )
        if torch.any(valid):
            valid_heads = heads[valid]
            valid_tails = tails[valid]
            mt_list = tep[valid_heads, valid_tails]
            row_sums = torch.sum(mt_list, dim=1)
            zero_positions_valid = torch.nonzero(row_sums == 0, as_tuple=True)[0]
            valid_indices = torch.nonzero(valid, as_tuple=True)[0]
            if zero_positions_valid.numel() > 0:
                mask[valid_indices[zero_positions_valid]] = False
    

  

        filtered_rel_pair_idxs = rel_pair_idxs[mask].long()

        denom = len(cp_rel_pair_idxs)
        filtered = denom - int(mask.sum().item())
        save_ratio = (len(filtered_rel_pair_idxs) / denom) if denom > 0 else 0.0
        print("filtered / all: ", str(filtered) +  "/" + str(len(cp_rel_pair_idxs)), " save_ratio: ", save_ratio)
            
        return [filtered_rel_pair_idxs.to(rel_pair_idxs.device)]

        
