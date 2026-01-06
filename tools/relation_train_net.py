# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""
import copy
import argparse
import datetime
import gzip
import os
import time
import os.path as osp
import re
import shutil
import tarfile
import torch
from mmcv.runner.checkpoint import save_checkpoint
import warnings
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from utils import show_params_status
from tqdm import tqdm

from mmrotate.models import build_detector

from mmdet.datasets import build_dataset as b_data
from mmdet.models import build_detector as b_det
from mmdet.datasets import build_dataloader as b_loader


from mmcv import Config, DictAction
# from mmdet.datasets import (build_dataloader, build_dataset,
#                             replace_ImageToTensor)

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from maskrcnn_benchmark.modeling.detector.b_test import build_dataloader


from mmdet.apis import init_random_seed, set_random_seed
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from collections import OrderedDict
from mmrotate.core.evaluation.eval_map import eval_rbbox_map
import mmcv
from mmcv.image import tensor2imgs
import matplotlib.pyplot as plt

from mmdet.core import encode_mask_results
from mmcv.cnn import fuse_conv_bn




from torch.optim import  lr_scheduler
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')
from numpy import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
import sys
seed_torch()
import torch.distributed as dist

from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

def _get_sgg_mode(cfg) -> str:
    """根据 cfg 推断 SGG 评测模式（predcls/sgcls/sgdet）。"""
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            return "predcls"
        return "sgcls"
    return "sgdet"

def _auto_update_longtail_ids(cfg, logger):
    """根据训练集谓词频次自动生成并写回 cfg.HEAD_IDS/BODY_IDS/TAIL_IDS。"""
    try:
        from maskrcnn_benchmark.data import get_dataset_statistics
    except Exception:
        from maskrcnn_benchmark.data.build import get_dataset_statistics

    stats = None
    if get_rank() == 0:
        try:
            stats = get_dataset_statistics(cfg)
        except Exception as e:
            logger.info("AUTO_LONGTAIL_IDS compute failed on rank0: {}".format(e))
            stats = None
    synchronize()
    if stats is None:
        try:
            stats = get_dataset_statistics(cfg)
        except Exception as e:
            logger.info("AUTO_LONGTAIL_IDS load failed: {}".format(e))
            return

    fg_matrix = stats.get("fg_matrix", None)
    if fg_matrix is None:
        logger.info("AUTO_LONGTAIL_IDS skipped: fg_matrix missing.")
        return

    if isinstance(fg_matrix, torch.Tensor):
        pred_freq = fg_matrix.sum(dim=(0, 1)).detach().cpu().numpy()
    else:
        pred_freq = torch.as_tensor(fg_matrix).sum(dim=(0, 1)).detach().cpu().numpy()

    if pred_freq.ndim != 1 or pred_freq.shape[0] <= 1:
        logger.info("AUTO_LONGTAIL_IDS skipped: invalid pred_freq shape {}.".format(getattr(pred_freq, "shape", None)))
        return

    pred_ids = list(range(1, int(pred_freq.shape[0])))
    pred_counts = [float(pred_freq[i]) for i in pred_ids]
    pred_ids_sorted = [i for i, _ in sorted(zip(pred_ids, pred_counts), key=lambda x: (-x[1], x[0]))]

    n = len(pred_ids_sorted)
    if n == 0:
        logger.info("AUTO_LONGTAIL_IDS skipped: no predicate classes.")
        return

    head_n = max(1, n // 3)
    body_n = max(1, n // 3)
    head_ids = sorted(pred_ids_sorted[:head_n])
    body_ids = sorted(pred_ids_sorted[head_n:head_n + body_n])
    tail_ids = sorted(pred_ids_sorted[head_n + body_n:])

    cfg.HEAD_IDS = head_ids
    cfg.BODY_IDS = body_ids
    cfg.TAIL_IDS = tail_ids
    logger.info(
        "AUTO_LONGTAIL_IDS updated: num_pred(no_bg)={} head={} body={} tail={}".format(
            n, len(head_ids), len(body_ids), len(tail_ids)
        )
    )


def _safe_harmonic_mean(a: float, b: float) -> float:
    """计算两个数的调和平均数，自动规避除零。"""
    denom = a + b
    if denom <= 0:
        return 0.0
    return 2.0 * a * b / denom


def _load_result_dict(result_dict_path: str):
    """从 result_dict.pytorch 读取评测字典，读取失败时返回 None。"""
    try:
        return torch.load(result_dict_path, map_location=torch.device("cpu"))
    except Exception:
        return None


def _compute_f1_from_result_dict(result_dict, mode: str, ks):
    """根据 result_dict 计算每个 K 的 R/mR/F1 以及平均 F1。"""
    per_k = {}
    f1_list = []
    for k in ks:
        recalls = result_dict.get(mode + "_recall", {}).get(k, [])
        r_k = float(sum(recalls) / max(len(recalls), 1))
        mr_k = float(result_dict.get(mode + "_mean_recall", {}).get(k, 0.0))
        f1_k = _safe_harmonic_mean(r_k, mr_k)
        per_k[int(k)] = dict(R=r_k, mR=mr_k, F1=f1_k)
        f1_list.append(f1_k)
    f1_avg = float(sum(f1_list) / max(len(f1_list), 1))
    return f1_avg, per_k


def _cleanup_checkpoints(output_dir: str, keep_last: int, keep_paths=()):
    """删除 output_dir 下多余的迭代 checkpoint，仅保留最近 keep_last 个。"""
    if not output_dir or not os.path.isdir(output_dir):
        return
    keep_paths = {os.path.abspath(p) for p in keep_paths if p}
    pattern = re.compile(r"^(\d+)\.pth$")
    candidates = []
    for name in os.listdir(output_dir):
        m = pattern.match(name)
        if not m:
            continue
        path = os.path.abspath(os.path.join(output_dir, name))
        if path in keep_paths:
            continue
        candidates.append((int(m.group(1)), path))
    candidates.sort(key=lambda x: x[0], reverse=True)
    to_delete = [p for _, p in candidates[keep_last:]]
    for p in to_delete:
        try:
            os.remove(p)
        except Exception:
            pass


def _cleanup_detectron_checkpoints(output_dir: str, keep_last: int, keep_model_final: bool = True):
    """删除 output_dir 下多余的 Detectron 命名 checkpoint，仅保留最近 keep_last 个。"""
    if not output_dir or not os.path.isdir(output_dir):
        return
    pattern = re.compile(r"^model_(\d+)\.pth$")
    candidates = []
    for name in os.listdir(output_dir):
        m = pattern.match(name)
        if not m:
            continue
        candidates.append((int(m.group(1)), os.path.abspath(os.path.join(output_dir, name))))
    candidates.sort(key=lambda x: x[0], reverse=True)
    keep_models = int(keep_last)
    if keep_model_final and os.path.exists(os.path.join(output_dir, "model_final.pth")):
        keep_models = max(keep_models - 1, 0)
    to_delete = [p for _, p in candidates[keep_models:]]
    for p in to_delete:
        try:
            os.remove(p)
        except Exception:
            pass


def _find_best_epoch_checkpoint(output_dir: str) -> str:
    """在 output_dir 下查找 best_epoch checkpoint，优先 best_epoch.pth，其次 best_epoch_*.pth（取编号最大）。"""
    if not output_dir or not os.path.isdir(output_dir):
        return ""
    direct = os.path.join(output_dir, "best_epoch.pth")
    if os.path.exists(direct):
        return direct

    pattern = re.compile(r"^best_epoch_(\d+)\.pth$")
    best_num = None
    best_path = ""
    for name in os.listdir(output_dir):
        m = pattern.match(name)
        if not m:
            continue
        num = int(m.group(1))
        path = os.path.join(output_dir, name)
        if best_num is None or num > best_num:
            best_num = num
            best_path = path
    return best_path


def _cleanup_best_epoch_checkpoints(output_dir: str, keep_paths=()):
    """删除 output_dir 下多余的 best_epoch checkpoint，仅保留 keep_paths 中列出的文件。"""
    if not output_dir or not os.path.isdir(output_dir):
        return
    keep_paths = {os.path.abspath(p) for p in keep_paths if p}
    patterns = (
        re.compile(r"^best_epoch\.pth$"),
        re.compile(r"^best_epoch_(\d+)\.pth$"),
    )
    for name in os.listdir(output_dir):
        if not any(p.match(name) for p in patterns):
            continue
        path = os.path.abspath(os.path.join(output_dir, name))
        if path in keep_paths:
            continue
        try:
            os.remove(path)
        except Exception:
            pass


def _build_mmcv_checkpoint_meta():
    """构造 mmcv.save_checkpoint 所需的 meta 信息。"""
    meta = {}
    meta["CLASSES"] = (
        'ship', 'boat', 'crane', 'goods_yard', 'tank', 'storehouse', 'breakwater', 'dock', 'airplane',
        'boarding_bridge', 'runway', 'taxiway', 'terminal', 'apron', 'gas_station', 'truck', 'car',
        'truck_parking', 'car_parking', 'bridge', 'cooling_tower', 'chimney', 'vapor', 'smoke', 'genset',
        'coal_yard', 'lattice_tower', 'substation', 'wind_mill', 'cement_concrete_pavement', 'toll_gate',
        'flood_dam', 'gravity_dam', 'ship_lock', 'ground_track_field', 'basketball_court', 'engineering_vehicle',
        'foundation_pit', 'intersection', 'soccer_ball_field', 'tennis_court', 'tower_crane', 'unfinished_building',
        'arch_dam', 'roundabout', 'baseball_diamond', 'stadium', 'containment_vessel'
    )
    return meta


def _format_bytes(num_bytes: int) -> str:
    """将字节数格式化为可读字符串（B/KB/MB/GB/TB）。"""
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(max(int(num_bytes), 0))
    for u in units:
        if size < 1024.0 or u == units[-1]:
            if u == "B":
                return "{}{}".format(int(size), u)
            return "{:.2f}{}".format(size, u)
        size /= 1024.0
    return "{:.2f}TB".format(size)


def _dir_size_bytes(path: str) -> int:
    """递归统计目录大小（字节），目录不存在时返回 0。"""
    if not path or not os.path.isdir(path):
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return int(total)


def _get_inference_compress_options(cfg):
    """从 cfg/环境变量读取 inference 压缩相关开关与参数。"""
    env_flag = os.environ.get("SGG_COMPRESS_INFERENCE", "").strip().lower()
    env_enabled = env_flag in ("1", "true", "yes", "y", "on")

    inf_cfg = getattr(cfg, "INFERENCE", None)
    enabled = bool(getattr(inf_cfg, "COMPRESS_OUTPUT", False)) or env_enabled
    compresslevel = int(getattr(inf_cfg, "COMPRESS_LEVEL", 1) if inf_cfg is not None else 1)
    delete_after = bool(getattr(inf_cfg, "DELETE_AFTER_COMPRESS", True) if inf_cfg is not None else True)
    min_size_mb = float(getattr(inf_cfg, "COMPRESS_MIN_SIZE_MB", 0) if inf_cfg is not None else 0)
    on_val = bool(getattr(inf_cfg, "COMPRESS_ON_VAL", True) if inf_cfg is not None else True)
    on_test = bool(getattr(inf_cfg, "COMPRESS_ON_TEST", True) if inf_cfg is not None else True)
    return dict(
        enabled=enabled,
        compresslevel=max(1, min(int(compresslevel), 9)),
        delete_after=delete_after,
        min_size_bytes=int(max(min_size_mb, 0.0) * 1024 * 1024),
        on_val=on_val,
        on_test=on_test,
    )


def _compress_dir_to_targz(src_dir: str, dst_targz_path: str, compresslevel: int, delete_after: bool, logger=None):
    """将目录打包为 .tar.gz，并可选删除原目录；过程耗时与压缩比会写入日志。"""
    if not src_dir or not os.path.isdir(src_dir):
        return ""
    dst_dir = os.path.dirname(os.path.abspath(dst_targz_path))
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    src_dir = os.path.abspath(src_dir)
    dst_targz_path = os.path.abspath(dst_targz_path)
    tmp_path = dst_targz_path + ".tmp"

    start = time.time()
    src_size = _dir_size_bytes(src_dir)

    if logger is not None:
        logger.info("[compress] start: {} -> {} (size={})".format(src_dir, dst_targz_path, _format_bytes(src_size)))

    try:
        for p in (tmp_path, dst_targz_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        with open(tmp_path, "wb") as raw_f:
            with gzip.GzipFile(fileobj=raw_f, mode="wb", compresslevel=int(compresslevel)) as gz_f:
                with tarfile.open(fileobj=gz_f, mode="w") as tar:
                    tar.add(src_dir, arcname=os.path.basename(src_dir))

        os.replace(tmp_path, dst_targz_path)
        dst_size = 0
        try:
            dst_size = int(os.path.getsize(dst_targz_path))
        except Exception:
            dst_size = 0

        if delete_after:
            try:
                shutil.rmtree(src_dir)
            except Exception:
                pass

        cost = time.time() - start
        if logger is not None:
            ratio = (float(dst_size) / float(src_size)) if src_size > 0 else 0.0
            logger.info(
                "[compress] done: {} (archive_size={}, ratio={:.3f}, time={:.2f}s, deleted_src={})".format(
                    dst_targz_path, _format_bytes(dst_size), ratio, cost, bool(delete_after)
                )
            )
        return dst_targz_path
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def parse_args_OBB(mmcf = None,mmwei = None):

    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--work-dir', default='/SGG_ToolKit/mmrote_RS/out', help='the dir to save logs and models')

    parser.add_argument('--config', default=mmcf, help='train config file path')
    parser.add_argument('--checkpoint',default= mmwei, help='checkpoint file')

    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--out',default='/SGG_ToolKit/mmrote_RS/checkpoints/1219/oriented_rcnn/outshiyan.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')

    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
                default='mAP',
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', default='/SGG_ToolKit/mmrote_RS/checkpoints', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')

    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    #####

    args = parser.parse_args(args=[])
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def parse_args_HBB(mmcf = None,mmwei = None):
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--config',default = mmcf, help='train config file path')

    parser.add_argument('--checkpoint',default= mmwei , help='')

    parser.add_argument('--work-dir', default='work-dir-RSLEAP',help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()

    parser.add_argument('--out',default="/SGG_ToolKit/mmdetection_RS/RSLEAP_HBB/15.pkl", help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')

    parser.add_argument(
        '--eval',
        type=str,
        default="bbox",
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')

    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args(args=[])
    
    # args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def train(cfg, local_rank, distributed, logger, debug=False,use_GAN = False,mmcf = None,mmwei = None):


    logger.info("***********************TYPE***********************")
    logger.info("do  " + cfg.Type)

    if cfg.Type != "CV": 
        logger.info("config: " + mmcf)
        logger.info("prewei: " + mmwei)

    if cfg.Type == "CV":  # ori SGG

        logger.info("***********************Step 1: model  construction***********************")
        print('\n')

        debug_print(logger, 'CV construction for faster rcnn -- bbox')
        model = build_detection_model(cfg)
        debug_print(logger, 'end model construction --- CV construction for faster rcnn -- bbox')

        logger.info('modules that should be always set in eval mode, their eval() method should be called after model.train() is called')
        eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
        fix_eval_modules(eval_modules)
        logger.info(" done ! param.requires_grad = False for model.rpn, model.backbone, model.roi_heads.box")

        # NOTE, we slow down the LR of the layers start with the names in slow_heads
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
            slow_heads = ["roi_heads.relation.box_feature_extractor",
                        "roi_heads.relation.union_feature_extractor.feature_extractor", ]
        else:
            slow_heads = []

        # load pretrain layers to new layers
        load_mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                        "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}

        if cfg.MODEL.ATTRIBUTE_ON:
            load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
            load_mapping[
                "roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        
        logger.info("print model parameters")
        logger.info(show_params_status(model))
        logger.info("done !!! print model parameters")

        device = cfg.MODEL.DEVICE # Btorch.device("cuda:0") #
        model.to(device)

        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        num_batch = cfg.SOLVER.IMS_PER_BATCH
        optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
        scheduler = make_lr_scheduler(cfg, optimizer, logger)
        debug_print(logger, 'end optimizer and shcedule')
        # Initialize mixed-precision training
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            logger.info('end distributed')
        else:
            logger.info('not distributed, singe GPU')

        arguments = {}
        arguments["iteration"] = 0

        logger.info("***********************Step 1: over***********************")
        print('\n')


        logger.info("***********************Step 1: model  construction over***********************")
        output_dir = cfg.OUTPUT_DIR


        logger.info("***********************Step 2: load pre_train_weights***********************")
        save_to_disk = get_rank() == 0
        checkpointer = DetectronCheckpointer(
            cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
        )
        # if there is certain checkpoint in output_dir, load it, else load pretrained detector
        if checkpointer.has_checkpoint():
            extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT,
                                                    update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
            arguments.update(extra_checkpoint_data)
        else:
            # load_mapping is only used when we init current model from detection model.
            checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
        debug_print(logger, 'end load checkpointer')

        logger.info("***********************Step 2: load pre_train_weights over***********************")
        
        logger.info("***********************Step 3: load datasets ***********************")

        cfg.SOLVER.START_ITER = arguments["iteration"]
        train_data_loader = make_data_loader(
            cfg,
            mode='train',
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        val_data_loaders = make_data_loader(
            cfg,
            mode='test',
            is_distributed=distributed,
        )

        
        
        debug_print(logger, 'end dataloader')
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        logger.info("***********************Step 3: load datasets over***********************")
        
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(train_data_loader)
        start_iter = arguments["iteration"]
        start_training_time = time.time()
        end = time.time()

    elif "OBB" in cfg.Type:

        ##### mmcv
        args_mmcv = parse_args_OBB(mmcf,mmwei)
        cfg_mmcv = Config.fromfile(args_mmcv.config)
        if args_mmcv.cfg_options is not None:
            cfg_mmcv.merge_from_dict(args_mmcv.cfg_options)

        print('\n')
        logger.info("***********************Step 1: model  construction***********************")
        logger.info('RS construction for faster rcnn -- rbox')

        #### 加入原始cfg
        cfg_mmcv.model["ori_cfg"] = cfg
        model_mmcv = build_detector(
            cfg_mmcv.model,
            train_cfg=cfg_mmcv.get('train_cfg'),
            test_cfg=cfg_mmcv.get('test_cfg'))
        # model_mmcv.init_weights()
        logger.info('end model construction --- RS construction for faster rcnn -- rbox')

        device = cfg.MODEL.DEVICE # to  GPU 
        model_mmcv.to(device)
 
        arguments = {}
        arguments["iteration"] = 0
        cfg["mmcv"] = cfg_mmcv.data.test


        ###
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
            slow_heads = ["roi_heads.relation.box_feature_extractor",
                        "roi_heads.relation.union_feature_extractor.feature_extractor",]
        else:
            slow_heads = []

        num_batch = cfg.SOLVER.IMS_PER_BATCH
        optimizer = make_optimizer(cfg, model_mmcv, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch)) 
        ####
        # optimizer = build_optimizer(model_mmcv, cfg_mmcv.optimizer)
        scheduler = make_lr_scheduler(cfg, optimizer, logger)

        logger.info("***********************Step 1: model  construction over and load pretrained weights ***********************")
        print('\n')
        logger.info("***********************Step 2: load datasets ***********************")
        arguments = {}
        arguments["iteration"] = 0

        cfg_trian = copy.deepcopy(cfg)
        cfg_val = copy.deepcopy(cfg)
        
        cfg_trian ["mmcv"] = cfg_mmcv.data.train
        cfg_val ["mmcv"] = cfg_mmcv.data.test
        train_data_loader = make_data_loader(
             cfg_trian,
            mode='train',
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )

        test_data_loaders = make_data_loader(
            cfg_val,
            mode='test',
            is_distributed=distributed,
        )

        val_data_loaders = make_data_loader(
            cfg_val,
            mode='val',
            is_distributed=distributed,
        )

        logger.info("***********************Step 2: load datasets over ***********************")
        print('\n')
        logger.info("***********************Step 3: Start training ***********************")
     
        meters = MetricLogger(delimiter="  ")
        max_iter = len(train_data_loader)

        start_iter = arguments["iteration"]
        cfg.SOLVER.START_ITER = arguments["iteration"]
        start_training_time = time.time()
        end = time.time()

        ###### 读取预训练权重
        checkpoint = load_checkpoint(model_mmcv, args_mmcv.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model_mmcv.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model_mmcv.CLASSES = dataset.CLASSES
        logger.info(args_mmcv.checkpoint)
        #### 
        model = model_mmcv
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optRimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

        eval_modules = (model.neck, model.backbone, model.rpn_head, model.roi_head )
    
    elif "HBB" in cfg.Type:

        args_mmcv = parse_args_HBB(mmcf,mmwei)
        cfg_mmcv = Config.fromfile(args_mmcv.config)
        if args_mmcv.cfg_options is not None:
            cfg_mmcv.merge_from_dict(args_mmcv.cfg_options)

        print('\n')
        logger.info("***********************Step 1: model  construction***********************")
        logger.info('RS construction for faster rcnn -- rbox')

        #### 加入原始cfg
        cfg_mmcv.model["ori_cfg"] = cfg
        model_mmcv = b_det(
            cfg_mmcv.model,
            train_cfg=cfg_mmcv.get('train_cfg'),
            test_cfg=cfg_mmcv.get('test_cfg'))
        logger.info('end model construction --- RS construction for faster rcnn -- rbox')

        device = cfg.MODEL.DEVICE # to  GPU 
        model_mmcv.to(device)
 
        arguments = {}
        arguments["iteration"] = 0
        cfg["mmcv"] = cfg_mmcv.data.test


        ###
        num_batch = cfg.SOLVER.IMS_PER_BATCH
        optimizer = make_optimizer(cfg, model_mmcv, logger, slow_heads=[], slow_ratio=10.0, rl_factor=float(num_batch)) 
        ####
        # optimizer = build_optimizer(model_mmcv, cfg_mmcv.optimizer)
        scheduler = make_lr_scheduler(cfg, optimizer, logger)

        logger.info("***********************Step 1: model  construction over and load pretrained weights ***********************")
        print('\n')
        logger.info("***********************Step 2: load datasets ***********************")
        arguments = {}
        arguments["iteration"] = 0

        cfg_trian = copy.deepcopy(cfg)
        cfg_val = copy.deepcopy(cfg)
        
        cfg_trian ["mmcv"] = cfg_mmcv.data.train
        cfg_val ["mmcv"] = cfg_mmcv.data.test
        train_data_loader = make_data_loader(
             cfg_trian,
            mode='train',
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )

        test_data_loaders = make_data_loader(
            cfg_val,
            mode='test',
            is_distributed=distributed,
        )

        val_data_loaders = make_data_loader(
            cfg_val,
            mode='val',
            is_distributed=distributed,
        )

        logger.info("***********************Step 2: load datasets over ***********************")
        print('\n')
        logger.info("***********************Step 3: Start training ***********************")
     
        meters = MetricLogger(delimiter="  ")
        max_iter = len(train_data_loader)

        start_iter = arguments["iteration"]
        cfg.SOLVER.START_ITER = arguments["iteration"]
        start_training_time = time.time()
        end = time.time()
      
        ###### 读取预训练权重
        checkpoint = load_checkpoint(model_mmcv, args_mmcv.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model_mmcv.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model_mmcv.CLASSES = dataset.CLASSES
        logger.info(args_mmcv.checkpoint)
        #### 
        model = model_mmcv
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

        eval_modules = (model.neck, model.backbone, model.rpn_head, model.roi_head)

      
    print_first_grad = True

    best_f1_avg = float("-inf")
    best_f1_iter = -1

    if cfg.Only_val:
        output_folder = getattr(cfg, "val_outpath", None)
        if output_folder in ("None", "none", "null", "NULL", ""):
            output_folder = None
        if output_folder is None and cfg.OUTPUT_DIR:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", "val")
        if output_folder:
            mkdir(output_folder)
        val_result = run_val(cfg, model, val_data_loaders, distributed, logger, output_folder=output_folder)
        sys.exit() 

    if cfg.Only_test:
        output_folder = getattr(cfg, "test_outpath", None)
        if output_folder in ("None", "none", "null", "NULL", ""):
            output_folder = None
        if output_folder is None and cfg.OUTPUT_DIR:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", "test")
        if output_folder:
            mkdir(output_folder)
        val_result = run_val(cfg, model, test_data_loaders, distributed, logger, output_folder=output_folder, dataset_names=cfg.DATASETS.TEST)
        sys.exit() 



    for iteration, (images, targets, _ , imgs, tar1) in enumerate(train_data_loader, start_iter):  
        
        if any(len(target) < 1 for target in targets):
             logger.error(
            f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
        data_time = time.time() - end
        iteration = iteration + 1 + cfg.ite_resume 



        arguments["iteration"] = iteration


        model.train()
        # model.eval()self.model.embedding.weight.data
        fix_eval_modules(eval_modules) # 先注释

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        
        
        loss_dict = model(images, targets, ite=iteration, logger = logger, sgd_data = [imgs, tar1] if imgs is not None else None)

        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        
        optimizer.zero_grad()
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad  # print grad or not


        print_first_grad = False

        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad],
                       max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)


        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % cfg.Print_iter == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        
        if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == max_iter:
            if cfg.Type != "CV":
                filename = cfg.OUTPUT_DIR + "/" + str(iteration) + ".pth"
                meta = _build_mmcv_checkpoint_meta()
                save_checkpoint(model, filename, optimizer=optimizer, meta=meta)
                _cleanup_checkpoints(cfg.OUTPUT_DIR, keep_last=3)
            else:
                        
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
                if iteration == max_iter:
                    checkpointer.save("model_final", **arguments)
                _cleanup_detectron_checkpoints(cfg.OUTPUT_DIR, keep_last=3, keep_model_final=True)
        
        val_result = None 
        if iteration % cfg.SOLVER.VAL_PERIOD == 0:  # 
             output_folder = getattr(cfg, "val_outpath", None)
             if output_folder in ("None", "none", "null", "NULL", ""):
                 output_folder = None
             if output_folder is None:
                 output_folder = getattr(cfg, "outpath", None)
             if output_folder in ("None", "none", "null", "NULL", ""):
                 output_folder = None
             if output_folder is None and cfg.OUTPUT_DIR:
                 output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", "val")
             if output_folder:
                 mkdir(output_folder)
             val_result = run_val(cfg, model, val_data_loaders, distributed, logger, output_folder=output_folder)
             if cfg.MODEL.RELATION_ON and output_folder:
                 mode = _get_sgg_mode(cfg)
                 try:
                     f1_target_ks = tuple(int(k) for k in cfg.TEST.RELATION.RECALL_K)
                 except Exception:
                     f1_target_ks = (50, 100, 200)
                 f1_values = []
                 dataset_names = tuple(cfg.DATASETS.VAL)
                 for dataset_name in dataset_names:
                     result_dict_path = os.path.join(output_folder, dataset_name, "result_dict.pytorch")
                     result_dict = _load_result_dict(result_dict_path)
                     if result_dict is None:
                         continue
                     f1_avg, per_k = _compute_f1_from_result_dict(result_dict, mode, f1_target_ks)
                     f1_values.append(f1_avg)
                     parts = []
                     for k in f1_target_ks:
                         v = per_k.get(int(k))
                         if not v:
                             continue
                         parts.append(
                             "F1@{}={:.4f} R={:.4f} mR={:.4f}".format(k, v["F1"], v["R"], v["mR"])
                         )
                     logger.info("[val][{}][{}] F1(avg)={:.4f} {}".format(mode, dataset_name, f1_avg, " ".join(parts)))

                 if len(f1_values) > 0:
                     f1_avg_all = float(sum(f1_values) / len(f1_values))
                     logger.info("[val][{}] F1(avg) over {} = {:.4f}".format(mode, dataset_names, f1_avg_all))
                     if f1_avg_all > best_f1_avg:
                         best_f1_avg = f1_avg_all
                         best_f1_iter = iteration
                         logger.info("[best] iteration={} F1(avg)={:.4f}".format(best_f1_iter, best_f1_avg))
                         if cfg.OUTPUT_DIR:
                             _cleanup_best_epoch_checkpoints(cfg.OUTPUT_DIR)
                             if cfg.Type != "CV":
                                 best_path = os.path.join(cfg.OUTPUT_DIR, "best_epoch_{}.pth".format(iteration))
                                 meta = _build_mmcv_checkpoint_meta()
                                 save_checkpoint(model, best_path, optimizer=optimizer, meta=meta)
                             else:
                                 last_checkpoint_file = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
                                 prev_last = None
                                 try:
                                     with open(last_checkpoint_file, "r") as f:
                                         prev_last = f.read()
                                 except Exception:
                                     prev_last = None
                                 checkpointer.save("best_epoch_{:07d}".format(iteration), **arguments)
                                 if prev_last is not None:
                                     try:
                                         with open(last_checkpoint_file, "w") as f:
                                             f.write(prev_last)
                                     except Exception:
                                         pass

             compress_opts = _get_inference_compress_options(cfg)
             if compress_opts["enabled"] and compress_opts["on_val"] and output_folder and get_rank() == 0:
                 try:
                     out_abs = os.path.abspath(output_folder)
                     infer_root = os.path.abspath(os.path.join(cfg.OUTPUT_DIR, "inference")) if cfg.OUTPUT_DIR else ""
                     if infer_root and out_abs.startswith(infer_root):
                         dataset_names_for_compress = tuple(cfg.DATASETS.VAL)
                         for ds_name in dataset_names_for_compress:
                             ds_dir = os.path.join(output_folder, ds_name)
                             if not os.path.isdir(ds_dir):
                                 continue
                             ds_size = _dir_size_bytes(ds_dir)
                             if ds_size < compress_opts["min_size_bytes"]:
                                 continue
                             _compress_dir_to_targz(
                                 ds_dir,
                                 ds_dir + ".tar.gz",
                                 compresslevel=compress_opts["compresslevel"],
                                 delete_after=compress_opts["delete_after"],
                                 logger=logger,
                             )
                 except Exception as e:
                     logger.info("[compress] failed: {}".format(e))
             synchronize()

        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                checkpointer.save("model_final", **arguments)
                break
        else:
            scheduler.step()
        torch.cuda.empty_cache() 

        if iteration == max_iter:
            break


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return model
 
        


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(),
        # otherwise the module will be in the test mode,
        # i.e., all self.training condition is set to False


def run_val(cfg, model, val_data_loaders, distributed, logger, m=None, ite=None, CCM=None, output_folder=None, vae=None, dataset_names=None):
    val = 1
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)

    dataset_names = cfg.DATASETS.VAL if dataset_names is None else dataset_names
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_output_folder = output_folder
        if output_folder:
            dataset_output_folder = os.path.join(output_folder, dataset_name)
            mkdir(dataset_output_folder)
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=dataset_output_folder,
            logger=logger,
            m=m,
            val=val,
            ite=ite,
            CCM = CCM,
            vae = vae
        )
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    torch.cuda.empty_cache()
    return val_result


def run_test(cfg, model, distributed, logger, m = None,CCM = None):
    val = 2
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
            val=val,
        )
        synchronize()
        compress_opts = _get_inference_compress_options(cfg)
        if compress_opts["enabled"] and compress_opts["on_test"] and output_folder and get_rank() == 0:
            try:
                out_abs = os.path.abspath(output_folder)
                infer_root = os.path.abspath(os.path.join(cfg.OUTPUT_DIR, "inference")) if cfg.OUTPUT_DIR else ""
                if infer_root and out_abs.startswith(infer_root):
                    out_size = _dir_size_bytes(output_folder)
                    if out_size >= compress_opts["min_size_bytes"]:
                        _compress_dir_to_targz(
                            output_folder,
                            output_folder + ".tar.gz",
                            compresslevel=compress_opts["compresslevel"],
                            delete_after=compress_opts["delete_after"],
                            logger=logger,
                        )
            except Exception as e:
                logger.info("[compress] failed: {}".format(e))
        synchronize()


def main(debug=False):  
 

    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default='/SGG_ToolKit/configs/e2e_relation_X_101_32_8_FPN_1x_trans__base.yaml',
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument("--local_rank", default=0)

    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument(
        "--log_name",
        default="log.txt",
        help="Do not test the final model",
        type=str,
    )

    parser.add_argument(
        "--mm_config",
        default='configs/RSOBB/STAR_obb_predcls_sgcls.py',
        help="Modify config options using the command-line",
        type=str,
    )

    parser.add_argument(
        "--mm_weight",
        default='PRE_WEI/OBB_Swin.pth',
        help="Modify config options using the command-line",
        type=str,

    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )





    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    local_rank = int(os.environ['LOCAL_RANK']) if "WORLD_SIZE" in os.environ else 0
    if args.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    try:
        from maskrcnn_benchmark.utils.imports import import_file
        from maskrcnn_benchmark.data.datasets.visual_genome import load_info

        paths_catalog = import_file(
            "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
        )
        DatasetCatalog = paths_catalog.DatasetCatalog
        dataset_names = cfg.DATASETS.TRAIN
        if isinstance(dataset_names, str):
            dataset_names = (dataset_names,)
        if dataset_names:
            data = DatasetCatalog.get(dataset_names[0], cfg)
            dict_file = data.get("args", {}).get("dict_file")
            if dict_file and os.path.exists(dict_file):
                ind_to_classes, ind_to_predicates, ind_to_attributes, *_ = load_info(dict_file)
                cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(ind_to_classes)
                cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES = len(ind_to_predicates)
                cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES = len(ind_to_attributes)
    except Exception:
        pass

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)


    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank(),filename=args.log_name)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
    logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    if getattr(cfg, "AUTO_LONGTAIL_IDS", False) and cfg.MODEL.RELATION_ON:
        _auto_update_longtail_ids(cfg, logger)
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    
    model = train(cfg, local_rank, args.distributed, logger, debug=debug, mmcf =  args.mm_config ,mmwei =  args.mm_weight)
    if not args.skip_test:
        if cfg.OUTPUT_DIR:
            best_ckpt = _find_best_epoch_checkpoint(cfg.OUTPUT_DIR)
            if best_ckpt and os.path.exists(best_ckpt):
                logger.info("Loading best checkpoint for test: {}".format(best_ckpt))
                model_to_load = model.module if args.distributed else model
                if cfg.Type == "CV":
                    DetectronCheckpointer(
                        cfg, model_to_load, save_dir=cfg.OUTPUT_DIR, save_to_disk=False, logger=logger
                    ).load(best_ckpt, with_optim=False)
                else:
                    load_checkpoint(model_to_load, best_ckpt, map_location="cpu", strict=False)
        run_test(cfg, model, args.distributed, logger)



if __name__ == "__main__":
    import sys

    print('running with system paths :', sys.path)
    main()

    


    
