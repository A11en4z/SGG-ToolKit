import argparse
import glob
import json
import os
import random

import numpy as np


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dict-file", default="/gz-data/DIOR_STAR/DIOR-SGG-dicts-with-attri.json")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--tsne-max", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_predicate_names(dict_file):
    """读取谓词名称映射。"""
    if not dict_file or not os.path.exists(dict_file):
        return None
    with open(dict_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "idx_to_predicate" in data:
            idx_to_pred = data["idx_to_predicate"]
            if isinstance(idx_to_pred, dict):
                max_k = max(int(k) for k in idx_to_pred.keys())
                names = [""] * (max_k + 1)
                for k, v in idx_to_pred.items():
                    names[int(k)] = v
                return names
            if isinstance(idx_to_pred, list):
                return idx_to_pred
        if "predicate_to_idx" in data:
            pred_to_idx = data["predicate_to_idx"]
            names = [""] * (max(int(v) for v in pred_to_idx.values()) + 1)
            for k, v in pred_to_idx.items():
                names[int(v)] = k
            return names
    return None


def collect_npz(vis_dir, max_samples):
    """收集可视化缓存的npz数据。"""
    files = sorted(glob.glob(os.path.join(vis_dir, "bckm_vis_*.npz")))
    rel_logits_raw = []
    rel_logits_bias = []
    rel_pred = []
    rel_rep1 = []
    rel_rep2 = []
    rel_gcn = []
    rel_fusion = []
    t_u0 = []
    t_u1 = []
    attn_sum = []
    rel_gt = []
    attn_count = 0
    total = 0
    for f in files:
        data = np.load(f)
        if "rel_logits_raw" in data:
            n = data["rel_logits_raw"].shape[0]
            remaining = max_samples - total
            if remaining <= 0:
                break
            take = min(n, remaining)
            rel_logits_raw.append(data["rel_logits_raw"][:take])
            rel_logits_bias.append(data["rel_logits_bias"][:take])
            rel_pred.append(data["rel_pred"][:take])
            if "rel_gt" in data:
                rel_gt.append(data["rel_gt"][:take])
            if "rel_rep1" in data:
                rel_rep1.append(data["rel_rep1"][:take])
            if "rel_rep2" in data:
                rel_rep2.append(data["rel_rep2"][:take])
            if "rel_gcn" in data:
                rel_gcn.append(data["rel_gcn"][:take])
            if "rel_fusion" in data:
                rel_fusion.append(data["rel_fusion"][:take])
            if "t_u0" in data:
                t_u0.append(data["t_u0"][:take])
            if "t_u1" in data:
                t_u1.append(data["t_u1"][:take])
            if "attn_sum" in data:
                attn_sum.append(data["attn_sum"][:take])
            if "attn_count" in data:
                attn_count += int(data["attn_count"][0])
            total += take
    if total == 0:
        return None
    return dict(
        rel_logits_raw=np.concatenate(rel_logits_raw, axis=0),
        rel_logits_bias=np.concatenate(rel_logits_bias, axis=0),
        rel_pred=np.concatenate(rel_pred, axis=0),
        rel_gt=np.concatenate(rel_gt, axis=0) if rel_gt else None,
        rel_rep1=np.concatenate(rel_rep1, axis=0) if rel_rep1 else None,
        rel_rep2=np.concatenate(rel_rep2, axis=0) if rel_rep2 else None,
        rel_gcn=np.concatenate(rel_gcn, axis=0) if rel_gcn else None,
        rel_fusion=np.concatenate(rel_fusion, axis=0) if rel_fusion else None,
        t_u0=np.concatenate(t_u0, axis=0) if t_u0 else None,
        t_u1=np.concatenate(t_u1, axis=0) if t_u1 else None,
        attn_sum=np.concatenate(attn_sum, axis=0) if attn_sum else None,
        attn_count=attn_count,
    )


def softmax_np(x):
    """计算numpy版softmax。"""
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)


def plot_bias_effect(data, out_dir, pred_names):
    """绘制偏置前后置信度变化与类别分布。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_prob = softmax_np(data["rel_logits_raw"])
    bias_prob = softmax_np(data["rel_logits_bias"])
    raw_max = np.max(raw_prob, axis=1)
    bias_max = np.max(bias_prob, axis=1)
    diff = bias_max - raw_max
    plt.figure(figsize=(6, 4))
    rel_gt = data.get("rel_gt", None)
    if rel_gt is not None and rel_gt.shape[0] == diff.shape[0]:
        bias_pred = np.argmax(bias_prob, axis=1)
        correct_bias = bias_pred == rel_gt
        plt.hist(diff[correct_bias], bins=50, color="#4C72B0", alpha=0.75, label="bias正确")
        plt.hist(diff[~correct_bias], bins=50, color="#C44E52", alpha=0.55, label="bias错误")
        plt.legend()
    else:
        plt.hist(diff, bins=50, color="#4C72B0", alpha=0.85)
    plt.title("Bias max-prob shift")
    plt.xlabel("max prob (bias) - max prob (raw)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bias_shift_hist.png"), dpi=150)
    plt.close()

    def entropy(p):
        p = np.clip(p, 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    ent_raw = entropy(raw_prob)
    ent_bias = entropy(bias_prob)
    raw_pred = np.argmax(raw_prob, axis=1)
    bias_pred = np.argmax(bias_prob, axis=1)
    flips = bias_pred != raw_pred
    flips_total = int(np.sum(flips))
    flips_to_bg = int(np.sum(flips & (bias_pred == 0)))
    flips_from_bg = int(np.sum(flips & (raw_pred == 0) & (bias_pred > 0)))
    flips_fg2fg = int(np.sum(flips & (raw_pred > 0) & (bias_pred > 0)))
    fg_rate_raw = float(np.mean(raw_pred > 0))
    fg_rate_bias = float(np.mean(bias_pred > 0))
    proxy = {
        "mean_max_prob_raw": float(np.mean(raw_max)),
        "mean_max_prob_bias": float(np.mean(bias_max)),
        "mean_max_prob_delta": float(np.mean(bias_max - raw_max)),
        "mean_entropy_raw": float(np.mean(ent_raw)),
        "mean_entropy_bias": float(np.mean(ent_bias)),
        "mean_entropy_delta": float(np.mean(ent_bias - ent_raw)),
        "flips_total": flips_total,
        "flips_to_bg": flips_to_bg,
        "flips_from_bg": flips_from_bg,
        "flips_fg_to_fg": flips_fg2fg,
        "fg_rate_raw": fg_rate_raw,
        "fg_rate_bias": fg_rate_bias,
        "samples": int(raw_prob.shape[0]),
    }
    with open(os.path.join(out_dir, "bias_shift_proxy.json"), "w", encoding="utf-8") as f:
        json.dump(proxy, f, ensure_ascii=False, indent=2)

    pred = data["rel_pred"].astype(np.int64)
    pred = pred[pred > 0]
    if pred.size == 0:
        return
    num_cls = int(np.max(pred)) + 1 if pred.size else 0
    counts = np.bincount(pred, minlength=num_cls)
    topk = min(20, num_cls)
    order = np.argsort(counts)[::-1][:topk]
    labels = [str(i) for i in order]
    if pred_names and len(pred_names) > max(order):
        labels = [pred_names[i] if pred_names[i] else str(i) for i in order]
    plt.figure(figsize=(8, 4))
    plt.bar(range(topk), counts[order], color="#55A868")
    plt.xticks(range(topk), labels, rotation=45, ha="right")
    plt.title("Top predicate counts (bias)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predicate_top_counts.png"), dpi=150)
    plt.close()
    if rel_gt is not None and rel_gt.shape[0] == raw_prob.shape[0]:
        raw_pred = np.argmax(raw_prob, axis=1)
        bias_pred = np.argmax(bias_prob, axis=1)
        correct_raw = raw_pred == rel_gt
        correct_bias = bias_pred == rel_gt
        fg = rel_gt > 0
        metrics = {
            "acc_raw": float(np.mean(correct_raw)),
            "acc_bias": float(np.mean(correct_bias)),
            "acc_raw_fg": float(np.mean(correct_raw[fg])) if np.any(fg) else None,
            "acc_bias_fg": float(np.mean(correct_bias[fg])) if np.any(fg) else None,
            "flip_to_correct": int(np.sum((~correct_raw) & correct_bias)),
            "flip_to_wrong": int(np.sum(correct_raw & (~correct_bias))),
            "total": int(rel_gt.shape[0]),
            "total_fg": int(np.sum(fg)),
        }
        with open(os.path.join(out_dir, "bias_shift_eval.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


def plot_attention(data, out_dir):
    """绘制注意力均值热力图（美化版）。
    
    说明：
    - 模型的注意力发生在3个token（tS/tO/tU）之间，因此原始权重矩阵为3x3。
    - 为提升可读性，这里对3x3矩阵进行双线性上采样到更高分辨率进行展示，同时保留坐标刻度。
    - 若需与原图叠加的空间热力图，需要额外导出union特征的空间激活图与图像路径，当前脚本不包含该数据。
    """
    if data["attn_sum"] is None or data["attn_count"] <= 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    attn_sum = data["attn_sum"]
    attn_count = float(data["attn_count"])
    attn_mean = attn_sum / max(attn_count, 1.0)
    attn_mean = attn_mean.mean(axis=0)
    # 上采样到更细网格以改善观感（例如放大到 48x48）
    upscale = 16
    attn_mean_up = zoom(attn_mean, (upscale, upscale), order=1)  # 双线性
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_mean_up, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    # 仍以 token 级别显示刻度
    ticks = [0, 1, 2]
    tick_pos = [int(t * upscale + upscale // 2) for t in ticks]
    plt.xticks(tick_pos, ["tS", "tO", "tU"])
    plt.yticks(tick_pos, ["tS", "tO", "tU"])
    plt.title("BCKM attention mean (upsampled)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bckm_attn_heatmap.png"), dpi=150)
    plt.close()
    rel_labels = data.get("rel_gt", None)
    if rel_labels is None or rel_labels.shape[0] != attn_sum.shape[0]:
        rel_labels = data.get("rel_pred", None)
    if rel_labels is None or rel_labels.shape[0] != attn_sum.shape[0]:
        return
    rel_labels = rel_labels.astype(np.int64)
    rel_labels = rel_labels[rel_labels > 0]
    if rel_labels.size == 0:
        return
    counts = np.bincount(rel_labels)
    order = np.argsort(counts)[::-1]
    order = order[order > 0]
    topk = min(9, order.shape[0])
    if topk <= 0:
        return
    order = order[:topk]
    import math
    ncols = 3
    nrows = int(math.ceil(topk / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.2))
    axes = np.array(axes).reshape(-1)
    for i, cls_id in enumerate(order):
        mask = data.get("rel_gt", None)
        if mask is None or mask.shape[0] != attn_sum.shape[0]:
            mask = data.get("rel_pred", None)
        mask = (mask == cls_id)
        if np.any(mask):
            attn_cls = (attn_sum[mask] / max(attn_count, 1.0)).mean(axis=0)
        else:
            attn_cls = attn_mean
        ax = axes[i]
        attn_cls_up = zoom(attn_cls, (upscale, upscale), order=1)
        ax.imshow(attn_cls_up, cmap="viridis", interpolation="nearest")
        ticks = [0, 1, 2]
        tick_pos = [int(t * upscale + upscale // 2) for t in ticks]
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(["tS", "tO", "tU"])
        ax.set_yticklabels(["tS", "tO", "tU"])
        ax.set_title(str(cls_id))
    for j in range(topk, axes.shape[0]):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bckm_attn_heatmap_by_pred.png"), dpi=150)
    plt.close()


def run_tsne(data, out_dir, pred_names, tsne_max, seed):
    """运行t-SNE或PCA并保存散点图。"""
    rel_labels = data.get("rel_gt", None)
    if rel_labels is None:
        rel_labels = data.get("rel_pred", None)
    if rel_labels is None:
        return
    rel_labels = rel_labels.astype(np.int64)
    feature_items = [
        ("gcn", data.get("rel_gcn", None), None, "t-SNE/PCA: GCN relation features"),
        ("transformer", data.get("rel_fusion", None), None, "t-SNE/PCA: Transformer fusion features"),
        ("rel_rep1", data.get("rel_rep1", None), None, "t-SNE/PCA: rel_rep1"),
        ("rel_rep2", data.get("rel_rep2", None), None, "t-SNE/PCA: rel_rep2"),
        ("logits_raw", data.get("rel_logits_raw", None), None, "t-SNE/PCA: raw logits"),
        ("logits_bias", data.get("rel_logits_bias", None), None, "t-SNE/PCA: bias logits"),
    ]
    for name, feature, proto_use, title in feature_items:
        if feature is None:
            continue
        n = feature.shape[0]
        if n == 0:
            continue
        labels = rel_labels
        if labels.shape[0] > n:
            labels = labels[:n]
        random.seed(seed)
        if n > tsne_max:
            idx = np.array(random.sample(range(n), tsne_max), dtype=np.int64)
        else:
            idx = np.arange(n, dtype=np.int64)
        feature = feature[idx]
        labels = labels[idx]
        if np.any(labels > 0):
            cls_ids = np.unique(labels[labels > 0])
            per_class = max(10, min(200, tsne_max // max(1, cls_ids.shape[0])))
            keep = []
            for cls_id in cls_ids:
                sub_idx = np.where(labels == cls_id)[0]
                if sub_idx.size == 0:
                    continue
                if sub_idx.size > per_class:
                    sub_idx = np.array(random.sample(list(sub_idx), per_class), dtype=np.int64)
                keep.append(sub_idx)
            if keep:
                keep = np.concatenate(keep, axis=0)
                feature = feature[keep]
                labels = labels[keep]
        features = feature
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=2, init="pca", random_state=seed, perplexity=30)
            emb = tsne.fit_transform(features)
        except Exception:
            mean = np.mean(features, axis=0, keepdims=True)
            centered = features - mean
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            emb = centered @ vh[:2].T
        rel_emb = emb[: feature.shape[0]]
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.scatter(rel_emb[:, 0], rel_emb[:, 1], s=8, c=labels[: rel_emb.shape[0]], cmap="tab20", alpha=0.6)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tsne_{name}.png"), dpi=150)
        plt.close()
        save_tsne_with_legend(rel_emb, labels[: rel_emb.shape[0]], pred_names, out_dir, name, seed)


def save_tsne_with_legend(rel_emb, labels, pred_names, out_dir, name, seed):
    """输出带图例的t-SNE图到独立目录。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    legend_dir = os.path.join(out_dir, "tsne_legend")
    os.makedirs(legend_dir, exist_ok=True)
    labels = labels.astype(np.int64)
    uniq = np.unique(labels[labels > 0])
    if uniq.size == 0:
        return
    try:
        cmap = plt.get_cmap("turbo", int(uniq.size))
    except Exception:
        cmap = plt.get_cmap("hsv", int(uniq.size))
    color_map = {int(cls_id): cmap(i) for i, cls_id in enumerate(uniq)}
    point_colors = []
    for cls_id in labels:
        if int(cls_id) in color_map:
            point_colors.append(color_map[int(cls_id)])
        else:
            point_colors.append((0.6, 0.6, 0.6, 0.35))
    plt.figure(figsize=(9.5, 6.5))
    plt.scatter(rel_emb[:, 0], rel_emb[:, 1], s=10, c=point_colors, alpha=0.75)
    handles = []
    for cls_id in uniq:
        if pred_names and int(cls_id) < len(pred_names) and pred_names[int(cls_id)]:
            label = pred_names[int(cls_id)]
        else:
            label = str(int(cls_id))
        handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6, color=color_map[int(cls_id)], label=label))
    plt.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        ncol=1,
    )
    plt.title(f"t-SNE/PCA: {name} (legend)")
    plt.tight_layout(rect=(0, 0, 0.78, 1))
    plt.savefig(os.path.join(legend_dir, f"tsne_{name}_legend.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """主入口：生成各类可视化图表。"""
    args = parse_args()
    data = collect_npz(args.vis_dir, args.max_samples)
    if data is None:
        return
    out_dir = args.out_dir or os.path.join(args.vis_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    pred_names = load_predicate_names(args.dict_file)
    plot_bias_effect(data, out_dir, pred_names)
    plot_attention(data, out_dir)
    run_tsne(data, out_dir, pred_names, args.tsne_max, args.seed)


if __name__ == "__main__":
    main()
