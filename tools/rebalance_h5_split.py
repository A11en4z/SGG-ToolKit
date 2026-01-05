import argparse
import math
import random

import h5py
import numpy as np


def _safe_int_array(x):
    """把任意numpy数组安全转换为int64，避免uint/-1占位导致的溢出。"""
    arr = np.asarray(x)
    if arr.dtype.kind in ("u", "b"):
        return arr.astype(np.int64, copy=False)
    return arr.astype(np.int64, copy=False)


def _valid_image_mask(h5, filter_empty_rels=True, filter_empty_boxes=True):
    """计算每张图是否可用的mask（box与rel指针合法且非空）。"""
    num_images = int(h5["split"].shape[0])

    im_first_box = _safe_int_array(h5["img_to_first_box"][:])
    im_last_box = _safe_int_array(h5["img_to_last_box"][:])
    num_boxes = int(h5["labels"].shape[0]) if "labels" in h5 else -1

    has_box = im_first_box >= 0
    has_box &= im_last_box >= im_first_box
    if num_boxes >= 0:
        has_box &= im_first_box < num_boxes
        has_box &= im_last_box < num_boxes

    if not filter_empty_boxes:
        has_box = np.ones((num_images,), dtype=bool)

    if not filter_empty_rels or "relationships" not in h5:
        has_rel = np.ones((num_images,), dtype=bool)
        return has_box & has_rel

    num_rels = int(h5["relationships"].shape[0])
    im_first_rel = _safe_int_array(h5["img_to_first_rel"][:])
    im_last_rel = _safe_int_array(h5["img_to_last_rel"][:])

    has_rel = im_first_rel >= 0
    has_rel &= im_last_rel >= im_first_rel
    has_rel &= im_first_rel < num_rels
    has_rel &= im_last_rel < num_rels

    return has_box & has_rel


def _infer_num_classes(h5, key, col=0, min_size=1):
    """从H5里推断类别数（按max+1），用于bincount的minlength。"""
    if key not in h5:
        return min_size
    data = _safe_int_array(h5[key][:])
    if data.ndim == 1:
        mx = int(data.max(initial=0))
    else:
        mx = int(data[:, col].max(initial=0))
    return max(min_size, mx + 1)


def _per_image_histograms(h5, valid_mask, num_obj_classes, num_pred_classes):
    """为每张图构建对象类别与谓词类别的直方图特征。"""
    num_images = int(h5["split"].shape[0])

    im_first_box = _safe_int_array(h5["img_to_first_box"][:])
    im_last_box = _safe_int_array(h5["img_to_last_box"][:])
    labels = _safe_int_array(h5["labels"][:, 0]) if "labels" in h5 else None

    im_first_rel = _safe_int_array(h5["img_to_first_rel"][:])
    im_last_rel = _safe_int_array(h5["img_to_last_rel"][:])
    predicates = _safe_int_array(h5["predicates"][:, 0]) if "predicates" in h5 else None

    obj_hists = np.zeros((num_images, num_obj_classes), dtype=np.int32)
    pred_hists = np.zeros((num_images, num_pred_classes), dtype=np.int32)

    for i in range(num_images):
        if not valid_mask[i]:
            continue

        if labels is not None:
            b0 = int(im_first_box[i])
            b1 = int(im_last_box[i])
            if b0 >= 0 and b1 >= b0:
                obj = labels[b0 : b1 + 1]
                obj_hists[i] = np.bincount(obj, minlength=num_obj_classes).astype(np.int32, copy=False)

        if predicates is not None:
            r0 = int(im_first_rel[i])
            r1 = int(im_last_rel[i])
            if r0 >= 0 and r1 >= r0:
                pr = predicates[r0 : r1 + 1]
                pred_hists[i] = np.bincount(pr, minlength=num_pred_classes).astype(np.int32, copy=False)

    return obj_hists, pred_hists


def _normalize(x):
    """把向量按L1归一化为分布；若全零则原样返回。"""
    x = x.astype(np.float64, copy=False)
    s = float(x.sum())
    if s <= 0:
        return x
    return x / s


def _greedy_select(indices, vectors, target_dist, start_sum, k, rng):
    """用贪心法从候选indices里选k个，使累计分布逼近target_dist（L1距离）。"""
    remaining = set(int(i) for i in indices)
    selected = []

    cur_sum = start_sum.astype(np.float64, copy=True)
    cur_total = float(cur_sum.sum())
    dim = int(cur_sum.shape[0])

    def score(sum_vec, total):
        if total <= 0:
            return float("inf")
        dist = sum_vec / total
        return float(np.abs(dist - target_dist).sum())

    while len(selected) < k and remaining:
        best = None
        best_score = None

        remaining_list = list(remaining)
        rng.shuffle(remaining_list)

        for idx in remaining_list:
            v = vectors[idx]
            new_sum = cur_sum + v
            new_total = cur_total + float(v.sum())
            s = score(new_sum, new_total)
            if best_score is None or s < best_score:
                best = idx
                best_score = s

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)
        v = vectors[best]
        cur_sum += v
        cur_total += float(v.sum())

        if math.isinf(best_score):
            break

    return selected, cur_sum


def rebalance_split(
    h5_path,
    expand_test,
    ratio_train=6,
    ratio_val=2,
    ratio_test=2,
    use_ratio=False,
    filter_empty_rels=True,
    filter_empty_boxes=True,
    obj_weight=1.0,
    pred_weight=1.0,
    seed=0,
    write=False,
):
    """基于对象/谓词分布做split重划分：扩充test，并保证val数量与test一致。"""
    rng = random.Random(seed)

    with h5py.File(h5_path, "r+") as h5:
        split = _safe_int_array(h5["split"][:])
        num_images = int(split.shape[0])

        valid = _valid_image_mask(h5, filter_empty_rels=filter_empty_rels, filter_empty_boxes=filter_empty_boxes)

        num_obj = _infer_num_classes(h5, "labels", col=0, min_size=1)
        num_pred = _infer_num_classes(h5, "predicates", col=0, min_size=1)

        obj_h, pred_h = _per_image_histograms(h5, valid, num_obj_classes=num_obj, num_pred_classes=num_pred)

        vec = np.concatenate(
            [
                obj_h.astype(np.float64, copy=False) * float(obj_weight),
                pred_h.astype(np.float64, copy=False) * float(pred_weight),
            ],
            axis=1,
        )

        eligible = np.where(valid)[0].astype(np.int64, copy=False)
        if eligible.size == 0:
            raise RuntimeError("No eligible images after filtering; cannot rebalance.")

        target_dist = _normalize(vec[eligible].sum(axis=0))

        if use_ratio:
            denom = int(ratio_train) + int(ratio_val) + int(ratio_test)
            if denom <= 0:
                raise ValueError("Invalid ratio: sum must be > 0")

            total = int(eligible.size)
            desired_test = int(round(total * (float(ratio_test) / float(denom))))
            desired_val = int(round(total * (float(ratio_val) / float(denom))))
            desired_test = max(0, min(desired_test, total))
            desired_val = max(0, min(desired_val, total - desired_test))

            candidate_for_test = [int(i) for i in eligible.tolist()]
            test_selected, test_sum = _greedy_select(
                candidate_for_test,
                vec,
                target_dist,
                np.zeros((vec.shape[1],), dtype=np.float64),
                desired_test,
                rng,
            )
            final_test_set = set(int(i) for i in test_selected)
        else:
            existing_test_all = np.where(split == 2)[0]
            existing_test_valid = existing_test_all[valid[existing_test_all]]
            existing_test_valid_set = set(int(i) for i in existing_test_valid.tolist())

            target_test_size = int(existing_test_valid.size) + int(expand_test)
            target_test_size = min(target_test_size, int(eligible.size))

            candidate_for_test = [int(i) for i in eligible.tolist() if int(i) not in existing_test_valid_set]

            test_sum0 = (
                vec[list(existing_test_valid_set)].sum(axis=0)
                if existing_test_valid_set
                else np.zeros((vec.shape[1],), dtype=np.float64)
            )
            add_k = max(0, target_test_size - int(existing_test_valid.size))
            add_test, test_sum = _greedy_select(candidate_for_test, vec, target_dist, test_sum0, add_k, rng)

            final_test_set = set(existing_test_valid_set) | set(int(i) for i in add_test)

        remaining_after_test = [int(i) for i in eligible.tolist() if int(i) not in final_test_set]

        if use_ratio:
            val_target_size = min(int(desired_val), len(remaining_after_test))
        else:
            val_target_size = min(int(len(final_test_set)), len(remaining_after_test))
        test_dist = _normalize(test_sum)
        val_selected, _ = _greedy_select(
            remaining_after_test,
            vec,
            test_dist,
            np.zeros((vec.shape[1],), dtype=np.float64),
            val_target_size,
            rng,
        )
        final_val_set = set(int(i) for i in val_selected)

        new_split = split.copy()
        new_split[:] = 0
        new_split[list(final_val_set)] = 1
        new_split[list(final_test_set)] = 2

        old_counts = {int(k): int((split == k).sum()) for k in np.unique(split).tolist()}
        new_counts = {0: int((new_split == 0).sum()), 1: int((new_split == 1).sum()), 2: int((new_split == 2).sum())}

        report = {
            "num_images": num_images,
            "eligible_images": int(eligible.size),
            "old_split_counts": old_counts,
            "new_split_counts": new_counts,
            "final_test_size": int(len(final_test_set)),
            "final_val_size": int(len(final_val_set)),
        }

        if write:
            if "split" not in h5:
                raise RuntimeError("H5 does not contain 'split' dataset; cannot write.")
            h5["split"][:] = new_split.astype(h5["split"].dtype, copy=False)
            h5.flush()

        return report


def main():
    """命令行入口：输出重划分报告，并在--write时回写H5的split字段。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--expand-test", type=int, default=None)
    parser.add_argument("--use-ratio", action="store_true", default=False)
    parser.add_argument("--ratio-train", type=int, default=6)
    parser.add_argument("--ratio-val", type=int, default=2)
    parser.add_argument("--ratio-test", type=int, default=2)
    parser.add_argument("--filter-empty-rels", dest="filter_empty_rels", action="store_true")
    parser.add_argument("--no-filter-empty-rels", dest="filter_empty_rels", action="store_false")
    parser.add_argument("--filter-empty-boxes", dest="filter_empty_boxes", action="store_true")
    parser.add_argument("--no-filter-empty-boxes", dest="filter_empty_boxes", action="store_false")
    parser.add_argument("--obj-weight", type=float, default=1.0)
    parser.add_argument("--pred-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true", default=False)
    parser.set_defaults(filter_empty_rels=True, filter_empty_boxes=True)
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as h5:
        split = _safe_int_array(h5["split"][:])
        existing_test = int((split == 2).sum())
    expand_test = args.expand_test if args.expand_test is not None else existing_test

    report = rebalance_split(
        h5_path=args.h5,
        expand_test=expand_test,
        ratio_train=args.ratio_train,
        ratio_val=args.ratio_val,
        ratio_test=args.ratio_test,
        use_ratio=args.use_ratio,
        filter_empty_rels=args.filter_empty_rels,
        filter_empty_boxes=args.filter_empty_boxes,
        obj_weight=args.obj_weight,
        pred_weight=args.pred_weight,
        seed=args.seed,
        write=args.write,
    )
    print(json_dumps(report))


def json_dumps(x):
    """用UTF-8友好的方式打印json（便于日志与复制）。"""
    import json

    return json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
