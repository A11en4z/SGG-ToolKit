# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
import sys

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("TORCH_HOME", "/data")
    os.environ.setdefault("TORCH_MODEL_ZOO", "/data")

    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="OpenCV camera index",
    )
    parser.add_argument(
        "--input-image",
        default=None,
        help="Path to a single image file for inference",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save visualization when using --input-image",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show OpenCV window (requires GUI)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override cfg.MODEL.DEVICE (e.g. cpu, cuda)",
    )
    parser.add_argument(
        "--config-file",
        default="/data/SGG/SGG-ToolKit/configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.device is not None:
        cfg.defrost()
        cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    if args.input_image is not None:
        img = cv2.imread(args.input_image)
        if img is None:
            raise RuntimeError(f"Cannot read image: {args.input_image}")
        composite = coco_demo.run_on_opencv_image(img)
        if args.output is not None:
            ok = cv2.imwrite(args.output, composite)
            if not ok:
                raise RuntimeError(f"Cannot write output image: {args.output}")
        if args.show:
            cv2.imshow("COCO detections", composite)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    cam = cv2.VideoCapture(args.camera_id)
    if not cam.isOpened():
        sys.stderr.write(f"Cannot open camera with index {args.camera_id}\n")
        return
    while True:
        start_time = time.time()
        ret_val, img = cam.read()
        if not ret_val or img is None:
            break
        composite = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        if args.show:
            cv2.imshow("COCO detections", composite)
            if cv2.waitKey(1) == 27:
                break
    cam.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
