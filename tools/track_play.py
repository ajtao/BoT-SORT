import sys
import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import cv2
import torch
from loguru import logger
import numpy as np

from tools.vb_demo import Predictor
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, setup_logger
from yolox.utils.visualize import plot_tracking_mc

from tracker.vball_sort import VbSORT
from tracker.tracking_utils.timer import Timer

from vtrak.match_config import Match
from vtrak.config import cfg
from vtrak.dataloader import LoadVideo
from vtrak.track_utils import TrackWriter


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=str, help="test img size (w,h)")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--viz-dets", action='store_true', help='visualize detections')
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.6, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="none", type=str,
                        help=("cmc method: files (Vidstab GMC) | orb | ecc"
                              "But we shouldn't need CMC with vb, so defaulting to off"))
    # ReID
    # parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use reid model")
    parser.add_argument("--sports-reid", action='store_true')
    parser.add_argument("--no-reid", dest="with_reid", default=True, action='store_false', help="don't use reid model")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    parser.add_argument('--detect-confusions', action='store_true')

    parser.add_argument(
        "--play-vid", required=True, default=None,
        help="name of a single video to evaluate"
    )
    parser.add_argument(
        "--tag", default=None, help="tag outputdir"
    )
    return parser


def track_play(dataloader, predictor, current_time, args, result_filename, result_root):
    """
    Meant to be called upon a single play to write tracking information for.
    Writes out CVAT-compatible tracking information, dumps frames.
    """
    output_video_path = osp.join(result_root, 'tracked.mp4')
    vid_writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
        dataloader.frame_rate, (dataloader.width, dataloader.height)
    )

    img_dir = osp.join(result_root, 'images')
    os.makedirs(img_dir, exist_ok=True)
    tracker = VbSORT(args, frame_rate=args.fps)

    # ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(tracker.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    timer = Timer()
    start_time = time.time()

    frame_id = 0
    results_wr = open(result_filename, 'w')
    dets_wr = open(osp.join(result_root, 'dets.csv'), 'w')
    header = 'frame,id,x1,y1,w,h,play,class,is_jumping,ori_fnum,dx,dy,tlen\n'
    results_wr.write(header)

    print_flag = True
    last_play = -1
    trk_writer = TrackWriter(result_root)

    for frame_id, (vid_fnum, play_num, frame) in enumerate(dataloader):
        if play_num != last_play:
            logger.info(f'Start play {play_num} frame {vid_fnum}')
        last_play = play_num
        if frame_id % 20 == 0:
            cur_time = time.time()
            fps = frame_id / (cur_time - start_time)
            logger.info('Processing play {} frame {} ({:.2f} fps)'.format(
                play_num, vid_fnum, fps))

        if frame_id == 0:
            video_frame_fn = result_filename.replace('csv', 'png')
            cv2.imwrite(video_frame_fn, frame)

        # Run tracker
        frame_print = None

        # Detect objects
        outputs, img_info = predictor.inference(frame, timer, dump_input=frame_print)
        dets = outputs[0]
        scale = min(exp.test_size[0] / float(img_info['height']),
                    exp.test_size[1] / float(img_info['width']))

        if print_flag:
            w, h = img_info['width'], img_info['height']
            print(f'img_size w,h = {w},{h}, scale={scale:2.4f}')
            print_flag = False

        if dets is not None:
            # scale bbox predictions according to image size
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale
            online_targets = tracker.update(detections, img_info["raw_img"],
                                            frame_print=frame_print)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_jumping = []
            online_nearfar = []
            for trk in online_targets:
                trk_writer.write(trk, vid_fnum)
                tlwh = trk.tlwh
                tid = trk.track_id
                dxdy = trk.dxdy
                tlen = trk.tracklet_len
                nearfar = trk.cls
                is_jumping = trk.jumping
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(trk.score)
                    online_jumping.append(is_jumping)
                    online_nearfar.append(nearfar)
                    csv_str = (f'{vid_fnum},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},'
                               f'{tlwh[3]:.2f},{play_num},{nearfar},{is_jumping},{vid_fnum},'
                               f'{dxdy[0]},{dxdy[1]},{tlen}\n')
                    results_wr.write(csv_str)
                    dets_str = (f'{vid_fnum},-1,{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},'
                                f'{tlwh[3]:.2f},{trk.score:0.2f},-1,-1\n')
                    dets_wr.write(dets_str)

            timer.toc()
            online_im = plot_tracking_mc(
                image=img_info['raw_img'],
                tlwhs=online_tlwhs,
                obj_ids=online_ids,
                jumping=online_jumping,
                nearfar=online_nearfar,
                num_classes=tracker.num_classes,
                frame_id=vid_fnum,
                fps=1. / timer.average_time,
                play_num=play_num,
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        vid_writer.write(online_im)

        img_fn = osp.join(img_dir, f'{vid_fnum:06d}.jpg')
        cv2.imwrite(img_fn, img_info['raw_img'])

    trk_writer.finish()
    logger.info(f"Saved tracking results to {result_filename}")


def viz_dets(dataloader, predictor, current_time, args, result_root):
    timer = Timer()
    start_time = time.time()
    frame_id = 0
    print_flag = True
    last_play = -1

    output_video_path = osp.join(result_root, 'dets.mp4')
    vid_writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
        dataloader.frame_rate, (dataloader.width, dataloader.height)
    )

    for frame_id, (vid_fnum, play_num, frame) in enumerate(dataloader):
        if play_num != last_play:
            logger.info(f'Start play {play_num} frame {vid_fnum}')
        last_play = play_num
        if frame_id % 20 == 0:
            cur_time = time.time()
            fps = frame_id / (cur_time - start_time)
            logger.info('Processing play {} frame {} ({:.2f} fps)'.format(
                play_num, vid_fnum, fps))

        # Run tracker
        frame_print = None

        # Detect objects
        outputs, img_info = predictor.inference(frame, timer, dump_input=frame_print)
        dets = outputs[0]
        scale = min(exp.test_size[0] / float(img_info['height']),
                    exp.test_size[1] / float(img_info['width']))

        if print_flag:
            w, h = img_info['width'], img_info['height']
            print(f'img_size w,h = {w},{h}, scale={scale:2.4f}')
            print_flag = False

        if dets is not None:
            # scale bbox predictions according to image size
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale

            if len(detections):
                bboxes = detections[:, :4]  # [num_players, [x1,y1,x2,y2]]
                scores = detections[:, 4]  # [num_players]
                if detections.shape[1] == 6:
                    classes = detections[:, 5]
                elif detections.shape[1] == 7:
                    scores2 = detections[:, 5]
                    scores *= scores2
                    classes = detections[:, 6]
                else:
                    raise

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_jumping = []
            online_nearfar = []
            for pl_idx, (bbox, score, cls) in enumerate(zip(bboxes, scores, classes)):
                tlwh = (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
                online_tlwhs.append(tlwh)
                online_ids.append(pl_idx)
                online_scores.append(score)
                online_jumping.append(0)
                online_nearfar.append(cls==0)

            timer.toc()
            online_im = plot_tracking_mc(
                image=img_info['raw_img'],
                tlwhs=online_tlwhs,
                obj_ids=online_ids,
                jumping=online_jumping,
                nearfar=online_nearfar,
                num_classes=0,
                frame_id=vid_fnum,
                fps=1. / timer.average_time,
                play_num=play_num,
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        vid_writer.write(online_im)


def setup_volleyvision(args):
    match_name = osp.basename(osp.dirname(osp.dirname(args.play_vid)))
    play_dir_basename = osp.basename(osp.dirname(args.play_vid))

    if args.tag is not None:
        result_root = osp.join(cfg.output_root, 'BotSort', args.tag,
                               match_name, play_dir_basename)
    else:
        result_root = osp.join(cfg.output_root, 'BotSort',
                               match_name, play_dir_basename)
    os.makedirs(result_root, exist_ok=True)
    setup_logger(result_root, filename="log.log")

    vid_basename = osp.splitext(osp.basename(args.play_vid))[0]
    result_filename = os.path.join(result_root, f'{vid_basename}.csv')

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if args.tsize is not None:
        img_size = [int(x) for x in args.tsize.split(',')]
    else:
        img_size = None

    print(f'Inference img_size {img_size}')

    dataloader = LoadVideo(args.play_vid, img_size)
    return result_root, result_filename, dataloader


def main(exp, args):
    result_root, result_filename, dataloader = setup_volleyvision(args)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = [int(x) for x in args.tsize.split(',')]

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(result_root, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        model_dir = ('/mnt/f/output/ByteTrack/YOLOX_outputs/yolox_x_fullcourt_'
                     'v5bytetrack-with-bad-touches-trt')
        model_dir = '/mnt/f/output/ByteTrack/YOLOX_outputs/dbg-trt_bs1'
        trt_file = osp.join(model_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), f"TensorRT model {trt_file} is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.viz_dets:
        viz_dets(dataloader, predictor, current_time, args, result_root)
    else:
        track_play(dataloader, predictor, current_time, args, result_filename,
                   result_root)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)
