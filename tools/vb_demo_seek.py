import argparse
import os
import os.path as osp
import time
import csv
import json
from queue import Queue
from collections import defaultdict

import contextlib
import cv2
import torch
import torchvision.transforms as transforms
from torch.cuda.amp import autocast

import ffmpegcv

from loguru import logger
import pandas as pd
import numpy as np
from tqdm import tqdm

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, setup_logger
from yolox.utils.visualize import plot_tracking_mc

from tracker.vball_sort import VbSORT
from tracker.tracking_utils.timer import Timer

from vtrak.config import cfg
from vtrak.vball_misc import run_ffprobe, read_play_frames, VidRdSeek
from vtrak.track_utils import TrackWriter
from vtrak.yolox_utils import Predictor


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


# context manager to help keep track of ranges of time, using NVTX
@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--tracker", default="bot-sort", help='deprecated')
    parser.add_argument("--plays-json", help='path to ball play frames json')
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device_name", default="cuda", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=str, help="test img size (w,h)")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--save-vid", action="store_true", help="write output video")
    parser.add_argument("--write-mot", action="store_true", help="write mot format")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    parser.add_argument("--det-thr", default=0, help='thresh for detection file')

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
    parser.add_argument("--no-reid", dest="with_reid", default=True, action='store_false', help="don't use reid model")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    parser.add_argument(
        "--unsquashed", action='store_true'
    )
    parser.add_argument(
        "--no-dv", action='store_true'
    )
    parser.add_argument(
        "--play-vid", default=None, required=True, help="name of a single video to evaluate"
    )
    parser.add_argument(
        "--match-name", default=None, help="match name"
    )
    parser.add_argument(
        "--view", default=None, help="view"
    )
    parser.add_argument(
        "--outdir", default=None, required=True, help="output dir"
    )
    parser.add_argument(
        "--max-plays", default=None, type=int, help="max plays"
    )
    parser.add_argument(
        "--max-frames", default=None, type=int, help="max frames"
    )
    parser.add_argument(
        "--start-pad", type=int, default=2,
    )
    parser.add_argument(
        "--end-pad", type=int, default=2,
    )
    parser.add_argument(
        "--prof", action='store_true'
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MyTransformCLS(torch.nn.Module):
    def __init__(self):
        super(MyTransformCLS, self).__init__()
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, img):
        img_tensor = self.to_tensor(img).cuda()
        norm_tensor = self.norm(img_tensor)
        input_batch = norm_tensor.unsqueeze(0)
        return input_batch


def separate_corners(detections, num_classes):
    """
    Strip corner boxes out of the detections since the player tracker doesn't care about them.
    """
    # These classes must match category order in the player detection dataset json
    if num_classes == 7:
        cls_name = [
            "near_nojump",
            "near_jump",
            "far_nojump",
            "far_jump",
            "upper",
            "mid",
            "lower"
        ]
    elif num_classes == 5:
        cls_name = [
            "near_nojump",
            "near_jump",
            "far_nojump",
            "far_jump",
            "ball"
        ]
        pass
    else:
        raise

    # expect this format for detection output:
    #   detections[:4] -> bboxes
    #   detections[4:6] = scores
    #   detections[6] = full class
    assert len(detections[0]) == 7  # yolox output
    corner_classes = ['upper', 'mid', 'lower']
    corners = {}
    players = []
    for det in detections:
        cls = int(det[6])
        if cls_name[cls] in corner_classes:
            corners[cls_name[cls]] = det
        elif cls_name[cls] == 'ball':
            continue
        elif 'near' in cls_name[cls] or 'far' in cls_name[cls]:
            # tracker should only see player classes now
            players.append(det)
        else:
            raise

    if len(players):
        players = np.stack(players)

    return players, corners


class CornerWriter():
    # Write court corners out, average over a window of the last 30 observations
    def __init__(self, out_fn, img_w, fps, det_thr=0.6, avg_window=30):
        fp = open(out_fn, 'w')
        self.writer = csv.writer(fp)
        self.mid_x = img_w // 2
        self.fps = int(fps)
        self.det_thr = det_thr
        self.writer.writerow(['fnum', 'ul_x', 'ul_y', 'ml_x', 'ml_y', 'ur_x', 'ur_y', 'mr_x', 'mr_y'])
        self.qs = {}
        self.q_names = ['left', 'right']
        for q_name in self.q_names:
            self.qs[q_name] = Queue(maxsize=avg_window)

    def update(self, fnum, corners):
        # Update the queue every fnum, but only write a new corner every second,
        # to save on filesize and also we don't need that level of precision
        for _cls, det in corners.items():
            if _cls != 'upper':
                continue
            bbox = det[:4]
            score = det[5] * det[6]
            if score < self.det_thr:
                # discard low confidence corners
                continue
            assert len(det) == 7  # yolox

            # assign corner side based on heuristic
            x1, y1, x2, y2 = bbox
            if x1 >= self.mid_x:
                cls = 'right'
            else:
                cls = 'left'
            assert cls in self.qs
            if self.qs[cls].full():
                self.qs[cls].get()
            self.qs[cls].put(bbox)

        if not np.all(np.array([q.qsize() for q in self.qs.values()])):
            # don't write out corner unless we have entries in all queues
            return

        if fnum % self.fps != 0:
            # only write out on second boundary
            return

        # Compute average bboxes across time window
        means = {}
        for q_name in self.q_names:
            # calculate mean coordinates across window
            vals = list(self.qs[q_name].queue)
            means[q_name] = np.mean(np.stack(vals), axis=0).astype(np.int32)

        # Court:
        #         +==============+
        #        /     court      \
        #       /     far side     \
        #      /                    \
        #     +======================+
        #    /        court           \
        #   /       near side          \
        #  /                            \
        # +==============================+
        #
        # Upper corner bboxes to court coordinates:
        #        ul              ur
        #     +---+===============+---+
        #     |  /|               |\  |
        #     | / |               | \ |
        #     |/  |               |  \|
        #  ml +===+===============+===+ mr
        #    /                         \
        #   /                           \

        row = [fnum]
        x1, y1, x2, y2 = means['left']
        row.extend([x2, y1])  # ul
        row.extend([x1, y2])  # ml
        x1, y1, x2, y2 = means['right']
        row.extend([x1, y1])  # ur
        row.extend([x2, y2])  # mr

        self.writer.writerow(row)


def imageflow_demo(predictor, current_time, args, exp):
    tracker = VbSORT(args, frame_rate=args.fps)

    # ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(tracker.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    timer = Timer()

    # Unified output
    os.makedirs(args.outdir, exist_ok=True)
    result_filename = os.path.join(args.outdir, 'tracker.csv')
    output_video_path = osp.join(args.outdir, 'tracker.mp4')

    dets_wr = open(osp.join(args.outdir, 'yolox.csv'), 'w')
    results_wr = open(result_filename, 'w')

    vid_info = run_ffprobe(args.play_vid)
    fps = vid_info.fps
    gpu_id = torch.cuda.current_device()
    logger.info(f'gpu_id {gpu_id}')
    print(f'\n\ngpu_id {gpu_id}\n\n')
    if args.save_vid:
        vid_writer = ffmpegcv.noblock(ffmpegcv.VideoWriterNV,
                                      output_video_path,
                                      codec='hevc',
                                      bitrate='4000k',
                                      fps=fps)

    if args.plays_json is None or not osp.isfile(args.plays_json):
        print('WARNING: args.plays_json is invalid, so skipping it')
        play_frames = None
        first_fnum = 1000
    else:
        play_frames = read_play_frames(args.plays_json)
        first_fnum = list(play_frames.keys())[0]
    vid_rd = VidRdSeek(args.play_vid, list(play_frames.keys()))

    # scale back to original image size
    native_img_size = [vid_rd.width, vid_rd.height]
    scale_x = exp.test_size[1] / float(vid_rd.width)
    scale_y = exp.test_size[0] / float(vid_rd.height)

    if args.write_mot:
        img_dir = osp.join(args.outdir, 'images')
        os.makedirs(img_dir, exist_ok=True)

    my_t = MyTransformCLS()
    my_t = my_t.to(args.device)
    my_t.eval()

    if args.prof:
        torch.cuda.cudart().cudaProfilerStart()

    pbar = tqdm(total=len(vid_rd), desc='tracking video', mininterval=10)
    fnum = 0
    header = 'frame,id,x1,y1,w,h,play,class,is_jumping,ori_fnum,dx,dy,tlen\n'
    results_wr.write(header)
    trk_writer = TrackWriter(args.outdir, enabled=args.write_mot)
    corner_fn = osp.join(args.outdir, 'corners.csv')
    corner_writer = CornerWriter(corner_fn, vid_info.width, vid_info.fps,
                                 args.det_thr)

    for fnum, raw_img in vid_rd:
        image = cv2.resize(raw_img, (exp.test_size[1],
                                     exp.test_size[0]))
        pbar.update()

        if fnum == first_fnum:
            print(f'Native video resolution w,h = {vid_info.width},{vid_info.height}, '
                  f'scale={scale_x:2.2f},{scale_y:2.2f}')
            print(f'Inference image shape {image.shape}')
            video_frame_fn = osp.join(args.outdir, f'{args.match_name}.png')
            cv2.imwrite(video_frame_fn, image)

        if not play_frames:
            play = 0
        elif fnum not in play_frames:
            raise
            continue
        else:
            play = play_frames[fnum]

        with torch.no_grad():
            # image shape = (h, w, c)
            # sample shape = (1, c, h, w)
            sample = my_t(image)

        # Run detector
        with nvtx_range('infer'):
            outputs, img_info = predictor.inference(sample, timer)

        if outputs[0] is None:
            corners = []
            img_dets = []
        else:
            outputs = outputs[0].cpu().numpy()

            # yolox output:
            #   [:4] -> bboxes
            #   [4:6] = scores
            #   [6] = full class
            detections = outputs[:, :7]

            # Scale detection back to native video resolution before passing to tracker
            detections[:, :4] /= np.array([scale_x, scale_y, scale_x, scale_y])

            # Separate out corners so that tracker only see people
            img_dets, corners = separate_corners(detections, exp.num_classes)

        if len(corners):
            corner_writer.update(fnum, corners)

        if len(img_dets):
            with nvtx_range('trk-update'):
                online_targets, raw_dets = tracker.update(img_dets, raw_img)

            with nvtx_range('results-wr'):
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_jumping = []
                online_nearfar = []
                for idx, trk in enumerate(online_targets):
                    trk_writer.write(trk, fnum)
                    tlwh = trk.tlwh
                    tid = trk.track_id
                    dxdy = trk.dxdy
                    tlen = trk.tracklet_len
                    nearfar = trk.cls
                    is_jumping = trk.jumping
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        csv_str = (f'{fnum},{tid},{int(tlwh[0]):d},{int(tlwh[1]):d},{int(tlwh[2]):d},'
                                   f'{int(tlwh[3]):d},{play},{nearfar},{is_jumping},{fnum},'
                                   f'{dxdy[0]:.4f},{dxdy[1]:.4f},{tlen:.4f}\n')
                        results_wr.write(csv_str)

                        # scale down to inference resolution in order to visualize
                        tlwh = tlwh * np.array([scale_x, scale_y, scale_x, scale_y])
                        dxdy = dxdy * np.array([scale_x, scale_y])
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(trk.score)
                        online_jumping.append(is_jumping)
                        online_nearfar.append(nearfar)

                # Write out raw detections
                bboxes = raw_dets['bboxes']
                scores = raw_dets['scores']
                classes = raw_dets['classes']
                jumps = raw_dets['jumping']
                for tlbr, score, cls, jumping in zip(bboxes, scores, classes, jumps):
                    x1, y1, x2, y2 = tlbr.astype(np.int32)
                    w = x2 - x1
                    h = y2 - y1
                    cls = int(cls)
                    jumping = int(jumping)
                    if score >= args.det_thr:
                        dets_str = (f'{fnum},-1,{x1},{y1},{w},{h},{score:0.2f},{play},{cls},{jumping}\n')
                        dets_wr.write(dets_str)

            timer.toc()
            if args.save_vid:
                with nvtx_range('plot'):
                    online_im = plot_tracking_mc(
                        image=image,
                        tlwhs=online_tlwhs,
                        obj_ids=online_ids,
                        jumping=online_jumping,
                        nearfar=online_nearfar,
                        num_classes=4,  # FIXME tracker.num_classes,
                        frame_id=fnum,
                        fps=1. / timer.average_time,
                        play_num=play,
                    )
        else:
            timer.toc()
            online_im = image

        if args.write_mot:
            img_fn = osp.join(img_dir, f'{fnum:06d}.jpg')
            cv2.imwrite(img_fn, raw_img)

        if args.save_vid:
            with nvtx_range('wr-vid'):
                vid_writer.write(online_im)

    trk_writer.finish()
    logger.info(f"Saved tracking results to {result_filename}")


def setup_volleyvision(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Support unified output dir
    setup_logger(args.outdir, filename="tracker.log")

    print(f'Max plays {args.max_plays}')
    print(f'Max frames {args.max_frames}')

    if not args.experiment_name:
        args.experiment_name = exp.exp_name


def main(exp, args):
    if args.trt:
        args.device_name = "gpu"
    args.device_name = "cuda" if (args.device_name == "gpu" or
                                  args.device_name == 'cuda') else "cpu"
    args.device = torch.device(args.device_name)

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
            ckpt_file = osp.join(args.outdir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file,
                          weights_only=False,
                          map_location=args.device)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        model.cuda()
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = args.ckpt
        assert osp.exists(
            trt_file
        ), f"TensorRT model {trt_file} is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device_name, args.fp16)
    current_time = time.localtime()
    with nvtx_range('vid-loop'):
        imageflow_demo(predictor, current_time, args, exp)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)
