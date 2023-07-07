from easydict import EasyDict
import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from tqdm import tqdm

import cv2
import torch
from loguru import logger
import numpy as np

from tools.vb_demo import Predictor
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, setup_logger
from yolox.utils.visualize import plot_tracking_mc

from tracker.conf_sort import ConfSORT
from tracker.vball_sort import VbSORT
from tracker.tracking_utils.timer import Timer

from vtrak.match_config import Match
from vtrak.config import cfg
from vtrak.dataloader import LoadVideo
from vtrak.track_utils import TrackWriter

from scipy.spatial.distance import cdist
from tracker import matching


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
    parser.add_argument("--conf_thresh", type=float, default=0.6, help="detection is confused if multiple tracks match with this IOU")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")
    parser.add_argument("--split-on-confusion", action='store_true')

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
        "--play-vid", required=True, default=None,
        help="name of a single video to evaluate"
    )
    parser.add_argument(
        "--tag", default=None, help="tag outputdir"
    )
    return parser


def run_detector(dataloader, predictor):
    timer = Timer()

    # results_wr = open(result_filename, 'w')
    # dets_wr = open(osp.join(result_root, 'dets.csv'), 'w')
    # header = 'frame,id,x1,y1,w,h,play,class,is_jumping,ori_fnum,dx,dy,tlen\n'
    # results_wr.write(header)
    frame_detections = {}

    for vid_fnum, play_num, frame in tqdm(dataloader, desc='run detector'):
        # Detect objects
        outputs, img_info = predictor.inference(frame, timer)
        dets = outputs[0]
        scale = min(exp.test_size[0] / float(img_info['height']),
                    exp.test_size[1] / float(img_info['width']))

        if dets is not None:
            # scale bbox predictions according to image size
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale
            frame_detections[vid_fnum] = detections, frame
            # header = 'frame,id,x1,y1,w,h,play,class,is_jumping,ori_fnum,dx,dy,tlen\n'
        else:
            frame_detections[vid_fnum] = None, frame

    return frame_detections


def run_tracker(detections, vid_writer, args):
    # detection is a dict[fnum] -> [dets]
    tracker = ConfSORT(args, frame_rate=args.fps)
    args.detect_confusions = True

    # Run tracker once and capture events and tracks
    tracks = defaultdict(dict)  # [tid][fnum]
    events = []  # [list]
    max_tid = -1
    for fnum, (dets, img) in tqdm(detections.items(), desc='run tracker'):
        if dets is not None:
            # run tracker
            online_targets, split_event = tracker.update(dets.copy(), img, fnum)

            # record the track
            for trk in online_targets:
                simple_trk = EasyDict()
                simple_trk.tlwh = trk.tlwh
                simple_trk.track_id = trk.track_id
                simple_trk.dxdy = trk.dxdy
                simple_trk.tracklet_len = trk.tracklet_len
                simple_trk.cls = trk.cls
                simple_trk.nearfar = trk.cls
                simple_trk.jumping = trk.jumping
                simple_trk.score = trk.score
                if fnum in trk.features:
                    simple_trk.features = trk.features[fnum]
                else:
                    simple_trk.features = None
                tracks[trk.track_id][fnum] = simple_trk
                if trk.track_id > max_tid:
                    max_tid = trk.track_id
            if split_event is not None:
                events.append(split_event)

            # Visualize
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_jumping = []
            online_nearfar = []
            for trk in online_targets:
                tlwh = trk.tlwh
                tid = trk.track_id
                nearfar = trk.cls
                is_jumping = trk.jumping
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(trk.score)
                    online_jumping.append(is_jumping)
                    online_nearfar.append(nearfar)

            online_im = plot_tracking_mc(
                image=img,
                tlwhs=online_tlwhs,
                obj_ids=online_ids,
                jumping=online_jumping,
                nearfar=online_nearfar,
                num_classes=tracker.num_classes,
                frame_id=fnum,
                fps=0,
                play_num=0
            )
        else:
            online_im = img

        vid_writer.write(online_im)

    # Split and rename tracks
    for event in events:
        assert len(event) == 3
        fnum_start, fnum_end, tids = event
        print(f'\nEvent: fnum_start {fnum_start} fnum end {fnum_end} tids {tids}')
        before_tracks = defaultdict(list)
        after_tracks = defaultdict(list)
        was_tid = {}  # table of what this tid used to be named
        for tid in tids:
            # new tid
            max_tid += 1
            new_tid = max_tid

            fnums = list(tracks[tid].keys())
            if fnums[0] >= fnum_start:
                was_tid[tid] = 'new'
                print(f' new trk {tid}')
            elif fnums[-1] <= fnum_end:
                was_tid[tid] = 'ends'
                print(f' {tid} ends')
            else:
                was_tid[new_tid] = tid
                print(f' {tid} splits into {new_tid}')

            for fnum, trk in tracks[tid].items():
                if fnum < fnum_start:
                    before_tracks[tid].append(trk)
                elif fnum > fnum_end:
                    if tid not in before_tracks:
                        # don't rename track if there's no before track
                        after_tracks[tid].append(trk)
                    else:
                        trk.track_id = new_tid
                        after_tracks[new_tid].append(trk)
                else:
                    # Within the event period, we discard the tracks.
                    # Eventually figure out how to reconnect these
                    #
                    # Should use discard_shortest() to throw away short bad tracks when
                    # we have more than 12 detections
                    #
                    # Should we eventually linearly interpolate between start/end and
                    # then find linear sum assignment best IOU for those detections?
                    pass

        def calc_reid_tracklet(tracklet):
            # tracklet = list of trk
            feats = np.array([trk.features for trk in tracklet if trk.features is not None])
            feats = np.average(feats, axis=0)
            return feats

        before_feats = [calc_reid_tracklet(trks) for trks in before_tracks.values()]
        after_feats = [calc_reid_tracklet(trks) for trks in after_tracks.values()]
        dists = embedding_distance(before_feats, after_feats)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=args.match_thresh)
        print(f'matches\n{matches}')
        print(f'dists\n{dists}')

        print('Connectivity after matching tracklets with reID:')
        for match in matches:
            idx_bef, idx_aft = match
            tid_bef = list(tids)[idx_bef]
            tid_aft = list(after_tracks.keys())[idx_aft]
            tid_aft_was = was_tid[tid_aft]
            print(f'  {tid_bef} connects to {tid_aft_was} [{tid_aft}]')


def embedding_distance(before, after, metric='cosine'):
    """
    :param before: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(before), len(after)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    after_features = np.asarray(after, dtype=float)
    before_features = np.asarray(before, dtype=float)

    cost_matrix = np.maximum(0.0, cdist(before_features, after_features, metric))
    return cost_matrix


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
    output_video_path = osp.join(result_root, 'tracked.mp4')
    vid_writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
        dataloader.frame_rate, (dataloader.width, dataloader.height)
    )
    return result_root, result_filename, vid_writer, dataloader


def main(exp, args):
    result_root, result_filename, vid_writer, dataloader = setup_volleyvision(args)

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
    detections = run_detector(dataloader, predictor)
    run_tracker(detections, vid_writer, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)
