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

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, setup_logger
from yolox.utils.visualize import plot_tracking_mc

from tracker.vball_sort import VbSORT, full_id_to_name
from tracker.tracking_utils.timer import Timer

from vtrak.match_config import Match
from vtrak.config import cfg
from vtrak.dataloader import LoadVideo
from vtrak.court import BevCourt


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=str, help="test img size (w,h)")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
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
    parser.add_argument("--no-reid", dest="with_reid", default=True, action='store_false', help="don't use reid model")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    parser.add_argument(
        "--unsquashed", action='store_true'
    )
    parser.add_argument(
        "--play-vid", default=None, help="name of a single video to evaluate"
    )
    parser.add_argument(
        "--match-name", default=None, help="match name"
    )
    parser.add_argument(
        "--view", default=None, help="view"
    )
    parser.add_argument(
        "--tag", default=None, help="tag outputdir"
    )
    parser.add_argument(
        "--max-plays", default=None, type=int, help="max plays"
    )
    parser.add_argument(
        "--start-pad", type=int, default=1,
    )
    parser.add_argument(
        "--end-pad", type=int, default=1,
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


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        print(f'BotSORT preproc test size {self.test_size}')
        print(f'BotSORT postproc conf_thresh {self.confthre}, nms thresh {self.nmsthre}')
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer, dump_input=False):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            if dump_input:
                np.save(f'inp_{dump_input}.npy', img.cpu())
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            if dump_input:
                np.save(f'oup_{dump_input}.npy', outputs.cpu())
                import pdb; pdb.set_trace()
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def imageflow_demo(dataloader, predictor, current_time, args, result_filename, vid_writer,
                   court):
    tracker = VbSORT(args, frame_rate=args.fps)

    # ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(tracker.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    timer = Timer()

    frame_id = 0
    results = ['frame,id,x1,y1,w,h,play,class,is_jumping,ori_fnum,dx,dy,tlen\n']
    print_flag = True

    last_play = -1
    for frame_id, (vid_fnum, play_num, frame) in enumerate(dataloader):
        if play_num != last_play:
            logger.info(f'Start play {play_num} frame {vid_fnum}')
        last_play = play_num
        if frame_id % 20 == 0:
            logger.info('Processing play {} frame {} ({:.2f} fps)'.format(
                play_num, vid_fnum, 1. / max(1e-5, timer.average_time)))

        if frame_id == 0:
            video_frame_fn = result_filename.replace('csv', 'png')
            cv2.imwrite(video_frame_fn, frame)

        # Run tracker
        frame_print = None
        """
        if vid_fnum >= 1870 and vid_fnum <= 1874:
            frame_print = vid_fnum
            cv2.imwrite(f'fr{vid_fnum}.png', frame)
        """

        # Detect objects
        outputs, img_info = predictor.inference(frame, timer, dump_input=frame_print)
        dets = outputs[0]
        scale = min(exp.test_size[0] / float(img_info['height']),
                    exp.test_size[1] / float(img_info['width']))

        if print_flag:
            w, h = img_info['width'], img_info['height']
            print(f'img_size w,h = {w},{h}, scale={scale}')
            print_flag = False

        if dets is not None:
            # scale bbox predictions according to image size
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale
            classes = outputs[:, 6]

            online_targets = tracker.update(detections, img_info["raw_img"],
                                            frame_print=frame_print)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_jumping = []
            online_nearfar = []
            for trk in online_targets:
                tlwh = trk.tlwh
                tid = trk.track_id
                tlwh = trk.tlwh
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
                    results.append(csv_str)

            # print(f'class: {classes} online_nearfar {online_nearfar}')
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

    with open(result_filename, 'w') as f:
        f.writelines(results)
    logger.info(f"Saved tracking results to {result_filename}")


def setup_volleyvision(args):
    if args.unsquashed:
        cfg.match_root = '/mnt/g/data/vball/matches'

    if args.tag is not None:
        result_root = osp.join(cfg.output_root, 'BotSort', args.tag, exp.exp_name, args.match_name)
    else:
        result_root = osp.join(cfg.output_root, 'BotSort', exp.exp_name, args.match_name)
    os.makedirs(result_root, exist_ok=True)

    setup_logger(result_root, filename="log.log")

    if args.play_vid:
        vid_basename = osp.splitext(osp.basename(args.play_vid))[0]
        result_filename = os.path.join(result_root, f'{vid_basename}.csv')
    else:
        result_filename = os.path.join(result_root, f'{args.view}.csv')

    print(f'Max plays {args.max_plays}')

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if args.tsize is not None:
        img_size = [int(x) for x in args.tsize.split(',')]
    else:
        img_size = None
    if args.play_vid:
        dataloader = LoadVideo(args.play_vid, img_size)
    else:
        mobj = Match(args.match_name,
                     args.view,
                     args.max_plays,
                     use_offset=False,
                     start_pad=args.start_pad,
                     end_pad=args.end_pad)
        try:
            pass
        except:
            print(f'ERROR: some problem reading in match {args.match_name} ... SKIPPING')
            raise

        dataloader = LoadVideo(mobj.vid_fn, img_size,
                               plays=mobj.plays)
    output_video_path = osp.join(result_root, f'{args.view}.mp4')
    vid_writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
        dataloader.frame_rate, (dataloader.width, dataloader.height)
    )
    court = BevCourt(args.match_name, args.view, result_root)
    return result_root, result_filename, vid_writer, court, dataloader


def main(exp, args):
    """
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
    """
    result_root, result_filename, vid_writer, court, dataloader = setup_volleyvision(args)

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
        trt_file = osp.join(result_root, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    imageflow_demo(dataloader, predictor, current_time, args, result_filename,
                   vid_writer, court)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)
