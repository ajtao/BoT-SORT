import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import contextlib
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from loguru import logger
import numpy as np
import ffmpegcv

# from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, setup_logger
from yolox.utils.visualize import plot_tracking_mc

from tracker.vball_sort import VbSORT
from tracker.tracking_utils.timer import Timer

from vtrak.match_config import Match
from vtrak.config import cfg
from vtrak.dataloader import LoadVideo
from vtrak.court import BevCourt


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
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="cuda", type=str, help="device to run our model, can either be cpu or gpu")
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


class ToTensor(nn.Module):
    def __init__(self, device, mean, std, input_size):
        super().__init__()

        self.device = device
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.input_size = input_size
        self.resize = transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR)

    def forward(self, img):
        with nvtx_range('totensor'):
            x = self.to_tensor(img)
        with nvtx_range('norm'):
            x = self.normalize(x)
        with nvtx_range('resize'):
            x = self.resize(x)
        c, h, w = x.shape

        with nvtx_range('pad'):
            if (h, w) == self.input_size:
                padded = x
            else:
                padded = torch.zeros(self.input_size).to(self.device)
                r = min(self.input_size[0] / x.shape[0], self.input_size[1] / x.shape[1])
                padded[: int(x.shape[0] * r), : int(x.shape[1] * r)] = x

            padded = padded.to(self.device).unsqueeze(0).float()
        return padded


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
        self.trt = trt_file is not None
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            if self.fp16:
                x = x.half()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.to_tensor = ToTensor(device=device, mean=self.rgb_means, std=self.std,
                                  input_size=self.test_size)
        self.to_tensor = self.to_tensor.to(device)

    def inference(self, img, timer, dump_input=False):
        with nvtx_range('infer0'):
            img_info = {"id": 0}
            if isinstance(img, str):
                img_info["file_name"] = osp.basename(img)
                img = cv2.imread(img)
            else:
                img_info["file_name"] = None

        with nvtx_range('infer1'):
            height, width = img.shape[:2]
            img_info["height"] = height
            img_info["width"] = width
            img_info["raw_img"] = img
            ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
            img_info["ratio"] = ratio
            input = self.to_tensor(img)
            if self.fp16 and not self.trt:
                input = input.half()

        with torch.no_grad():
            with nvtx_range('Model'):
                timer.tic()
                if dump_input:
                    np.save(f'inp_{dump_input}.npy', img.cpu())
                outputs = self.model(input)

            with nvtx_range('decode'):
                if self.decoder is not None:
                    outputs = self.decoder(outputs, dtype=outputs.type())
                if dump_input:
                    np.save(f'oup_{dump_input}.npy', outputs.cpu())
                    import pdb; pdb.set_trace()
            with nvtx_range('post'):
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre, agnostic=True)
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
    start_time = time.time()

    frame_id = 0
    results_wr = open(result_filename, 'w')
    header = 'frame,id,x1,y1,w,h,play,class,is_jumping,ori_fnum,dx,dy,tlen\n'
    results_wr.write(header)

    print_flag = True
    last_play = -1
    if args.prof:
        torch.cuda.cudart().cudaProfilerStart()
    for frame_id, (vid_fnum, play_num, frame) in enumerate(dataloader):
        if play_num != last_play:
            logger.info(f'Start play {play_num} frame {vid_fnum}')
        last_play = play_num
        if frame_id % 100 == 0:
            cur_time = time.time()
            fps = frame_id / (cur_time - start_time)
            logger.info('Processing play {} frame {} ({:.2f} fps)'.format(
                play_num, vid_fnum, fps))

        if frame_id == 0:
            video_frame_fn = result_filename.replace('csv', 'png')
            cv2.imwrite(video_frame_fn, frame)

        if args.prof and frame_id == 200:
            torch.cuda.cudart().cudaProfilerStop()
            return

        # Run tracker
        frame_print = None

        # Detect objects
        with nvtx_range('infer'):
            outputs, img_info = predictor.inference(frame, timer, dump_input=frame_print)
            dets = outputs[0]
            # scale back to original image size
            scale_x = exp.test_size[1] / float(img_info['width'])
            scale_y = exp.test_size[0] / float(img_info['height'])

        if print_flag:
            w, h = img_info['width'], img_info['height']
            print(f'img_size w,h = {w},{h}, scale={scale_x:2.2f},{scale_y:2.2f}')
            print_flag = False

        if dets is not None:
            with nvtx_range('trk-update'):
                # scale bbox predictions according to image size
                outputs = outputs[0].cpu().numpy()
                detections = outputs[:, :7]
                detections[:, :4] /= np.array([scale_x, scale_y, scale_x, scale_y])
                online_targets = tracker.update(detections, img_info["raw_img"],
                                                frame_print=frame_print)

            with nvtx_range('results-wr'):
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
                        results_wr.write(csv_str)

            timer.toc()
            with nvtx_range('plot'):
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

        with nvtx_range('wr-vid'):
            vid_writer.write(online_im)

    logger.info(f"Saved tracking results to {result_filename}")


def setup_volleyvision(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Support unified output dir
    setup_logger(args.outdir, filename="tracker.log")
    result_filename = os.path.join(args.outdir, 'tracker.csv')
    output_video_path = osp.join(args.outdir, 'tracker.mp4')

    print(f'Max plays {args.max_plays}')
    print(f'Max frames {args.max_frames}')

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    dataloader = LoadVideo(args.play_vid)
    #vid_writer = cv2.VideoWriter(
    #    output_video_path, cv2.VideoWriter_fourcc(*"avc1"),
    #    dataloader.frame_rate, (dataloader.width, dataloader.height)
    #)
    vid_writer = ffmpegcv.noblock(ffmpegcv.VideoWriterNV,
                                  output_video_path,
                                  codec='hevc',
                                  fps=dataloader.frame_rate)
    court = BevCourt(args.match_name, args.view, args.outdir)
    return result_filename, vid_writer, court, dataloader


def main(exp, args):
    result_filename, vid_writer, court, dataloader = setup_volleyvision(args)

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
            ckpt_file = osp.join(args.outdir, "best_ckpt.pth.tar")
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
    with nvtx_range('vid-loop'):
        imageflow_demo(dataloader, predictor, current_time, args, result_filename,
                       vid_writer, court)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)
