export PYTHONPATH=$PWD:${PWD}/../vball_tracking:../player_id:../vball-mmdet:${PWD}/../PyTrackNet:${PWD}/../ActionDet:${PWD}:../mmpose
export CUDA_VISIBLE_DEVICES=0

MATCH=20230126_rzk_bdk_1s

GPU=0
unset REID
unset TRT
unset GEN_BALL
while getopts 'bg:m:rth' opt; do
    case "$opt" in
	b)
	    GEN_BALL="TRUE"
	    echo "Will generate ball"
	    ;;
 
	g)
	    GPU="$OPTARG"
	    echo "Set GPU to $GPU"
	    ;;
 
	m)
	    MATCH="$OPTARG"
	    echo "Will process match $MATCH"
	    ;;

	r)
	    REID="TRUE"
	    echo "Enabling re-id"
	    ;;

	t)
	    TRT="TRUE"
	    echo "Using TRT"
	    ;;

	?|h)
	    echo "Usage: $(basename $0) [-m match] [-g gpu]"
	    exit 1
	    ;;
    esac
done
shift "$(($OPTIND -1))"

TAG=actdet-rugged
if [[ -v TRT ]];
then
    TRT_OPTS="--fp16 --trt"
    TAG=${TAG}-trt
else
    TRT_OPTS="--fp16 --fuse"
fi

if [[ -v REID ]];
then
    REID_ARGS=""
    TAG=${TAG}-reid
else
    REID_ARGS="--no-reid"
fi

BALL_TAG=$TAG

if [[ $GPU == 1 ]];
then
    echo "Using GPU 1"
    export CUDA_VISIBLE_DEVICES=1
fi

MATCH_DIR=/mnt/g/data/vball/matches/*/${MATCH}
MATCH_DIR=/mnt/g/data/vball/skill/vnl23/${MATCH}
RAW_VID=${MATCH_DIR}/end0.mp4
BYTETRACK_CKPT=/mnt/f/output/ByteTrack/YOLOX_outputs/yolox_x_fullcourt_v5bytetrack-with-bad-touches/latest_ckpt.pth.tar
BYTETRACK_CFG=../ByteTrack/exps/example/mot/yolox_x_fullcourt.py
PYTRACKNET_MODEL=rugged-rook_TrackNetImgNetStr2No_v14-win5-noinit
PYTRACKNET_CKPT=/mnt/h/output/PyTrackNet/skill/${PYTRACKNET_MODEL}/latest.pt

CMD="python tools/vb_demo_fast.py --unsquashed --match-name $MATCH --view end0 --no-dv --ckpt $BYTETRACK_CKPT -f $BYTETRACK_CFG --tag $TAG $TRT_OPTS $REID_ARGS --play-vid $RAW_VID"
TRACKING_CSV="/mnt/f/output/BotSort/${TAG}/yolox_x_fullcourt/${MATCH}/end0.csv"
if test ! -f "${TRACKING_CSV}"; then
    echo $CMD
    $CMD &
fi

if [[ -v GEN_BALL ]];
then
    CMD="python ../PyTrackNet/scripts/eval.py --task eval --load-weights $PYTRACKNET_CKPT --match-dir $MATCH_DIR --tag $BALL_TAG --output-root /mnt/f/output --thr 0.4"
    echo $CMD
    $CMD &
fi
BALL_CSV=/mnt/f/output/PyTrackNet/eval/${BALL_TAG}_${PYTRACKNET_MODEL}/${MATCH}.csv
BALL_VID=/mnt/f/output/PyTrackNet/eval/${BALL_TAG}_${PYTRACKNET_MODEL}/${MATCH}.mp4

wait

CMD="python ../vball_tracking/apply_heuristics.py --match-name $MATCH --tracking-csv $TRACKING_CSV --view end0 --tag $TAG --task just_tracks_unsquashed --ball-csv $BALL_CSV"
echo $CMD
$CMD

HEUR_CSV="/mnt/f/output/heuristics/${TAG}/${MATCH}/end0.csv"
POSE_CFG=../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py
POSE_CKPT=https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth

if [[ -v TRT ]];
then
    CMD="python ../mmpose/demo/top_down_video_demo_with_bot_trt.py $POSE_CFG --video-path $RAW_VID --output-root /mnt/f/output/mmpose/$TAG --tracking-csv $HEUR_CSV --match $MATCH --save-vid"
else
    CMD="python ../mmpose/demo/top_down_video_demo_with_bot.py     $POSE_CFG $POSE_CKPT --video-path $RAW_VID --output-root /mnt/f/output/mmpose/$TAG --tracking-csv $HEUR_CSV --match $MATCH --save-vid"
fi
echo $CMD
$CMD

POSE_CSV=/mnt/f/output/mmpose/${TAG}/botsort_${MATCH}_hrnet_w48_coco_256x192.csv
POSE_VID=/mnt/f/output/mmpose/${TAG}/botsort_${MATCH}_hrnet_w48_coco_256x192.mp4
ACT_MODEL=nonchalant-avocet
ACT_CKPT=/mnt/f/output/ActionDet/skill/${ACT_MODEL}_ActionEncoderV2_baseline/baseline_model_118.pt
#ACT_MODEL=fervent-beluga
#ACT_CKPT=/mnt/f/output/ActionDet/skill/${ACT_MODEL}_ActionEncoderV2_plus5vnl/plus5vnl_model_118.pt

CMD="python ../PyTrackNet/scripts/auto_label.py --unsquashed --temporal-eval --window-pad 2 --window-slide-div 1 --tag $TAG --eval-match ${MATCH} --pose-csv $POSE_CSV --ball-csv $BALL_CSV --action-weights $ACT_CKPT --load_weights $PYTRACKNET_CKPT --vid-fn $POSE_VID"
#echo $CMD
#$CMD

# final viz
TOUCH_CSV=/mnt/h/output/PyTrackNet/skill-eval/${ACT_MODEL}_${TAG}/${MATCH}/touch.csv
TOUCH_VID=/mnt/f/output/PyTrackNet/skill-eval/${ACT_MODEL}_${TAG}/${MATCH}/${MATCH}_${ACT_MODEL}.mp4

CMD="python ../vball_tracking/apply_heuristics.py --match-name ${MATCH} --tracking-csv $HEUR_CSV --view end0 --tag ${TAG} --jumping-posadj --assign-canonical --backproject --smooth-bev --task visualize-nodv --touch-csv ${TOUCH_CSV} --show-bev-ball --ball-csv $BALL_CSV --viz-vid $TOUCH_VID"
#echo $CMD
#$CMD
