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

TAG=stable-new-heur

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
BALL_TAG=stable-new-heur

if [[ $GPU == 1 ]];
then
    export CUDA_VISIBLE_DEVICES=1
fi

RAW_VID=/mnt/g/data/vball/matches/*/${MATCH}/end0.mp4
BYTETRACK_CKPT=/mnt/f/output/ByteTrack/YOLOX_outputs/yolox_x_fullcourt_v5bytetrack-with-bad-touches/latest_ckpt.pth.tar
BYTETRACK_CFG=../ByteTrack/exps/example/mot/yolox_x_fullcourt.py

CMD="python tools/vb_demo_unsquashed.py --match-name $MATCH --view end0 --unsquashed --ckpt $BYTETRACK_CKPT -f $BYTETRACK_CFG --tag $TAG $TRT_OPTS $REID_ARGS --path /mnt/f/output/vid_frames/${MATCH}"
echo $CMD
$CMD&

if [[ -v GEN_BALL ]];
then
    python ../PyTrackNet/scripts/auto_label.py --load_weights /mnt/f/output/PyTrackNet/skill/muscular-whale_TrackJointTouch_v9_30_55/latest.pt --unsquashed --eval --eval-match $MATCH --tag $BALL_TAG&
    echo
fi

wait

python ../vball_tracking/apply_heuristics.py --match-name $MATCH --tracking-csv /mnt/f/output/BotSort/${TAG}/yolox_x_fullcourt/${MATCH}/end0.csv --view end0 --tag $TAG --task just_tracks_unsquashed

HEUR_CSV="/mnt/f/output/heuristics/${TAG}/${MATCH}/end0.csv"
POSE_CFG=../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py
POSE_CKPT=https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth

if [[ -v TRT ]];
then
    python ../mmpose/demo/top_down_video_demo_with_bot_trt.py $POSE_CFG --video-path $RAW_VID --output-root /mnt/f/output/mmpose/$TAG --tracking-csv $HEUR_CSV --match $MATCH --save-vid
    echo
else
    python ../mmpose/demo/top_down_video_demo_with_bot.py     $POSE_CFG $POSE_CKPT --video-path $RAW_VID --output-root /mnt/f/output/mmpose/$TAG --tracking-csv $HEUR_CSV --match $MATCH --save-vid
    echo
fi
POSE_CSV=/mnt/f/output/mmpose/${TAG}/botsort_${MATCH}_hrnet_w48_coco_256x192.csv
BALL_CSV=/mnt/f/output/PyTrackNet/eval/${BALL_TAG}_muscular-whale_TrackJointTouch_v9_30_55/${MATCH}.csv
BALL_VID=/mnt/f/output/PyTrackNet/eval/${BALL_TAG}_muscular-whale_TrackJointTouch_v9_30_55/${MATCH}.mp4
POSE_VID=/mnt/f/output/mmpose/${TAG}/botsort_${MATCH}_hrnet_w48_coco_256x192.mp4
ACT_CKPT=/mnt/f/output/ActionDet/skill/complex-cow_ActionEncoderV2_moredata_newbaseline/moredata_newbaseline_model_148.pt
ACT_CKPT=/mnt/f/output/ActionDet/skill/stimulating-oriole_ActionEncoderV2_dropout/dropout_model_148.pt
ACT_CKPT=/mnt/f/output/ActionDet/skill/nonchalant-avocet_ActionEncoderV2_baseline/baseline_model_118.pt

python ../PyTrackNet/scripts/auto_label.py --unsquashed --temporal-eval --window-pad 2 --window-slide-div 1 --tag $TAG --eval-match ${MATCH} --pose-csv $POSE_CSV --ball-csv $BALL_CSV --action-weights $ACT_CKPT --load_weights /mnt/f/output/PyTrackNet/skill/muscular-whale_TrackJointTouch_v9_30_55/latest.pt --vid-fn $POSE_VID

