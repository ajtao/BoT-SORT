# Usage (can combine any of these on commandline)
#
# Set GPU to use
# > bash gen_vid.sh -g 0    # set GPU
#
# Run just this match
# > bash gen_vid.sh -g 0 -m 20211119_kansasstate_texas
#
# Set the tag
# > bash gen_vid.sh -t <sometag>
#
# Latest:
# > bash gen_vid.sh -g 0 -t all-in-one -p 3 -v


export PYTHONPATH=$PWD:${PWD}/../vball_tracking:../player_id:../vball-mmdet:${PWD}/../PyTrackNet:${PWD}/../ActionDet:${PWD}:../mmpose
export CUDA_VISIBLE_DEVICES=0

GPU=0
FORCE=""
TAG="export_test"
MAX_PLAYS=""



while getopts 'jfg:m:p:t:h' opt; do
    case "$opt" in
	f)
	    FORCE="FORCE"
	    echo "Force to regenerate all"
	    ;;

	g)
	    GPU="$OPTARG"
	    echo "Set GPU to $GPU"
	    ;;

	m)
	    SINGLE_MATCH="$OPTARG"
	    echo "Will run just match $SINGLE_MATCH"
	    ;;

	p)
	    MAX_PLAYS="--max-plays $OPTARG"
	    echo "Setting max-plays to $MAX_PLAYS"
	    ;;

	t)
	    TAG="$OPTARG"
	    echo "Use tag $TAG"
	    ;;

	j)
	    DOJERSEYS="True"
	    ;;

	?|h)
	    echo "Usage: $(basename $0) [-m match] [-g gpu]"
	    exit 1
	    ;;
    esac
done
shift "$(($OPTIND -1))"


MATCHES_1=( 20230306_uci_psu)
MATCHES_0=( 20230226_usc_stanford 20220128_ksu_texas )
 
 
if [[ -v SINGLE_MATCH ]];
then
    MATCHES_0=( $SINGLE_MATCH)
fi

ALL=("${MATCHES_0[@]}" "${MATCHES_1[@]}")
echo "m0 ${MATCHES_0[@]}"
echo "m1 ${MATCHES_1[@]}"

if [ $GPU == 0 ]
then
    MATCHES=("${MATCHES_0[@]}")
elif [ $GPU == 1 ]
then
    MATCHES=("${MATCHES_1[@]}")
    export CUDA_VISIBLE_DEVICES=1
else
    MATCHES=("${MATCHES_0[@]} ${MATCHES_1[@]}")
fi

echo "matches ${MATCHES[@]}"

AH_OPTS="--jumping-posadj --assign-canonical --canonical-id-frame-offset 0 --backproject --smooth-bev"

MODELS=( yolox_x_fullcourt_v7_2)
MODELS=( yolox_x_tracked_players_v1baseline)
MODELS=( yolox_x_fullcourt_640p_v8_2_every)
MODELS=( yolox_x_fullcourt_v8_2)
MODELS=( yolox_l_fullcourt_v5bytetrack-with-bad-touches-trt)
MODELS=( yolox_x_fullcourt_v5bytetrack-with-bad-touches-trt)

EXP=yolox_l_fullcourt
EXP=yolox_x_fullcourt
CFG="-f ../ByteTrack/exps/example/mot/${EXP}.py"

BOT_HPARAMS="--nms 0.65 --track_high_thresh 0.5 --new_track_thresh 0.6"

PYTRACKNET_MODEL="spiffy-shrew_TrackJointTouch_baseline"
PYTRACKNET_WEIGHTS="/mnt/f/output/PyTrackNet/skill/${PYTRACKNET_MODEL}/latest.pt"
PYTRACKNET_OUTPUT="/mnt/f/output/PyTrackNet/eval/${TAG}_${PYTRACKNET_MODEL}"


for MODEL in "${MODELS[@]}"
do
    BYTE_CKPT="/mnt/f/output/ByteTrack/YOLOX_outputs/${MODEL}/latest_ckpt.pth.tar"

    for _MATCH in "${MATCHES[@]}"
    do
	MATCH=$(basename $_MATCH)
	echo "WORKING ON MATCH $MATCH"

	# Run Tracker ...
	TRK_VID="/mnt/f/output/BotSort/${TAG}/${EXP}/${MATCH}/end0.mp4"
	TRK_CSV="/mnt/f/output/BotSort/${TAG}/${EXP}/${MATCH}/end0.csv"
	if test ! -f "${TRK_VID}${FORCE}"; then
	    CMD="python tools/vb_demo_fast.py --match-name $MATCH --view end0 \
                 --ckpt $BYTE_CKPT $CFG --tag $TAG --start-pad 2 --end-pad 1 $BOT_HPARAMS \
                 $MAX_PLAYS --fp16 --trt --no-reid"
	    echo $CMD
	    $CMD&
	fi
	
	# Generate ball predictions
	BALL_VID="${PYTRACKNET_OUTPUT}/${MATCH}.mp4"
	BALL_CSV="${PYTRACKNET_OUTPUT}/${MATCH}.csv"
	MATCH_DIR="/mnt/g/data/vball/squashed/squashed/${MATCH}"
	if test ! -f "${BALL_VID}${FORCE}"; then
	    CMD="python ../PyTrackNet/scripts/eval.py --task eval \
                 --load-weights $PYTRACKNET_WEIGHTS \
                 --match-dir $MATCH_DIR --tag $TAG --output-root /mnt/f/output"
	    echo $CMD
	    $CMD&
	fi
	wait

	# Just run heuristics to isolate 12 players
	HEUR_CSV="/mnt/f/output/heuristics/${TAG}/${MATCH}/end0.csv"
	CMD="python ../vball_tracking/apply_heuristics.py --match-name $MATCH \
             --tracking-csv $TRK_CSV \
             --view end0 --tag ${TAG}_just_tracks --task just_tracks $MAX_PLAYS"
	echo $CMD
	#$CMD
	
	# mmpose
	POSE_CFG=configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py
	POSE_CKPT=https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
	CSV_12P="/mnt/f/output/heuristics/${TAG}_just_tracks/${MATCH}/end0.csv"
	VID="${MATCH_DIR}/end0.mp4"
	POSE_CSV=/mnt/f/output/mmpose/${TAG}/botsort_${MATCH}_hrnet_w48_coco_256x192.csv
	if test ! -f "${POSE_CSV}${FORCE}"; then
	    pushd ../mmpose
	    CMD="python demo/top_down_video_demo_with_bot_trt.py $POSE_CFG --video-path $VID \
      		 --output-root /mnt/f/output/mmpose/${TAG} --tracking-csv $CSV_12P \
 		 --match $MATCH $MAX_PLAYS --save-vid"
	    echo $CMD
	    $CMD
	    popd
	fi

	# Now we can run ActionDetection
	AD_RUN=nonchalant-avocet
	AD_TAG=baseline
	EPOCH=118
	SKILL_WEIGHTS=/mnt/f/output/ActionDet/skill/${AD_RUN}_ActionEncoderV2_${AD_TAG}/${AD_TAG}_model_${EPOCH}.pt
	TOUCH_CSV="/mnt/f/output/PyTrackNet/skill-eval/${AD_RUN}_${TAG}/${MATCH}/touch.csv"
	CMD="python ../PyTrackNet/scripts/auto_label.py \
	   --temporal-eval --window-pad 2 --window-slide-div 1 \
	   --tag $TAG \
	   $MAX_PLAYS \
	   --eval-match $MATCH --pose-csv $POSE_CSV --ball-csv $BALL_CSV \
	   --action-weights $SKILL_WEIGHTS \
	   --load_weights $PYTRACKNET_WEIGHTS --vid-fn $BALL_VID"
	if test ! -f "${TOUCH_CSV}${FORCE}"; then
	    echo $CMD
	    $CMD
	fi

	# Final heuristics run
	HEUR_CSV="/mnt/f/output/heuristics/${TAG}/${MATCH}/end0.csv"

	if test ! -f "${HEUR_CSV}${FORCE}"; then
	    TRK_CSV="/mnt/f/output/BotSort/${TAG}/${EXP}/${MATCH}/end0.csv"
	    CMD="python ../vball_tracking/apply_heuristics.py --match-name $MATCH \
	      --tracking-csv $TRK_CSV \
              --view end0 --tag ${TAG} --jumping-posadj --assign-canonical \
	      --backproject --smooth-bev --task visualize \
 	      $MAX_PLAYS \
              --touch-csv $TOUCH_CSV \
              --show-bev-ball --ball-csv $BALL_CSV --id-players \
	      --viz-vid $BALL_VID"
	    echo $CMD
	    $CMD
	fi
    done
done
