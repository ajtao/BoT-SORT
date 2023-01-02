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
TAG="v5_12k_frames"
MAX_PLAYS="--max-plays 5"



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


MATCHES_0=( 20210919_kentucky_stanford 20211001_arizonastate_stanford)
MATCHES_0=( 20211209_louisville_florida)
MATCHES_0=( 20210919_kentucky_stanford)
MATCHES_0=( 20211218_jas_rad)
MATCHES_0=( 20211218_rze_zaw 20220723_poland_usa_left)
MATCHES_0=( 20220723_poland_usa_left)
MATCHES_1=( 20211002_olemiss_florida 20211021_iowastate_texas)
MATCHES_1=( 20211124_usc_stanford  20211014_tcu_texas )

MATCHES_0=( /mnt/g/data/vball/ball/skill_touches_ball/20220908_poland_usa_left)
MATCHES_1=( /mnt/g/data/vball/ball/skill_touches_ball/20220911_brazil_slovenia_left)

MATCHES_0=(/mnt/g/data/vball/ball/skill_touches_ball/20*[0-4]_*)
MATCHES_1=(/mnt/g/data/vball/ball/skill_touches_ball/20*[5-9]_*)
MATCHES_0=(/mnt/g/data/vball/ball/skill_touches_ball/20220911_brazil_slovenia_left)
MATCHES_0=( 20220911_brazil_slovenia_left 20210919_kentucky_stanford)
MATCHES_1=( 20220908_poland_usa_left 20211002_olemiss_florida)
MATCHES_0=( 20220713_usa_serbia_left 20220716_turkiye_italy_right)
MATCHES_1=( 20220713_brazil_japan_right)
MATCHES_0=( 20220716_turkiye_italy_right)
MATCHES_0=( 20220713_brazil_japan_right 20220713_usa_serbia_left 20220713_usa_serbia_right 20220714_italy_china_left 20220714_italy_china_right 20220717_turkiye_serbia_left)
MATCHES_1=( 20220714_turkiye_thailand_left 20220714_turkiye_thailand_right 20220716_serbia_brazil_left 20220716_serbia_brazil_right 20220716_turkiye_italy_left 20220716_turkiye_italy_right 20220717_turkiye_serbia_right)
MATCHES_0=( 20221217_texas_louisville 20221215_texas_usd)
 
 
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

PYTRACKNET_MODEL="muscular-whale_TrackJointTouch_v9_30_55"
PYTRACKNET_WEIGHTS="/mnt/f/output/PyTrackNet/skill/${PYTRACKNET_MODEL}/latest.pt"
PYTRACKNET_OUTPUT="/mnt/f/output/PyTrackNet/eval/${PYTRACKNET_MODEL}"


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
	    CMD="python tools/vb_demo.py --match-name $MATCH --view end0 \
                 --ckpt $BYTE_CKPT $CFG --tag $TAG --start-pad 2 --end-pad 1 $BOT_HPARAMS \
                 $MAX_PLAYS --fp16 --trt"
	    # --fp16 --fuse --trt
	    echo $CMD
	    $CMD&
	    echo $CMD > /mnt/f/output/BotSort/${TAG}/${EXP}/${MATCH}/cmd.sh
	fi

	# Generate ball predictions
	BALL_VID="${PYTRACKNET_OUTPUT}/${MATCH}.mp4"
	BALL_CSV="${PYTRACKNET_OUTPUT}/${MATCH}.csv"
	if test ! -f "${BALL_VID}${FORCE}"; then
	    pushd ../PyTrackNet
	    CMD="python scripts/auto_label.py --load_weights $PYTRACKNET_WEIGHTS --eval \
	    	 --eval-match $MATCH $MAX_PLAYS"
	    echo $CMD
	    $CMD&
	    popd
	fi
	wait

	# Just run heuristics to isolate 12 players
	HEUR_CSV="/mnt/f/output/heuristics/${TAG}/${MATCH}/end0.csv"
	CMD="python ../vball_tracking/apply_heuristics.py --match-name $MATCH \
             --tracking-csv $TRK_CSV \
             --view end0 --tag ${TAG}_just_tracks --task just_tracks $MAX_PLAYS"
	echo $CMD
	$CMD

	# mmpose
	POSE_CFG=configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py
	POSE_CKPT=https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
	CSV_12P="/mnt/f/output/heuristics/${TAG}_just_tracks/${MATCH}/end0.csv"
	VID="/mnt/g/data/vball/squashed/squashed/${MATCH}/end0.mp4"
	POSE_CSV=/mnt/f/output/mmpose/${TAG}/botsort_${MATCH}_hrnet_w48_coco_256x192.csv
	if test ! -f "${POSE_CSV}${FORCE}"; then
	    pushd ../mmpose
	    CMD="python demo/top_down_video_demo_with_bot.py $POSE_CFG $POSE_CKPT --video-path $VID \
      		 --output-root /mnt/f/output/mmpose/${TAG} --tracking-csv $CSV_12P \
 		 --match $MATCH $MAX_PLAYS --save-vid"
	    echo $CMD
	    $CMD
	    popd
	fi

	# Now we can run ActionDetection
	pushd ../PyTrackNet
	AD_RUN=complex-cow
	AD_TAG=moredata_newbaseline
	AD_RUN=watchful-dogfish
	AD_TAG=all_trnval
	EPOCH=148
	BALL_CSV=/mnt/f/output/PyTrackNet/eval/muscular-whale_TrackJointTouch_v9_30_55/${MATCH}.csv
	SKILL_WEIGHTS=/mnt/f/output/ActionDet/skill/${AD_RUN}_ActionEncoderV2_${AD_TAG}/${AD_TAG}_model_${EPOCH}.pt
	VID_FN=/mnt/f/output/PyTrackNet/eval/muscular-whale_TrackJointTouch_v9_30_55/${MATCH}.mp4
	CMD="python scripts/auto_label.py \
	   --temporal-eval --window-pad 2 --window-slide-div 1 \
	   --tag $TAG \
	   $MAX_PLAYS \
	   --eval-match $MATCH --pose-csv $POSE_CSV --ball-csv $BALL_CSV \
	   --action-weights $SKILL_WEIGHTS \
	   --load_weights $PYTRACKNET_WEIGHTS --vid-fn $VID_FN "
	echo $CMD
	$CMD
	popd

	# Final heuristics run
	TOUCH_CSV="/mnt/f/output/PyTrackNet/skill-eval/${AD_RUN}/${MATCH}/touch.csv"
	VIZ_VID="/mnt/f/output/mmpose/v5_12k_frames/botsort_${MATCH}_hrnet_w48_coco_256x192.mp4"
	VIZ_VID="/mnt/f/output/PyTrackNet/skill-eval/${AD_RUN}/${MATCH}/${MATCH}_${AD_RUN}.mp4"
	VIZ_VID=/mnt/f/output/PyTrackNet/eval/muscular-whale_TrackJointTouch_v9_30_55/${MATCH}.mp4
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
	      --viz-vid $VIZ_VID"
	    echo $CMD
	    $CMD
	fi
    done
done
