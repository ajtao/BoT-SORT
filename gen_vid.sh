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
# Limit max plays
# > bash gen_vid.sh -p <maxplays>
#
# Latest:
# > bash gen_vid.sh -g 0 -t all-in-one -p 3 -v


export PYTHONPATH=$PWD:${PWD}/../vball_tracking:../player_id:../vball-mmdet:${PWD}/../PyTrackNet
export CUDA_VISIBLE_DEVICES=0

GPU=0
TAG="BotSORT_8_2"
MAXPLAYS=5

while getopts 'jg:m:t:p:h' opt; do
    case "$opt" in
	g)
	    GPU="$OPTARG"
	    echo "Set GPU to $GPU"
	    ;;

	m)
	    SINGLE_MATCH="$OPTARG"
	    echo "Will run just match $SINGLE_MATCH"
	    ;;

	t)
	    TAG="$OPTARG"
	    echo "Use tag $TAG"
	    ;;

	j)
	    DOJERSEYS="True"
	    ;;

	p)
	    MAXPLAYS="$OPTARG"
	    echo "Set MAXPLAYS to $MAXPLAYS"
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
MATCHES_0=( 20211218_rze_zaw)
MATCHES_0=( 20220723_poland_usa_right)
MATCHES_1=( 20211002_olemiss_florida 20211021_iowastate_texas)
MATCHES_1=( 20211124_usc_stanford  20211014_tcu_texas )

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

if [[ -v DOJERSEYS ]];
then
    AH_OPTS="--jumping-posadj --assign-canonical --detect-jerseys --backproject"
else
    AH_OPTS="--jumping-posadj --assign-canonical --id-players --canonical-id-frame-offset 0 --backproject --smooth-bev"
fi

MODELS=( yolox_x_fullcourt_v7_2)
MODELS=( yolox_x_tracked_players_v1baseline)
MODELS=( yolox_x_fullcourt_640p_v8_2_every)
MODELS=( yolox_x_fullcourt_v8_2)
EXP=yolox_x_fullcourt
CFG="-f ../ByteTrack/exps/example/mot/${EXP}.py"

PYTRACKNET_MODEL="romantic-bobcat_TrackJointTouch_77-v5-fr1_fix"
BOT_HPARAMS="--nms 0.65 --track_high_thresh 0.5 --new_track_thresh 0.6"
PYTRACKNET_WEIGHTS="/mnt/g/output/PyTrackNet/skill/${PYTRACKNET_MODEL}/latest.pt"
PYTRACKNET_OUTPUT="/mnt/g/output/PyTrackNet/eval/${PYTRACKNET_MODEL}"

for MODEL in "${MODELS[@]}"
do

    for MATCH in "${MATCHES[@]}"
    do
	echo "WORKING ON MATCH $MATCH"

	# Run Tracker ...
	TRK_VID="/mnt/g/output/BotSort/${TAG}/yolox_x_fullcourt/${MATCH}/end0.mp4"
	if test ! -f "$TRK_VID"; then
	    CKPT="/mnt/g/output/ByteTrack/YOLOX_outputs/${MODEL}/latest_ckpt.pth.tar"
	    CMD="python tools/vb_demo.py  --fp16 --fuse --match-name $MATCH --view end0 --ckpt $CKPT  $CFG \
	    	 --tag $TAG --max-plays $MAXPLAYS --start-pad 2 --end-pad 1 $XYWH $BOT_HPARAMS "
	    echo $CMD
	    $CMD
	    echo $CMD > /mnt/g/output/BotSort/${TAG}/${EXP}/${MATCH}/cmd.sh
	fi

	# Generate ball predictions
	BALL_VID="${PYTRACKNET_OUTPUT}/${MATCH}.mp4"
	if test ! -f "$BALL_VID"; then
	    pushd ../PyTrackNet
	    CMD="python auto_label.py --load_weights $PYTRACKNET_WEIGHTS --eval --eval_match $MATCH"
	    echo $CMD
	    $CMD
	    popd
	fi
	VIZVID="--viz-vid $BALL_VID"

	# Glue everything together ...
	TRK_CSV="/mnt/g/output/BotSort/${TAG}/${EXP}/${MATCH}/end0.csv"
	CMD="python ../vball_tracking/apply_heuristics.py --match-name $MATCH --tracking-csv $TRK_CSV --view end0 --tag $TAG $AH_OPTS $VIZVID"
	echo $CMD
	$CMD

    done
done
