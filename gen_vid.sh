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


export PYTHONPATH=$PWD:${PWD}/../vball_tracking:../player_id:../vball-mmdet
export CUDA_VISIBLE_DEVICES=0

GPU=0
TAG="BotSORT_trackv1"
MAXPLAYS=5

while getopts 'jg:m:t:p:hv' opt; do
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

	v)
	    VIZVID="True"
	    ;;

	?|h)
	    echo "Usage: $(basename $0) [-m match] [-g gpu]"
	    exit 1
	    ;;
    esac
done
shift "$(($OPTIND -1))"


MATCHES_0=( 20210919_kentucky_stanford 20211001_arizonastate_stanford)
MATCHES_0=( 20211002_olemiss_florida)
MATCHES_1=( )
MATCHES_1=( 20211124_usc_stanford  20211014_tcu_texas 20211209_louisville_florida)
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

MODELS=( yolox_x_fullcourt_v8_2)
MODELS=( yolox_x_fullcourt_v7_2)
MODELS=( yolox_x_tracked_players_v1baseline)
EXP=yolox_x_fullcourt
CFG="-f ../ByteTrack/exps/example/mot/${EXP}.py"

BOT_HPARAMS="--nms 0.65 --track_high_thresh 0.5 --new_track_thresh 0.6"

for MODEL in "${MODELS[@]}"
do

    for MATCH in "${MATCHES[@]}"
    do
	echo "WORKING ON MATCH $MATCH"

	# Run Tracker ...
	CKPT="/mnt/g/output/ByteTrack/YOLOX_outputs/${MODEL}/latest_ckpt.pth.tar"
	CMD="python tools/vb_demo.py  --fp16 --fuse --match-name $MATCH --view end0 --ckpt $CKPT  $CFG \
    	     --tag $TAG --max-plays $MAXPLAYS --start-pad 2 --end-pad 1 $XYWH $BOT_HPARAMS"
	echo $CMD
	$CMD
	echo $CMD > /mnt/g/output/BotSort/${TAG}/${EXP}/${MATCH}/cmd.sh

	# Run heuristics ...
	if [[ -v VIZVID ]];
	then
	    VIZVID="--viz-vid /mnt/g/output/PyTrackNet/autolabel/all-in-one_spotted-coucal_TrackJointTouch_77-v6-newval/${MATCH}.mp4"
	else
	    VIZVID=""
	fi
	TRK_CSV="/mnt/g/output/BotSort/${TAG}/${EXP}/${MATCH}/end0.csv"
	CMD="python ../vball_tracking/apply_heuristics.py --match-name $MATCH --tracking-csv $TRK_CSV --view end0 --tag $TAG $AH_OPTS $VIZVID"
	echo $CMD
	$CMD

    done
done
