
TRT_CKPT="/mnt/h/output/best_models/bytetrack/yolox_x_fullcourt_v5bytetrack-with-bad-touches-trt-4090/model_trt.pth"
CFG="../ByteTrack/exps/example/mot/yolox_x_fullcourt.py"
CKPT_CFG="--ckpt $TRT_CKPT -f $CFG --fp16 --trt"

MATCH_ROOT=/mnt/h/data/vball/vid-plays/tracking_player_id
# MATCHES=($(ls -d ${MATCH_ROOT}/${1}*))
MATCH=20220907_italy_france_left
MATCHES=( "${MATCH_ROOT}/20220907_italy_france_left" )

TRACKER=botsort-reid
TRACKEVAL_DIR=/home/atao/devel/unified-id/TrackEval/data/trackers/mot_challenge/vball-short-train/${TRACKER}/data
mkdir -p $TRACKEVAL_DIR

echo "Matches: ${MATCHES[@]}"

for MATCH_DIR in "${MATCHES[@]}"
do
    MATCH=$(basename $MATCH_DIR)
    echo $MATCH

    # PLAYS=($(ls -d ${MATCH_DIR}/play_*))
    PLAYS=(play_0 play_1 play_2 play_3 play_4)
    PLAYS=(play_1 )

    # for PLAY_DIR in "${PLAYS[@]}"
    for PLAY_NAME in "${PLAYS[@]}"
    do
	PLAY_VID="${MATCH_DIR}/${PLAY_NAME}/${PLAY_NAME}.mp4"
	# PLAY_NAME=$(basename $PLAY_DIR)
	python tools/track_play.py $CKPT_CFG --tag $TRACKER --play-vid $PLAY_VID
	MATCH_PLAY=${MATCH}_${PLAY_NAME}
	DST_FN=${TRACKEVAL_DIR}/${MATCH_PLAY}.txt
	cp /mnt/h/output/BoT-SORT/${TRACKER}/20220907_italy_france_left/${PLAY_NAME}/task/gt/gt.txt $DST_FN
    done

done
