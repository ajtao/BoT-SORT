CKPT_CFG="--ckpt /mnt/f/output/ByteTrack/YOLOX_outputs/yolox_x_fullcourt_v5bytetrack-with-bad-touches/latest_ckpt.pth.tar -f ../ByteTrack/exps/example/mot/yolox_x_fullcourt.py --fp16 --trt"

MATCH_ROOT=/mnt/f/output/vid-plays/tracking_player_id
MATCHES=($(ls -d ${MATCH_ROOT}/${1}*))

for MATCH_DIR in "${MATCHES[@]}"
do
    MATCH=$(basename $MATCH_DIR)
    echo $MATCH

    PLAYS=($(ls -d ${MATCH_DIR}/play_*))
    echo "${PLAYS[@]}"

    for PLAY_DIR in "${PLAYS[@]}"
    do
	PLAY_VID="${PLAY_DIR}/*.mp4"
	python tools/track_play.py $CKPT_CFG --tag tracking_player_id --play-vid $PLAY_VID
    done

done
