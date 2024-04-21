
SUSHI_ROOT=/home/atao/devel/public/SUSHI/data/vball-short/test

for TAG in yolox_mot_ablation
do
    MODEL=tracking_player_id_${TAG}
    DET_FN=${TAG}
    DET_ROOT=/mnt/f/output/BotSort/${MODEL}/20220907_italy_france_left
    for i in {0..4}
    do
	DETS=${DET_ROOT}/play_${i}/dets.csv
	SUSHI_DIR=${SUSHI_ROOT}/play_${i}/det
	cp $DETS ${SUSHI_DIR}/${DET_FN}.txt
    done
done

