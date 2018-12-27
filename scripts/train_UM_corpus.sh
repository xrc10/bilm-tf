PROCESSED_DATA_PATH=../data/UM/preprocessed;
RAW_DATA_PATH=../data/UM/raw;
ALIGN_DATA_PATH=../data/UM/align;

DEBUG=1;
RUN_NAME=bilm_v0.1
TRAIN_SUFFIX=''

if [ $DEBUG -gt 0 ]
then
    echo "Debug Mode";
    RUN_NAME="$RUN_NAME"_debug;
    TRAIN_SUFFIX='.small';
fi

# train with ELMo

export CUDA_VISIBLE_DEVICES=$(python ../../tools/test_cuda/test_cuda.py 6000 2>&1)
echo "Training on GPU $CUDA_VISIBLE_DEVICES"

# training the language model
# for MY_LANG in en
# do
#     EXP_PATH=exp/"$RUN_NAME"/"$MY_LANG"/;
#     mkdir -p "$EXP_PATH"
#     python train_LM.py \
#         --save_dir "$EXP_PATH" \
#         --vocab_file "$RAW_DATA_PATH"/dict.bilm."$MY_LANG".txt \
#         --train_prefix "$RAW_DATA_PATH"/"$MY_LANG".sent.txt"$TRAIN_SUFFIX" \
#          &> "$EXP_PATH"/run.log.txt;
#     # | tee "$EXP_PATH"/run.log.txt
#     # sleep 2000;
# done

# train the cross-lingual
EXP_PATH=exp/"$RUN_NAME"/en-zh/;
mkdir -p "$EXP_PATH"

python train_align.py \
        --save_dir "$EXP_PATH" \
        --src_vocab_file "$RAW_DATA_PATH"/dict.bilm.en.txt \
        --trg_vocab_file "$RAW_DATA_PATH"/dict.bilm.zh.txt \
        --src_train_prefix "$RAW_DATA_PATH"/en.sent.txt"$TRAIN_SUFFIX" \
        --trg_train_prefix "$RAW_DATA_PATH"/zh.sent.txt"$TRAIN_SUFFIX" \
        | tee "$EXP_PATH"/run.log.txt;
