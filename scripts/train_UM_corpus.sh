PROCESSED_DATA_PATH=../data/UM/preprocessed;
RAW_DATA_PATH=../data/UM/raw;

# train with ELMo

export CUDA_VISIBLE_DEVICES=$(python ../../tools/test_cuda/test_cuda.py 6000 2>&1)
echo "Training on GPU $CUDA_VISIBLE_DEVICES"

# for LANG in en zh
for MY_LANG in en
do
    EXP_PATH=exp/bilm/"$MY_LANG"/;
    mkdir -p "$EXP_PATH"
    python scripts/train_UM_corpus.py \
        --save_dir "$EXP_PATH" \
        --vocab_file "$RAW_DATA_PATH"/dict.bilm."$MY_LANG".txt \
        --train_prefix "$RAW_DATA_PATH"/"$MY_LANG".sent.txt &> "$EXP_PATH"/run.log.txt &
    # | tee "$EXP_PATH"/run.log.txt
    # sleep 2000;
done

# # covert to hdf5
# LANG=zh;
# python bin/dump_weights.py \
#     --save_dir exp/"$LANG" \
#     --outfile exp/"$LANG"/weights.hdf5
