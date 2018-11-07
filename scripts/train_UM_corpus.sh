DATA_PATH=/usr2/home/ruochenx/research/cross_emb/data/UM_Corpus;

export CUDA_VISIBLE_DEVICES=$(python ../../tools/test_cuda/test_cuda.py 6000 2>&1)
echo "Training on GPU $CUDA_VISIBLE_DEVICES"

mkdir -p exp/en
python scripts/train_UM_corpus.py \
    --save_dir exp/en \
    --vocab_file "$DATA_PATH"/en.vocab.txt \
    --train_prefix "$DATA_PATH"/en.sent.txt \
    --n-train-tokens 49903461;

# mkdir -p exp/zh
# python scripts/train_UM_corpus.py \
#     --save_dir exp/zh \
#     --vocab_file "$DATA_PATH"/zh.vocab.txt \
#     --train_prefix "$DATA_PATH"/zh.sent.txt \
#     --n-train-tokens 45060031;
